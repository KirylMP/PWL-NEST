#!/usr/bin/env python3
"""Finite return-map scan of LR and a fitted PWA near the lower FM transition."""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/fm_am_minimal_zones/rest_refinement'))
from model import Model
from run import json_numpy


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('model_dir',type=Path)
    ap.add_argument('--points',type=int,default=int(os.environ.get('LOWER_POINTS','81')))
    ap.add_argument('--seeds',type=int,default=int(os.environ.get('LOWER_SEEDS','48')))
    ap.add_argument('--workers',type=int,default=int(os.environ.get('LOWER_WORKERS','4')))
    ap.add_argument('--start',type=float,default=float(os.environ.get('LOWER_I_START','0.525')))
    ap.add_argument('--stop',type=float,default=float(os.environ.get('LOWER_I_STOP','0.534')))
    args=ap.parse_args()
    if args.points<2 or args.seeds<4 or args.workers<1 or args.start>=args.stop:ap.error('Invalid grid or worker settings')
    path=args.model_dir/'candidate.json'
    if not path.exists():path=args.model_dir/'parameters.json'
    data=json.loads(path.read_text())
    count=len(data['physical_thresholds_C'])
    if data.get('case')!='fm' or count not in (3,4):raise ValueError('Expected 3- or split 4-threshold FM candidate')
    construction=data.get('construction','split' if count==4 else 'base')
    if construction not in ('base','split','split_fixed'):raise ValueError('Unsupported FM construction')
    model=Model('fm','mixed',construction,data.get('fm_lower_type','node'))
    x=model.constrain_events(data['parameters']);m=model.unpack(x);f=model.f
    bounds=f.phi(np.array([1e-12,f.CMAX]),m['shape'])
    grid=np.unique(np.r_[np.linspace(args.start,args.stop,args.points),model.event_I[0]])
    grid=grid[(grid>=args.start)&(grid<=args.stop)]
    def scan(I):
        I=float(I);lreq=f.eqs(I);pwaeq=model.equilibrium(I,m)
        lr,ld=f.cycles(I,lreq,n=args.seeds,tol=1e-9)
        pw,pd=f.cycles(I,pwaeq,m['co'],m['padded'],bounds,n=args.seeds,tol=1e-9)
        for cycle in pw:
            cycle['Cmin'],cycle['Cmax']=map(float,f.inverse(np.asarray([cycle['min'],cycle['max']]),m))
        for cycle in lr:cycle['Cmin'],cycle['Cmax']=float(cycle['min']),float(cycle['max'])
        return dict(I=I,lr=lr,pwa=pw,lr_diagnostics=ld,pwa_diagnostics=pd,
                    lr_equilibria=[e.tolist() for e in lreq],pwa_equilibria=pwaeq)
    rows=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for j,row in enumerate(pool.map(scan,grid),1):
            rows.append(row)
            if j==1 or j%5==0 or j==len(grid):
                print(f'LOWER {j}/{len(grid)} I={row["I"]:.9f} LR={len(row["lr"])} PWA={len(row["pwa"])}',flush=True)
    result=dict(method='finite return-map scan in both time directions, on equilibrium sections; no continuation or completeness proof',
                candidate_stage=data.get('stage'),candidate=str(path.resolve()),points=len(grid),seeds=args.seeds,
                grid_start=args.start,grid_stop=args.stop,rows=rows)
    (args.model_dir/'lower_bifurcation.json').write_text(json.dumps(result,indent=2,default=json_numpy)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
    for key,color,label in [('lr','tab:purple','LR'),('pwa','black','PWA')]:
        for row in rows:
            for q in row[key]:
                kw=dict(s=14,edgecolors=color,facecolors=color if q['stable'] else 'none',linewidths=.8)
                axes[0].scatter([row['I']]*2,[q['Cmin'],q['Cmax']],**kw)
                axes[1].scatter(row['I'],q['T'],**kw)
        for stable,style in [(True,'-'),(False,'--')]:
            axes[2].step([r['I'] for r in rows],[sum(q['stable']==stable for q in r[key]) for r in rows],
                         where='mid',color=color,linestyle=style,label=f'{label} '+('stable' if stable else 'unstable'))
    for ax in axes:ax.axvline(model.event_I[0],color='gray',linestyle=':',alpha=.8);ax.set_xlabel('I (µM)');ax.grid(alpha=.25)
    axes[0].set_ylabel('Cycle extrema C (µM)');axes[1].set_ylabel('Period (s)');axes[2].set_ylabel('Found cycles');axes[2].legend()
    fig.suptitle('Lower FM transition: filled = stable, open = unstable; finite scan')
    fig.savefig(args.model_dir/'lower_bifurcation.png',dpi=160);plt.close(fig)
    print(f'LOWER saved {args.model_dir}/lower_bifurcation.json and lower_bifurcation.png',flush=True)


if __name__=='__main__':main()
