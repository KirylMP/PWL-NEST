#!/usr/bin/env python3
"""Rescan the lower transition and independently integrate detected PWA cycles."""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/fm_am_minimal_zones/rest_refinement'))
from model import Model
from run import json_numpy

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('model_dir',type=Path)
    p.add_argument('--start',type=float,default=.5228);p.add_argument('--stop',type=float,default=.5236)
    p.add_argument('--points',type=int,default=17);p.add_argument('--seeds',type=int,default=64)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--closure-tol',type=float,default=1e-5)
    a=p.parse_args()
    if a.start>=a.stop or a.points<2 or a.seeds<4 or a.workers<1 or a.closure_tol<=0:p.error('Invalid scan settings')
    path=a.model_dir/'parameters.json'
    if not path.exists():path=a.model_dir/'candidate.json'
    data=json.loads(path.read_text())
    if data.get('case')!='fm':raise ValueError('Expected FM model')
    model=Model('fm',data.get('coordinate_family','mixed'),data['construction'],data.get('fm_lower_type','node'))
    x=model.constrain_events(data['parameters']);m=model.unpack(x);f=model.f
    bounds=f.phi(np.array([1e-12,f.CMAX]),m['shape'])
    def scan(I):
        I=float(I);cycles,diag=f.cycles(I,model.equilibrium(I,m),m['co'],m['padded'],bounds,n=a.seeds,tol=1e-9)
        results=[]
        def rhs(t,y):return m['co']@np.r_[1.,y[0],y[1],I,np.maximum(y[0]-m['padded'],0.)]
        for q in cycles:
            T=float(q['T']);y0=np.array([q['section'],q['s']])
            sol=solve_ivp(rhs,(0,T),y0,method='DOP853',rtol=2e-10,atol=2e-12,max_step=.01)
            initial=np.array([float(f.inverse(np.asarray(y0[0]),m)),y0[1]])
            end=np.array([float(f.inverse(np.asarray(sol.y[0,-1]),m)),sol.y[1,-1]])
            closure=float(np.linalg.norm(end-initial)) if sol.success and np.all(np.isfinite(end)) else None
            results.append(dict(stable=bool(q['stable']),Cmin=float(f.inverse(np.asarray(q['min']),m)),Cmax=float(f.inverse(np.asarray(q['max']),m)),T=T,log_multiplier=float(q['log_multiplier']),initial_z_h=y0.tolist(),closure_physical_norm=closure,closure_passed=bool(closure is not None and closure<a.closure_tol),integrator_success=bool(sol.success)))
        return dict(I=I,cycles=results,diagnostics=diag)
    rows=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for n,row in enumerate(pool.map(scan,np.linspace(a.start,a.stop,a.points)),1):
            rows.append(row)
            print(f'CHECK {n}/{a.points} I={row["I"]:.8f} cycles={len(row["cycles"])} closed={sum(q["closure_passed"] for q in row["cycles"])} diagnostics={row["diagnostics"]}',flush=True)
    result=dict(candidate=str(path.resolve()),settings=dict(start=a.start,stop=a.stop,points=a.points,seeds=a.seeds,workers=a.workers,closure_tol=a.closure_tol),method='Finite return-map scan; independent fixed-period DOP853 integration checks physical closure. Stability is from the original return map. Non-detection does not prove absence.',rows=rows)
    out=a.model_dir/'small_cycle_check.json';out.write_text(json.dumps(result,indent=2,default=json_numpy,allow_nan=False)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for row in rows:
        for q in row['cycles']:
            color='black' if q['closure_passed'] else 'red'
            kw=dict(s=20,edgecolors=color,facecolors=color if q['stable'] else 'none')
            axes[0].scatter([row['I']]*2,[q['Cmin'],q['Cmax']],**kw)
            axes[1].scatter(row['I'],q['T'],**kw)
    for ax in axes:ax.set_xlabel('I (µM)');ax.grid(alpha=.2)
    axes[0].set_ylabel('C extrema (µM)');axes[1].set_ylabel('Period (s)')
    fig.suptitle('PWA cycles: black = closure passed, red = failed; filled = stable by return map')
    fig.savefig(a.model_dir/'small_cycle_check.png',dpi=150);plt.close(fig)
    print(f'Saved {out} and {a.model_dir / "small_cycle_check.png"}',flush=True)

if __name__=='__main__':main()
