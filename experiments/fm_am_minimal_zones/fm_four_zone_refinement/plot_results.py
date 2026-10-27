#!/usr/bin/env python3
"""Scan saved FM candidates and LR on the same I grid and plot the results."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(name,'1')
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from model import Model, reference as f

ROOT=Path(__file__).resolve().parent

def json_numpy(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return value.tolist()
    raise TypeError(f'Cannot serialize {type(value).__name__}')

VARIANTS=('base','region1','region2','region3','all')
LABELS={'base':'3 thresholds','region1':'extra in region 1','region2':'extra in region 2','region3':'extra in region 3','all':'extra in all 4 regions'}


def scan(task):
    name,I,parameters,seeds,tol=task
    if name=='LR':
        states=f.eqs(I)
        equilibria=[dict(C=float(y[0]),z=[float(y[0]),float(y[1])],
            stable=bool(np.max(np.linalg.eigvals(f.jac(y,I)).real)<0)) for y in states]
        cycles,diag=f.cycles(I,states,n=seeds,tol=tol)
        for q in cycles:q['Cmin']=float(q['min']);q['Cmax']=float(q['max'])
    else:
        model=Model(name);m=model.unpack(np.asarray(parameters,dtype=float))
        equilibria=model.equilibrium(I,m)
        bounds=f.phi(np.array([1e-12,f.CMAX]),m['shape'])
        cycles,diag=f.cycles(I,equilibria,m['co'],m['padded'],bounds,n=seeds,tol=tol)
        for q in cycles:
            extrema=f.inverse(np.array([q['min'],q['max']]),m)
            q['Cmin']=float(extrema[0]);q['Cmax']=float(extrema[1])
    return dict(model=name,I=float(I),equilibria=equilibria,cycles=cycles,diagnostics=diag)


def render(axes, rows, lr, cuts, label, xmin, xmax):
    a,b,c=axes
    for data,color,alpha,size in [(lr,'#a855c7',.65,9),(rows,'#111111',.8,13)]:
        for stable in (True,False):
            items=[(r['I'],q) for r in data for q in r['cycles'] if bool(q['stable'])==stable]
            if not items:continue
            for key in ('Cmin','Cmax'):
                a.scatter([x for x,q in items],[q[key] for x,q in items],s=size,
                    facecolors=color if stable else 'none',edgecolors=color,lw=.65,alpha=alpha)
            b.scatter([x for x,q in items],[q['T'] for x,q in items],s=size,
                facecolors=color if stable else 'none',edgecolors=color,lw=.65,alpha=alpha)
    for data,color,alpha in [(lr,'#a855c7',.6),(rows,'#111111',.8)]:
        for stable in (True,False):
            pairs=[(r['I'],e['C']) for r in data for e in r['equilibria'] if e['stable']==stable]
            if pairs:a.scatter(*zip(*pairs),s=5,marker='.' if stable else 'x',color=color,alpha=alpha)
    for cut in cuts:a.axhline(cut,color='#0ea5e9',ls='--',lw=.6,alpha=.55)
    for data,col,label_prefix in [(lr,'#a855c7','LR'),(rows,'#111111','model')]:
        for stable,style in [(True,'-'),(False,'--')]:
            c.step([r['I'] for r in data],[sum(bool(q['stable'])==stable for q in r['cycles']) for r in data],
                   where='mid',color=col,ls=style,lw=1.3,label=f'{label_prefix} {"stable" if stable else "unstable"}')
    a.set(title=f'{label}: equilibria and cycle extrema',ylabel=r'$C$ ($\mu$M)',xlim=(xmin,xmax))
    b.set(title='Cycle periods',ylabel='Period (s)',xlim=(xmin,xmax));b.set_ylim(bottom=0)
    c.set(title='Detected cycles',ylabel='Number of cycles',xlim=(xmin,xmax));c.set_ylim(bottom=-.1)
    for ax in axes:
        ax.set_xlabel(r'$I$ ($\mu$M)');ax.grid(alpha=.16);ax.spines[['top','right']].set_visible(False)
    c.legend(fontsize=7,loc='best')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grid-size',type=int,default=81,help='number of equally spaced I values')
    p.add_argument('--i-min',type=float,default=.4);p.add_argument('--i-max',type=float,default=1.25)
    p.add_argument('--seeds',type=int,default=48,help='return-map heights per direction')
    p.add_argument('--tol',type=float,default=1e-9);p.add_argument('--workers',type=int,default=16)
    args=p.parse_args()
    if args.grid_size<2 or args.seeds<4 or args.workers<1 or not (np.isfinite(args.i_min) and np.isfinite(args.i_max) and 0<=args.i_min<args.i_max) or not (np.isfinite(args.tol) and args.tol>0):p.error('invalid grid, range, seed count, worker count, or tolerance')
    candidates={};hashes={}
    for name in VARIANTS:
        path=ROOT/'results'/name/'candidate.json'
        if not path.exists():raise SystemExit(f'Missing candidate: {path}')
        raw=path.read_bytes();c=json.loads(raw)
        if c.get('stage')!='closed' or c.get('variant')!=name:raise SystemExit(f'Candidate is not a finished closed-stage {name} model: {path}')
        model=Model(name)
        if not np.allclose(c['physical_thresholds_C'],model.cuts,rtol=0,atol=1e-12):raise SystemExit(f'Candidate thresholds do not match {name}: {path}')
        model.unpack(np.asarray(c['parameters'],dtype=float))
        candidates[name]=c;hashes[name]=hashlib.sha256(raw).hexdigest()
    grid=np.linspace(args.i_min,args.i_max,args.grid_size)
    config=dict(candidate_sha256=hashes,grid=grid.tolist(),seeds=args.seeds,tol=args.tol,method_version=1)
    key=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
    out=ROOT/'results'/'plots'/key;out.mkdir(parents=True,exist_ok=True)
    cache=out/'scan.jsonl';completed={}
    if cache.exists():
        with cache.open() as src:
            for line_no,line in enumerate(src,1):
                try:row=json.loads(line)
                except json.JSONDecodeError:raise SystemExit(f'Invalid cached row {line_no} in {cache}; move this file aside and rerun')
                completed[(row['model'],float(row['I']))]=row
    tasks=[(name,float(I),None if name=='LR' else candidates[name]['parameters'],args.seeds,args.tol)
           for name in ('LR',*VARIANTS) for I in grid if (name,float(I)) not in completed]
    print(f'Output: {out}\nGrid: {len(grid)} points, seeds: {args.seeds}, workers: {args.workers}; cached: {len(completed)}, remaining: {len(tasks)}',flush=True)
    with cache.open('a') as dst, ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(scan,t):t for t in tasks}
        for index,future in enumerate(as_completed(futures),1):
            task=futures[future]
            try:row=future.result()
            except Exception as e:raise RuntimeError(f'Scan failed for model={task[0]}, I={task[1]:.12g}') from e
            dst.write(json.dumps(row,allow_nan=False,default=json_numpy)+'\n');dst.flush()
            completed[(row['model'],row['I'])]=row
            if index==1 or index%10==0 or index==len(tasks):
                print(f'Scan {index}/{len(tasks)}: {row["model"]} I={row["I"]:.6g} cycles={len(row["cycles"])} diagnostics={row["diagnostics"]}',flush=True)
    rows={name:[completed[(name,float(I))] for I in grid] for name in ('LR',*VARIANTS)}
    plt.rcParams.update({'font.size':9,'figure.dpi':120})
    fig,axes=plt.subplots(5,3,figsize=(16,21),constrained_layout=True)
    for j,name in enumerate(VARIANTS):
        render(axes[j],rows[name],rows['LR'],candidates[name]['physical_thresholds_C'],LABELS[name],args.i_min,args.i_max)
        one,one_axes=plt.subplots(1,3,figsize=(16,4.6),constrained_layout=True)
        render(one_axes,rows[name],rows['LR'],candidates[name]['physical_thresholds_C'],LABELS[name],args.i_min,args.i_max)
        one.suptitle('Black: model; purple: Li–Rinzel. Filled: stable cycle; open: unstable. Dots/x: stable/unstable equilibria.')
        one.savefig(out/f'{name}_bifurcation.png',dpi=180);one.savefig(out/f'{name}_bifurcation.pdf');plt.close(one)
    fig.suptitle('FM threshold refinements vs Li–Rinzel | Black: model; purple: LR | filled/open: stable/unstable cycles',fontsize=13)
    fig.savefig(out/'all_bifurcations.png',dpi=150);fig.savefig(out/'all_bifurcations.pdf');plt.close(fig)
    summary=dict(config=config,generated_utc=datetime.now(timezone.utc).isoformat(),source_candidates={name:str(ROOT/'results'/name/'candidate.json') for name in VARIANTS},
        caveat='Finite-grid return-map scan; absence of a detected cycle is not proof of absence. No independent orbit verification.',
        models={name:dict(points=len(rows[name]),max_detected_cycles=max(len(r['cycles']) for r in rows[name]),
           timeouts=sum(r['diagnostics'].get('timeouts',0) for r in rows[name]),
           unresolved=sum(r['diagnostics'].get('unresolved',0) for r in rows[name]),
           no_equilibrium_sections=sum(not r['equilibria'] for r in rows[name])) for name in rows})
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary['models'],indent=2),flush=True)
    print(f'Plots and scan data saved in {out}',flush=True)

if __name__=='__main__':main()
