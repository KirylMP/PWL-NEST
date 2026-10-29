#!/usr/bin/env python3
"""Run independent K fits and summarize every completed model."""
import argparse, concurrent.futures, json, os, subprocess, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--values',type=float,nargs='+',default=[.05,.06,.07,.08,.09,.10]);p.add_argument('--workers',type=int,default=6);p.add_argument('--starts',type=int,default=4);p.add_argument('--maxiter',type=int,default=1200);p.add_argument('--itinerary',choices=['both','auto'],default='both');p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if a.workers<1 or a.starts<1 or a.maxiter<1:p.error('workers, starts and maxiter must be positive')
    jobs=[(K,kind) for K in a.values for kind in ('fm','am')]
    def job(job):
        K,kind=job;d=a.out/f'K_{K:.4f}_{kind}';d.mkdir(parents=True,exist_ok=True);env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        cmd=[sys.executable,'-u',str(HERE/'fit_k.py'),'--K',str(K),'--kind',kind,'--starts',str(a.starts),'--maxiter',str(a.maxiter),'--itinerary',a.itinerary,'--out',str(d)]
        with (d/'fit.log').open('w') as log:code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env).returncode
        f=d/'parameters.json';info=json.loads(f.read_text()) if f.exists() else dict(K=K,kind=kind,status='process_error')
        print(f'K={K:.4f} {kind} status={info["status"]} exit={code} log={d/"fit.log"}',flush=True);return info
    print(f'Running {len(jobs)} independent models with {a.workers} workers; output={a.out}',flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:rows=list(pool.map(job,jobs))
    (a.out/'summary.json').write_text(json.dumps(dict(rows=rows,warning='Only verified_local points plotted; no global orbit completeness or uniqueness proof. kind is regional-type construction, not a demonstrated AM/FM classification.'),indent=2)+'\n')
    fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    labels=[('a','a'),('b','b'),('loss','vector field loss'),('fold_amplitude','fold amplitude'),('thresholds','thresholds C'),('traces','regional traces')]
    for kind in ('fm','am'):
        group=sorted([r for r in rows if r['kind']==kind and r['status']=='verified_local'],key=lambda r:r['K']);ks=[r['K'] for r in group]
        if not group:continue
        for ax,(key,label) in zip(axs.flat,labels):
            if key in ('a','b','loss'):ax.plot(ks,[r[key] for r in group],'.-',label=kind)
            elif key=='fold_amplitude':ax.plot(ks,[r['fold']['sampled_amplitude'] for r in group],'.-',label=kind);ax.plot(ks,[r['LR_target']['amplitude'] for r in group],':',label='LR '+kind)
            elif key=='thresholds':
                for j in range(2):ax.plot(ks,[r['cuts'][j] for r in group],'.-',label=f'{kind} c{j+1}')
            else:
                for j in range(3):ax.plot(ks,[r['trace'][j] for r in group],'.-',label=f'{kind} T{j+1}')
    for ax,(_,label) in zip(axs.flat,labels):ax.set_title(label);ax.set_xlabel('K_ER (µM)');ax.grid(alpha=.25);ax.legend(fontsize=7)
    fig.suptitle('Two-threshold models: locally verified candidates');fig.savefig(a.out/'parameter_comparison.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(2,5,figsize=(17,7),layout='constrained')
    for kind in ('fm','am'):
        group=sorted([r for r in rows if r['kind']==kind and r['status']=='verified_local'],key=lambda r:r['K'])
        if not group:continue
        for j in range(5):
            for row,key in enumerate(('alpha','beta')):
                axs[row,j].plot([r['K'] for r in group],[r[key][j] for r in group],'.-',label=kind)
    for row,key in enumerate(('alpha','beta')):
        for j in range(5):
            ax=axs[row,j];ax.set_title(f'{key}[{j}]');ax.set_xlabel('K_ER (µM)');ax.grid(alpha=.25)
            if ax.has_data():ax.legend(fontsize=7)
    fig.suptitle('PWA coefficients in the same physical-C, identity-q coordinates');fig.savefig(a.out/'coefficients_vs_k.png',dpi=150);plt.close(fig)
    print(f'SUMMARY {a.out/"summary.json"}; graph {a.out/"parameter_comparison.png"}',flush=True)
if __name__=='__main__':main()
