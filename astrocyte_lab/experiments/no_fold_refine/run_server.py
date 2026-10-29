#!/usr/bin/env python3
"""Refine top randomly initialized candidates from an existing no-fold run."""
import argparse,concurrent.futures,json,os,shutil,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]

def main():
 p=argparse.ArgumentParser();p.add_argument('--prior-run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--top',type=int,default=5);p.add_argument('--workers',type=int,default=15);p.add_argument('--powell-maxfev',type=int,default=2500);p.add_argument('--ls-max-nfev',type=int,default=300);p.add_argument('--fd-relative',type=float,default=.001);p.add_argument('--duration',type=float,default=350);p.add_argument('--validation-duration',type=float,default=450);p.add_argument('--dt',type=float,default=.01);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 if min(a.top,a.workers,a.powell_maxfev,a.ls_max_nfev)<1 or min(a.duration,a.validation_duration)<100 or a.dt<=0 or a.fd_relative<=0:raise ValueError('Invalid settings')
 prior=a.prior_run.resolve();prev=json.loads((prior/'summary.json').read_text());env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR=str(a.out/'mpl'),XDG_CACHE_HOME=str(a.out/'cache'));engine=a.out/'shape_engine.so';subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(HERE/'shape_engine.cpp'),'-o',str(engine)],check=True)
 ks=[.05,.075,.1];tasks=[]
 for k in ks:
  options=sorted([r for r in prev['trials'] if r['K']==k and r['status']=='training_cycles_found' and r['loss'] is not None],key=lambda r:r['loss'])[:a.top]
  for row in options:
   path=prior/f'K_{k:.3f}'/f'trial_{row["trial"]:03d}'/'parameters.json'
   if not path.exists():raise FileNotFoundError(path)
   tasks.append((k,row['trial'],path))
 print(f'REFINE {len(tasks)} seeds, workers={a.workers}, source={prior}, output={a.out}',flush=True)
 def job(item):
  k,j,seed=item;d=a.out/f'K_{k:.3f}'/f'trial_{j:03d}';d.mkdir(parents=True,exist_ok=True);log=d/'fit.log';cmd=[sys.executable,'-u',str(HERE/'refine.py'),'--seed-model',str(seed),'--engine',str(engine.resolve()),'--out',str(d),'--powell-maxfev',str(a.powell_maxfev),'--ls-max-nfev',str(a.ls_max_nfev),'--fd-relative',str(a.fd_relative),'--duration',str(a.duration),'--dt',str(a.dt)]
  with log.open('w') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
  path=d/'parameters.json';r=json.loads(path.read_text()) if path.exists() else {};item=dict(K=k,source_trial=j,exit_code=p.returncode,status=r.get('status','process_error'),initial_loss=r.get('initial_loss'),loss=r.get('loss'),path=str(path),log=str(log),optimizers=r.get('optimizers'));print('TRIAL',json.dumps(item),flush=True);return item
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:items=list(pool.map(job,tasks))
 winners={};validation={}
 for k in ks:
  group=[r for r in items if r['K']==k and r['status']=='training_cycles_found' and r['loss'] is not None]
  if not group:print('NO CANDIDATE',k,flush=True);continue
  winner=min(group,key=lambda r:r['loss']);winners[str(k)]=winner;dest=a.out/f'K_{k:.3f}'/'model';dest.mkdir(parents=True,exist_ok=True);shutil.copy2(winner['path'],dest/'parameters.json');log=dest/'validate.log';print('VALIDATE',k,'log=',log,flush=True)
  cmd=[sys.executable,'-u',str(ROOT/'astrocyte_lab/experiments/no_fold_three_fits/validate.py'),str(dest/'parameters.json'),'--duration',str(a.validation_duration)]
  with log.open('w') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
  record=dict(exit_code=p.returncode,path=str(dest/'validation.json'),plot=str(dest/'comparison.png'),log=str(log))
  prevpath=prior/f'K_{k:.3f}'/'model'/'validation.json'
  if p.returncode==0 and prevpath.exists():
   old=json.loads(prevpath.read_text());new=json.loads((dest/'validation.json').read_text())
   def stats(d):
    out={}
    for split in ('train','holdout'):
     total=[r for r in d['cycles'] if r['split']==split];good=[r for r in total if r['PWA'] is not None]
     out[split]=dict(total=len(total),valid=len(good),**{f'mean_abs_{name}':sum(abs(q['error'][name]) for q in good)/len(good) if good else None for name in ('Cmin','Cmax','period','shape_rmse')})
    return out
   cmp=dict(K=k,previous_random_winner=stats(old),refined=stats(new),scope='Independent finite DOP853 results on same input values. No fold, no global cycle count.')
   (dest/'refined_vs_previous.json').write_text(json.dumps(cmp,indent=2)+'\n');record['comparison']=str(dest/'refined_vs_previous.json')
  validation[str(k)]=record
 summary=dict(scope='Refinement of top independent random seeds, selected by training loss. No fold analysis. Optimization and validation are finite; inspect convergence and all held-out results.',source=str(prior),trials=items,winners=winners,validation=validation)
 (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('SAVED',a.out/'summary.json',flush=True)
 if len(winners)!=3 or any(r['exit_code'] for r in validation.values()):raise SystemExit('Some models or validations failed; inspect logs and summary')
if __name__=='__main__':main()
