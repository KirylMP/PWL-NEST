#!/usr/bin/env python3
"""Prepare references, run multistart fits, validate each selected model."""
import argparse,concurrent.futures,json,os,shutil,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=16);p.add_argument('--trials',type=int,default=6);p.add_argument('--maxiter',type=int,default=1000);p.add_argument('--init-max-nfev',type=int,default=1500);p.add_argument('--duration',type=float,default=350);p.add_argument('--reference-duration',type=float,default=600);p.add_argument('--validation-duration',type=float,default=450);p.add_argument('--dt',type=float,default=.02);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 if min(a.workers,a.trials,a.maxiter,a.init_max_nfev)<1 or min(a.duration,a.reference_duration,a.validation_duration)<100 or a.dt<=0:raise ValueError('Invalid settings')
 env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR=str(a.out/'mpl'),XDG_CACHE_HOME=str(a.out/'cache'));engine=a.out/'shape_engine.so';subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(HERE/'shape_engine.cpp'),'-o',str(engine)],check=True)
 Ks=[.05,.075,.1]
 def prepare(k):
  d=a.out/f'K_{k:.3f}'/'reference';d.mkdir(parents=True,exist_ok=True);log=d/'prepare.log';cmd=[sys.executable,'-u',str(HERE/'prepare.py'),'--K',str(k),'--out',str(d),'--duration',str(a.reference_duration)]
  with log.open('w') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
  print(f'REFERENCE K={k:.3f} exit={r.returncode} log={log}',flush=True);return k,r.returncode
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:prep=dict(pool.map(prepare,Ks))
 tasks=[(k,j) for k in Ks if prep[k]==0 for j in range(a.trials)];print(f'RUN {len(tasks)} trials with {a.workers} workers at {a.out}',flush=True)
 def fit(task):
  k,j=task;parent=a.out/f'K_{k:.3f}';d=parent/f'trial_{j:03d}';d.mkdir(parents=True,exist_ok=True);log=d/'fit.log';cmd=[sys.executable,'-u',str(HERE/'fit_model.py'),'--reference',str((parent/'reference/reference.json').resolve()),'--engine',str(engine.resolve()),'--out',str(d),'--seed',str(20261100+j),'--jitter',str(0 if j==0 else .08 if j<4 else .2),'--maxiter',str(a.maxiter),'--init-max-nfev',str(a.init_max_nfev),'--duration',str(a.duration),'--dt',str(a.dt)]
  with log.open('w') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
  path=d/'parameters.json';data=json.loads(path.read_text()) if path.exists() else {};item=dict(K=k,trial=j,exit_code=r.returncode,status=data.get('status','process_error'),loss=data.get('loss'),path=str(path),log=str(log));print('TRIAL',json.dumps(item),flush=True);return item
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:items=list(pool.map(fit,tasks))
 winners={};validation={}
 for k in Ks:
  group=[q for q in items if q['K']==k and q['status']=='fold_verified' and q['loss'] is not None]
  if not group:print('NO VERIFIED WINNER',k,flush=True);continue
  winner=min(group,key=lambda q:q['loss']);dest=a.out/f'K_{k:.3f}'/'model';dest.mkdir(parents=True,exist_ok=True);shutil.copy2(winner['path'],dest/'parameters.json');winners[str(k)]=winner
  log=dest/'validate.log';cmd=[sys.executable,'-u',str(HERE/'validate.py'),str(dest/'parameters.json'),'--reference',str((a.out/f'K_{k:.3f}'/'reference/reference.json').resolve()),'--duration',str(a.validation_duration)]
  print(f'VALIDATING K={k:.3f} log={log}',flush=True)
  with log.open('w') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
  validation[str(k)]=dict(exit_code=r.returncode,path=str(dest/'validation.json'),log=str(log),plot=str(dest/'comparison.png'))
 summary=dict(scope='Winners selected by training waveform and period loss among locally verified folds. Independently inspect validation.json for missed cycles, holdout errors, nearby stable and unstable roots; absence in finite scan is not nonexistence.',preparation=prep,trials=items,winners=winners,validation=validation)
 (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('SAVED',a.out/'summary.json',flush=True)
 if len(winners)!=3 or any(q['exit_code'] for q in validation.values()):raise SystemExit('Some references, models or validations failed; inspect logs and summary.')
if __name__=='__main__':main()
