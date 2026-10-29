#!/usr/bin/env python3
"""Fit three physical-C models without fold constraint and compare frozen models."""
import argparse,concurrent.futures,json,os,shutil,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];FREE=ROOT/'astrocyte_lab/experiments/free_hidden_refine'
sys.path.insert(0,str(ROOT/'astrocyte_lab/experiments/fm2_amplitude_refine'))
from run_server import stats

def main():
 p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--starts',type=int,default=6);p.add_argument('--workers',type=int,default=16);p.add_argument('--jitter',type=float,default=.035);p.add_argument('--powell-maxfev',type=int,default=6000);p.add_argument('--ls-max-nfev',type=int,default=400);p.add_argument('--duration',type=float,default=350);p.add_argument('--validation-duration',type=float,default=450);p.add_argument('--dt',type=float,default=.01);p.add_argument('--validate-top',type=int,default=2);a=p.parse_args()
 if min(a.starts,a.workers,a.powell_maxfev,a.ls_max_nfev,a.validate_top)<1 or min(a.duration,a.validation_duration)<100 or a.dt<=0 or a.jitter<0:raise ValueError('Invalid options')
 out=a.out.resolve();out.mkdir(parents=True,exist_ok=True);snapshot=a.snapshot.resolve();env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR=str(out/'mpl'),XDG_CACHE_HOME=str(out/'cache'))
 for mode in ('fm','afm','am'):
  for coord in ('physical','mixed'):
   if not (snapshot/f'{mode}_{coord}'/'parameters.json').is_file():raise FileNotFoundError(snapshot/f'{mode}_{coord}'/'parameters.json')
 engine=out/'shape_engine.so';subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(HERE/'shape_engine.cpp'),'-o',str(engine)],check=True)
 tasks=[(mode,i) for mode in ('fm','afm','am') for i in range(a.starts)];print(f'FIT {len(tasks)} starts, workers={a.workers}, output={out}',flush=True)
 def job(task):
  mode,i=task;dest=out/mode/f'trial_{i:02d}';dest.mkdir(parents=True,exist_ok=True);peak=1. if mode=='fm' else 0.;src=snapshot/f'{mode}_physical'/'parameters.json';jitter=0 if i==0 else a.jitter
  cmd=[sys.executable,'-u',str(HERE/'refine.py'),'--seed-model',str(src),'--engine',str(engine),'--out',str(dest),'--powell-maxfev',str(a.powell_maxfev),'--ls-max-nfev',str(a.ls_max_nfev),'--duration',str(a.duration),'--dt',str(a.dt),'--peak-weight',str(peak),'--jitter',str(jitter),'--random-seed',str(4200+i)]
  with (dest/'fit.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
  path=dest/'parameters.json';d=json.loads(path.read_text()) if path.exists() else {};row=dict(mode=mode,start=i,peak_weight=peak,jitter=jitter,exit_code=r.returncode,status=d.get('status','process_error'),loss=d.get('loss'),initial_loss=d.get('initial_loss'),optimizers=d.get('optimizers'),path=str(path),log=str(dest/'fit.log'));print('TRIAL',json.dumps(row),flush=True);return row
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:trials=list(pool.map(job,tasks))
 report=dict(scope='Physical C models with unrestricted q; no fold or Poincare constraint, no fold scan. Selection by training loss; DOP853 validation on common finite inputs.',snapshot=str(snapshot),settings=dict(starts=a.starts,jitter=a.jitter,peak_weight={'fm':1.,'afm':0.,'am':0.},duration=a.duration,validation_duration=a.validation_duration,dt=a.dt),trials=trials,validation={})
 def validate(mode,label,src,script):
  dest=out/mode/'comparison'/label;dest.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest/'parameters.json')
  with (dest/'validate.log').open('w') as f:r=subprocess.run([sys.executable,'-u',str(script),str(dest/'parameters.json'),'--duration',str(a.validation_duration)],env=env,stdout=f,stderr=subprocess.STDOUT)
  rec=dict(exit_code=r.returncode,source=str(src),path=str(dest/'validation.json'),plot=str(dest/'comparison.png'),log=str(dest/'validate.log'))
  if r.returncode==0 and (dest/'validation.json').exists():rec['metrics']=stats(dest/'validation.json')
  report['validation'].setdefault(mode,{})[label]=rec;print('VALIDATION',mode,label,json.dumps(rec),flush=True)
 for mode in ('fm','afm','am'):
  validate(mode,'old_physical',snapshot/f'{mode}_physical'/'parameters.json',HERE/'validate.py')
  validate(mode,'frozen_mixed',snapshot/f'{mode}_mixed'/'parameters.json',FREE/'validate.py')
  candidates=sorted([r for r in trials if r['mode']==mode and r['exit_code']==0 and r['status']=='training_cycles_found' and r['loss'] is not None],key=lambda r:r['loss'])
  if not candidates:print('NO CANDIDATE',mode,flush=True)
  for i,row in enumerate(candidates[:a.validate_top]):validate(mode,f'new_rank_{i+1:02d}',Path(row['path']),HERE/'validate.py')
 (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print('SAVED',out/'summary.json',flush=True)
 if any(not any(r['mode']==mode and r['status']=='training_cycles_found' and r['exit_code']==0 for r in trials) for mode in ('fm','afm','am')) or any(r['exit_code'] for group in report['validation'].values() for r in group.values()):raise SystemExit('Some candidates or validations failed; inspect summary and logs')
if __name__=='__main__':main()
