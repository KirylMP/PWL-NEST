#!/usr/bin/env python3
"""Compare soft period weights for AFM local HLB2 cycle."""
import argparse,concurrent.futures,json,os,shutil,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];FREE=ROOT/'astrocyte_lab/experiments/free_hidden_refine';HLB=ROOT/'astrocyte_lab/experiments/hlb2_afm'
sys.path.insert(0,str(ROOT/'astrocyte_lab/experiments/fm2_amplitude_refine'));from run_server import stats

def main():
 p=argparse.ArgumentParser();p.add_argument('--seed-model',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--weights',type=float,nargs='+',default=[.001,.01,.1]);p.add_argument('--starts',type=int,default=2);p.add_argument('--workers',type=int,default=6);p.add_argument('--jitter',type=float,default=.02);p.add_argument('--maxfev',type=int,default=1800);p.add_argument('--ls-max-nfev',type=int,default=120);p.add_argument('--duration',type=float,default=350);p.add_argument('--validation-duration',type=float,default=450);p.add_argument('--dt',type=float,default=.01);a=p.parse_args()
 if min(a.weights)<=0 or min(a.starts,a.workers,a.maxfev,a.ls_max_nfev)<1 or a.jitter<0 or min(a.duration,a.validation_duration)<100 or a.dt<=0:raise ValueError('Invalid settings')
 out=a.out.resolve();out.mkdir(parents=True,exist_ok=True);seed=a.seed_model.resolve();d=json.loads(seed.read_text())
 if abs(d['K']-.075)>1e-9 or d.get('coordinate_family')!='mixed':raise ValueError('Expected mixed AFM seed')
 engine=out/'shape_engine.so';subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(FREE/'shape_engine.cpp'),'-o',str(engine)],check=True)
 env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR=str(out/'mpl'),XDG_CACHE_HOME=str(out/'cache'));tasks=[(weight,i) for weight in a.weights for i in range(a.starts)];print('FIT',len(tasks),'tasks',flush=True)
 def job(task):
  weight,i=task;dest=out/f'weight_{weight:g}'/f'trial_{i:02d}';dest.mkdir(parents=True,exist_ok=True);cmd=[sys.executable,'-u',str(HERE/'refine.py'),'--seed-model',str(seed),'--engine',str(engine),'--out',str(dest),'--period-weight',str(weight),'--peak-weight','0','--jitter',str(0 if i==0 else a.jitter),'--random-seed',str(50300+i),'--powell-maxfev',str(a.maxfev),'--ls-max-nfev',str(a.ls_max_nfev),'--duration',str(a.duration),'--dt',str(a.dt)]
  with (dest/'fit.log').open('w') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
  path=dest/'parameters.json';val=json.loads(path.read_text()) if path.exists() else {};row=dict(weight=weight,start=i,exit_code=r.returncode,status=val.get('status','process_error'),loss=val.get('loss'),large_cycle_loss=val.get('large_cycle_loss'),local_period=val.get('local_period'),optimizers=val.get('optimizers'),path=str(path),log=str(dest/'fit.log'));print('TRIAL',json.dumps(row),flush=True);return row
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:trials=list(pool.map(job,tasks))
 summary=dict(scope='AFM mixed coordinate; fixed node/focus crossing; soft local HLB2 period term, no fold condition. Winners selected within each weight by training objective. Finite local and DOP853 validation; frozen source untouched.',seed=str(seed),settings=dict(weights=a.weights,starts=a.starts,period_target=14.388890297911214,period_term='weight*((T_limit-target)/target)^2',duration=a.duration,validation_duration=a.validation_duration,dt=a.dt),trials=trials,models={})
 candidates=[('previous',seed)]
 for weight in a.weights:
  group=sorted([r for r in trials if r['weight']==weight and r['exit_code']==0 and r['status']=='training_cycles_found' and r['loss'] is not None],key=lambda r:r['loss'])
  if group:candidates.append((f'weight_{weight:g}',Path(group[0]['path'])))
  else:print('NO CANDIDATE weight=',weight,flush=True)
 for label,path in candidates:
  dest=out/'comparison'/label;dest.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest/'parameters.json');print('VALIDATE',label,flush=True)
  with (dest/'validate.log').open('w') as f:r=subprocess.run([sys.executable,'-u',str(FREE/'validate.py'),str(dest/'parameters.json'),'--duration',str(a.validation_duration)],stdout=f,stderr=subprocess.STDOUT,env=env)
  record=dict(validation_exit=r.returncode,path=str(dest/'parameters.json'),validation_path=str(dest/'validation.json'),plot=str(dest/'comparison.png'),log=str(dest/'validate.log'))
  if r.returncode==0 and (dest/'validation.json').exists():record['metrics']=stats(dest/'validation.json')
  # Run independent high-precision matching with sampled itinerary for the saved model.
  code="import json,sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);from check import check;r=check(sys.argv[2]);Path(sys.argv[3]).write_text(json.dumps(r,indent=2,allow_nan=False)+'\\n');print(json.dumps(r),flush=True)"
  with (dest/'hlb2_audit.log').open('w') as f:proc=subprocess.run([sys.executable,'-c',code,str(HLB),str(dest/'parameters.json'),str(dest/'hlb2_audit.json')],stdout=f,stderr=subprocess.STDOUT,env=env)
  record['hlb2_audit_exit']=proc.returncode;record['hlb2_audit_log']=str(dest/'hlb2_audit.log');record['hlb2_audit_path']=str(dest/'hlb2_audit.json')
  if proc.returncode==0 and (dest/'hlb2_audit.json').exists():record['hlb2_audit']=json.loads((dest/'hlb2_audit.json').read_text())
  summary['models'][label]=record;(out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
 print('SAVED',out/'summary.json',flush=True)
 if len(candidates)!=len(a.weights)+1 or any(r['validation_exit'] or r['hlb2_audit_exit'] for r in summary['models'].values()):raise SystemExit('Some fits or audits did not complete; inspect logs')
if __name__=='__main__':main()
