#!/usr/bin/env python3
"""Compare two rational FM2 calcium coordinates on common targets."""
import argparse,concurrent.futures,json,os,shutil,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
FREE=ROOT/'astrocyte_lab/experiments/free_hidden_refine'
sys.path.insert(0,str(ROOT/'astrocyte_lab/experiments/fm2_amplitude_refine'))
from run_server import stats

def main():
 p=argparse.ArgumentParser();p.add_argument('--prior-run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--top',type=int,default=4);p.add_argument('--starts',type=int,default=2);p.add_argument('--workers',type=int,default=16);p.add_argument('--jitter',type=float,default=.03);p.add_argument('--peak-weight',type=float,default=1.);p.add_argument('--powell-maxfev',type=int,default=6000);p.add_argument('--ls-max-nfev',type=int,default=400);p.add_argument('--duration',type=float,default=350);p.add_argument('--validation-duration',type=float,default=450);p.add_argument('--dt',type=float,default=.01);p.add_argument('--validate-top',type=int,default=3);a=p.parse_args()
 if min(a.top,a.starts,a.workers,a.powell_maxfev,a.ls_max_nfev,a.validate_top)<1 or min(a.duration,a.validation_duration)<100 or a.dt<=0 or min(a.jitter,a.peak_weight)<0:raise ValueError('Invalid options')
 out=a.out.resolve();out.mkdir(parents=True,exist_ok=True);prior=a.prior_run.resolve();old=json.loads((prior/'summary.json').read_text());rows=sorted([r for r in old['trials'] if r['exit_code']==0 and r['status']=='training_cycles_found' and r.get('loss') is not None],key=lambda r:r['loss'])[:a.top]
 if not rows:raise ValueError('Prior run contains no seeds')
 env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR=str(out/'mpl'),XDG_CACHE_HOME=str(out/'cache'))
 engines={}
 for family in ('fractional','rational'):
  engine=out/(family+'.so');cmd=['g++','-O3','-std=c++17','-shared','-fPIC']+(['-DFLEXIBLE_RATIONAL'] if family=='rational' else [])+[str(HERE/'shape_engine.cpp'),'-o',str(engine)];subprocess.run(cmd,check=True);engines[family]=engine
 tasks=[]
 for family in engines:
  for rank,row in enumerate(rows):
   source=prior/f"seed_{row['source_trial']:03d}_start_{row['start']:02d}"/'parameters.json'
   if not source.is_file():raise FileNotFoundError(source)
   for n in range(a.starts):tasks.append((family,rank,row,source,n))
 print(f'FIT {len(tasks)} starts, workers={a.workers}, output={out}',flush=True)
 def job(task):
  family,rank,row,source,n=task;dest=out/family/f'seed_{rank:02d}_start_{n:02d}';dest.mkdir(parents=True,exist_ok=True);runenv=dict(env,RATIONAL_FAMILY=family)
  cmd=[sys.executable,'-u',str(HERE/'refine.py'),'--seed-model',str(source),'--engine',str(engines[family]),'--out',str(dest),'--powell-maxfev',str(a.powell_maxfev),'--ls-max-nfev',str(a.ls_max_nfev),'--duration',str(a.duration),'--dt',str(a.dt),'--peak-weight',str(a.peak_weight),'--jitter',str(0 if n==0 else a.jitter),'--random-seed',str(83100+100*rank+n)]
  with (dest/'fit.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=runenv)
  path=dest/'parameters.json';d=json.loads(path.read_text()) if path.exists() else {};result=dict(family=family,seed_rank=rank,start=n,seed=str(source),exit_code=r.returncode,status=d.get('status','process_error'),loss=d.get('loss'),initial_loss=d.get('initial_loss'),path=str(path),log=str(dest/'fit.log'),optimizers=d.get('optimizers'));print('TRIAL',json.dumps(result),flush=True);return result
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:trials=list(pool.map(job,tasks))
 report=dict(scope='FM2, weighted Cmax and waveform/period objective. Finite validation only; no fold or global cycle count. The mixed-coordinate seeds are reparameterized as starting guesses, not equivalent models.',source=str(prior),settings=dict(top=a.top,starts=a.starts,peak_weight=a.peak_weight,jitter=a.jitter,duration=a.duration,validation_duration=a.validation_duration,dt=a.dt),trials=trials,validation={})
 def validate(label,src,script,runenv):
  dest=out/'comparison'/label;dest.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest/'parameters.json')
  with (dest/'validate.log').open('w') as f:r=subprocess.run([sys.executable,'-u',str(script),str(dest/'parameters.json'),'--duration',str(a.validation_duration)],stdout=f,stderr=subprocess.STDOUT,env=runenv)
  rec=dict(exit_code=r.returncode,path=str(dest/'validation.json'),plot=str(dest/'comparison.png'),log=str(dest/'validate.log'),source=str(src))
  if r.returncode==0 and (dest/'validation.json').exists():rec['metrics']=stats(dest/'validation.json')
  report['validation'][label]=rec;print('VALIDATION',label,json.dumps(rec),flush=True)
 validate('previous_mixed',prior/'comparison/rank_01/parameters.json',FREE/'validate.py',env)
 for family in engines:
  valid=sorted([r for r in trials if r['family']==family and r['exit_code']==0 and r['status']=='training_cycles_found' and r['loss'] is not None],key=lambda r:r['loss'])
  if not valid:print('NO CANDIDATE',family,flush=True)
  for i,row in enumerate(valid[:a.validate_top]):validate(f'{family}_rank_{i+1:02d}',Path(row['path']),HERE/'validate.py',dict(env,RATIONAL_FAMILY=family))
 (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print('SAVED',out/'summary.json',flush=True)
 if any(not any(r['family']==fam and r['status']=='training_cycles_found' and r['exit_code']==0 for r in trials) for fam in engines) or any(r['exit_code'] for r in report['validation'].values()):raise SystemExit('Some families or validations failed; inspect summary and logs')
if __name__=='__main__':main()
