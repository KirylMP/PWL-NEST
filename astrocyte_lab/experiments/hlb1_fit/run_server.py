#!/usr/bin/env python3
"""Refit six two-threshold models with subcritical focus-focus lower transition."""
import argparse,concurrent.futures,json,os,shutil,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];FREE=ROOT/'astrocyte_lab/experiments/free_hidden_refine';PHYS=ROOT/'astrocyte_lab/experiments/physical_no_fold';AUDIT=ROOT/'astrocyte_lab/experiments/lower_hlb/audit.py'
sys.path.insert(0,str(ROOT/'astrocyte_lab/experiments/fm2_amplitude_refine'))
from run_server import stats

def main():
 p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--physical-run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--starts',type=int,default=6);p.add_argument('--workers',type=int,default=16);p.add_argument('--screen-tries',type=int,default=80);p.add_argument('--jitter',type=float,default=.04);p.add_argument('--powell-maxfev',type=int,default=6000);p.add_argument('--ls-max-nfev',type=int,default=400);p.add_argument('--duration',type=float,default=350);p.add_argument('--validation-duration',type=float,default=450);p.add_argument('--dt',type=float,default=.01);p.add_argument('--audit-starts',type=int,default=4);p.add_argument('--audit-workers',type=int,default=8);a=p.parse_args()
 if min(a.starts,a.workers,a.screen_tries,a.powell_maxfev,a.ls_max_nfev,a.audit_starts,a.audit_workers)<1 or min(a.duration,a.validation_duration)<100 or a.dt<=0 or a.jitter<0:raise ValueError('Invalid parameters')
 out=a.out.resolve();out.mkdir(parents=True,exist_ok=True);snapshot=a.snapshot.resolve();prior=a.physical_run.resolve()
 sources={}
 for mode in ('am','afm','fm'):
  for coord in ('physical','mixed'):
   src=(prior/mode/'comparison/new_rank_01/parameters.json') if coord=='physical' else (snapshot/f'{mode}_mixed/parameters.json')
   if not src.is_file():raise FileNotFoundError(src)
   sources[(mode,coord)]=src
 env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR=str(out/'mpl'),XDG_CACHE_HOME=str(out/'cache'));engines={}
 for coord,folder in (('physical',PHYS),('mixed',FREE)):
  engine=out/(coord+'_engine.so');subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(folder/'shape_engine.cpp'),'-o',str(engine)],check=True);engines[coord]=engine
 tasks=[(mode,coord,i) for mode in ('am','afm','fm') for coord in ('physical','mixed') for i in range(a.starts)];print(f'FITTING {len(tasks)} starts workers={a.workers}; output={out}',flush=True)
 def job(task):
  mode,coord,i=task;dest=out/mode/coord/f'trial_{i:02d}';dest.mkdir(parents=True,exist_ok=True);peak=1. if mode=='fm' else 0.
  cmd=[sys.executable,'-u',str(HERE/'refine.py'),'--seed-model',str(sources[(mode,coord)]),'--engine',str(engines[coord]),'--out',str(dest),'--powell-maxfev',str(a.powell_maxfev),'--ls-max-nfev',str(a.ls_max_nfev),'--screen-tries',str(a.screen_tries),'--duration',str(a.duration),'--dt',str(a.dt),'--peak-weight',str(peak),'--jitter',str(0. if i==0 else a.jitter),'--random-seed',str(81200+100*i+(0 if coord=='physical' else 20))]
  with (dest/'fit.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=env)
  path=dest/'parameters.json';d=json.loads(path.read_text()) if path.is_file() else {};row=dict(mode=mode,coordinate=coord,start=i,exit_code=r.returncode,status=d.get('status','process_error'),loss=d.get('loss'),initial_loss=d.get('initial_loss'),hlb1_alpha=d.get('hlb1_alpha'),screen_cycles=d.get('screen_cycles'),optimizers=d.get('optimizers'),path=str(path),log=str(dest/'fit.log'));print('TRIAL',json.dumps(row),flush=True);return row
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:rows=list(pool.map(job,tasks))
 report=dict(scope='Six two-threshold models; hard stable-focus and positive HLB1 coefficient. No fold or Poincare fitting. Local cycles and stable trajectory validation are finite; frozen inputs unchanged.',settings=dict(starts=a.starts,screen_tries=a.screen_tries,jitter=a.jitter,duration=a.duration,validation_duration=a.validation_duration,dt=a.dt,peak_weight={'am':0.,'afm':0.,'fm':1.}),trials=rows,winners={},validation={})
 for mode,coord in sources:
  group=sorted([r for r in rows if r['mode']==mode and r['coordinate']==coord and r['exit_code']==0 and r['status']=='training_cycles_found' and r['loss'] is not None],key=lambda r:r['loss']);key=mode+'_'+coord
  if not group:print('NO CANDIDATE',key,flush=True);continue
  winner=group[0];report['winners'][key]=winner;validator=PHYS/'validate.py' if coord=='physical' else FREE/'validate.py'
  for label,src in [('previous',sources[(mode,coord)]),('candidate',Path(winner['path']))]:
   dest=out/mode/coord/label;dest.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest/'parameters.json')
   with (dest/'validate.log').open('w') as f:r=subprocess.run([sys.executable,'-u',str(validator),str(dest/'parameters.json'),'--duration',str(a.validation_duration)],stdout=f,stderr=subprocess.STDOUT,env=env)
   val=dict(exit_code=r.returncode,path=str(dest/'validation.json'),plot=str(dest/'comparison.png'),log=str(dest/'validate.log'))
   if r.returncode==0 and (dest/'validation.json').exists():val['metrics']=stats(dest/'validation.json')
   report['validation'].setdefault(key,{})[label]=val;print('VALIDATION',key,label,json.dumps(val),flush=True)
  dest=out/mode/coord/'candidate';print('LOWER CYCLES',key,flush=True)
  with (dest/'lower_audit.log').open('w') as f:r=subprocess.run([sys.executable,'-u',str(AUDIT),'--candidate',str(dest/'parameters.json'),'--out',str(dest/'lower_audit'),'--starts',str(a.audit_starts),'--workers',str(a.audit_workers),'--deltas','.0001','.001','.005'],stdout=f,stderr=subprocess.STDOUT,env=env)
  v=report['validation'][key]['candidate'];v['lower_audit_exit_code']=r.returncode;v['lower_audit_path']=str(dest/'lower_audit/lower_hlb.json');v['lower_audit_log']=str(dest/'lower_audit.log')
  if r.returncode==0 and (dest/'lower_audit/lower_hlb.json').exists():
   check=json.loads((dest/'lower_audit/lower_hlb.json').read_text())['models']['candidate'];v['lower_local']=check['local'];v['lower_cycles']=check['cycles']
  (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
 (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print('SAVED',out/'summary.json',flush=True)
 if len(report['winners'])!=6 or any(v['exit_code'] or v.get('lower_audit_exit_code',0) for g in report['validation'].values() for v in g.values()):raise SystemExit('Some fits or validations failed; inspect summary and logs')
if __name__=='__main__':main()
