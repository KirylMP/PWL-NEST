#!/usr/bin/env python3
"""Parallel three-model fitting without fold constraints."""
import argparse,concurrent.futures,json,os,shutil,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=16);p.add_argument('--trials',type=int,default=6);p.add_argument('--max-nfev',type=int,default=1200);p.add_argument('--init-tries',type=int,default=500);p.add_argument('--duration',type=float,default=350);p.add_argument('--validation-duration',type=float,default=450);p.add_argument('--dt',type=float,default=.02);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 if min(a.workers,a.trials,a.max_nfev,a.init_tries)<1 or min(a.duration,a.validation_duration)<100 or a.dt<=0:raise ValueError('Invalid settings')
 env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR=str(a.out/'mpl'),XDG_CACHE_HOME=str(a.out/'cache'));engine=a.out/'shape_engine.so';subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(ROOT/'astrocyte_lab/experiments/mixed_three_fits/shape_engine.cpp'),'-o',str(engine)],check=True)
 ks=[.05,.075,.1];tasks=[(k,j) for k in ks for j in range(a.trials)];print(f'RUN {len(tasks)} trials workers={a.workers} out={a.out}',flush=True)
 def fit(item):
  k,j=item;d=a.out/f'K_{k:.3f}'/f'trial_{j:03d}';d.mkdir(parents=True,exist_ok=True);log=d/'fit.log';cmd=[sys.executable,'-u',str(HERE/'fit_model.py'),'--K',str(k),'--engine',str(engine.resolve()),'--out',str(d),'--max-nfev',str(a.max_nfev),'--duration',str(a.duration),'--dt',str(a.dt),'--seed',str(20261200+int(k*1000)*1000+j),'--init-tries',str(a.init_tries)]
  with log.open('w') as f:proc=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
  path=d/'parameters.json';data=json.loads(path.read_text()) if path.exists() else {};result=dict(K=k,trial=j,status=data.get('status','process_error'),exit_code=proc.returncode,loss=data.get('loss'),path=str(path),log=str(log),optimizer=data.get('optimizer'));print('TRIAL',json.dumps(result),flush=True);return result
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:items=list(pool.map(fit,tasks))
 winners={};validation={}
 for k in ks:
  group=[q for q in items if q['K']==k and q['status']=='training_cycles_found' and q['loss'] is not None]
  if not group:print(f'NO CANDIDATE K={k:.3f}',flush=True);continue
  win=min(group,key=lambda q:q['loss']);winners[str(k)]=win;dest=a.out/f'K_{k:.3f}'/'model';dest.mkdir(parents=True,exist_ok=True);shutil.copy2(win['path'],dest/'parameters.json');log=dest/'validate.log';print(f'VALIDATE K={k:.3f} log={log}',flush=True)
  with log.open('w') as f:proc=subprocess.run([sys.executable,'-u',str(HERE/'validate.py'),str(dest/'parameters.json'),'--duration',str(a.validation_duration)],env=env,stdout=f,stderr=subprocess.STDOUT)
  validation[str(k)]=dict(exit_code=proc.returncode,path=str(dest/'validation.json'),comparison=str(dest/'baseline_comparison.json'),plot=str(dest/'comparison.png'),log=str(log))
 summary=dict(scope='Winners chosen by training waveform and period loss; stable trajectories validated at fixed inputs. No fold analysis or unstable-cycle search.',trials=items,winners=winners,validation=validation)
 (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('SAVED',a.out/'summary.json',flush=True)
 if len(winners)!=3 or any(r['exit_code'] for r in validation.values()):raise SystemExit('Some models or validations failed; inspect summary and logs')
if __name__=='__main__':main()
