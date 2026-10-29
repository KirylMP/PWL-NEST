#!/usr/bin/env python3
"""Independent joint linear fits in parallel, then validate training winner."""
import argparse,concurrent.futures,json,os,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--trials',type=int,default=16);p.add_argument('--workers',type=int,default=16);p.add_argument('--max-nfev',type=int,default=600);p.add_argument('--duration',type=float,default=350.);p.add_argument('--dt',type=float,default=.02);p.add_argument('--validation-duration',type=float,default=400.);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR=str(a.out/'mpl'),XDG_CACHE_HOME=str(a.out/'cache'))
 def job(j):
  d=a.out/f'trial_{j:03d}';d.mkdir(parents=True,exist_ok=True);log=d/'fit.log';cmd=[sys.executable,'-u',str(HERE/'fit_joint.py'),'--out',str(d),'--starts','1','--max-nfev',str(a.max_nfev),'--duration',str(a.duration),'--dt',str(a.dt),'--seed',str(8300+j),'--jitter',str(0 if j==0 else .04 if j<8 else .12)]
  with log.open('w') as handle:ret=subprocess.run(cmd,stdout=handle,stderr=subprocess.STDOUT,env=env).returncode
  f=d/'family.json';r=json.loads(f.read_text()) if f.exists() else None;item=dict(trial=j,exit_code=ret,loss=r['train_loss'] if r else None,log=str(log),path=str(f) if r else None,optimizer=r['attempts'][0] if r else None);print('TRIAL',json.dumps(item),flush=True);return item
 print(f'RUN {a.trials} trials, workers={a.workers}, output={a.out}',flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:items=list(pool.map(job,range(a.trials)))
 valid=[r for r in items if r['loss'] is not None];summary=dict(trials=items,winner=None,scope='Winner selected by training loss. Check failed or unresolved training and holdout cycles, optimizer status, event heights and independent validation.')
 if not valid:
  (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');raise SystemExit('No completed trials')
 winner=min(valid,key=lambda q:q['loss']);summary['winner']=winner;print('WINNER',json.dumps(winner),flush=True)
 log=a.out/'validation.log';cmd=[sys.executable,'-u',str(HERE/'validate_joint.py'),winner['path'],'--duration',str(a.validation_duration)]
 with log.open('w') as handle:ret=subprocess.run(cmd,stdout=handle,stderr=subprocess.STDOUT,env=env).returncode
 summary['independent_validation']=dict(exit_code=ret,log=str(log),path=str(Path(winner['path']).parent/'independent_holdout.json'))
 (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('SAVED',a.out/'summary.json','VALIDATION',ret,flush=True)
if __name__=='__main__':main()
