"""Parallel independent FM fits. Environment variables configure budget and paths."""
import concurrent.futures,json,os,shutil,subprocess,sys,time,zipfile
from pathlib import Path
P=Path(__file__).resolve().parent
workers=int(os.environ.get('FIT_WORKERS','16')); starts=int(os.environ.get('FIT_RANDOM_STARTS','100000')); maxfev=int(os.environ.get('FIT_MAXFEV','20000')); seed0=int(os.environ.get('FIT_SEED_BASE','47000'));root=Path(os.environ.get('FIT_RUN_DIR',str(P/'runs'/time.strftime('fit_%Y%m%dT%H%M%SZ',time.gmtime())))).resolve();root.mkdir(parents=True,exist_ok=True)
if workers<1 or starts<1 or maxfev<1:raise SystemExit('FIT_WORKERS, FIT_RANDOM_STARTS, FIT_MAXFEV must be positive')
print('RUN_DIR='+str(root),flush=True)
print(f'workers={workers} starts_per_worker={starts} maxfev_per_worker={maxfev}',flush=True)
compiler=os.environ.get('CXX','g++');lib=root/'shape_checked.so'
subprocess.run([compiler,'-O3','-std=c++17','-shared','-fPIC',str(P/'shape.cpp'),'-o',str(lib)],check=True)
files=['fit_free_q.py','validate.py','seed.json','initial_scan.json','reference_cycles.json','geometry.json']
def worker(k):
 wd=root/f'worker_{k:02d}';wd.mkdir(exist_ok=True)
 for name in files:shutil.copy2(P/name,wd/name)
 shutil.copy2(lib,wd/'shape_checked.so')
 env=os.environ.copy();env.update({'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','FIT_SEED':'seed.json','FIT_RANDOM_SEED':str(seed0+k),'FIT_RANDOM_STARTS':str(starts),'FIT_MAXFEV':str(maxfev),'FIT_OUTPUT':'candidate.json','VALIDATION_WORKERS':'1'})
 start=time.monotonic()
 with (wd/'fit.log').open('w') as log:fit=subprocess.run([sys.executable,'-u',str(wd/'fit_free_q.py')],cwd=wd,env=env,stdout=log,stderr=subprocess.STDOUT)
 row={'worker':k,'seed':seed0+k,'fit_exit':fit.returncode,'elapsed_s':round(time.monotonic()-start,2),'candidate':str(wd/'candidate.json')}
 if (wd/'candidate.json').exists():
  try:
   x=json.loads((wd/'candidate.json').read_text());row.update({'loss':x['loss'],'fast_detected':sum(r['detected'] for r in x['training']),'input_scale':x['input_scale'],'anchor_max_residual':x['anchor_max_residual']})
   with (wd/'validation.log').open('w') as log:check=subprocess.run([sys.executable,'-u',str(wd/'validate.py')],cwd=wd,env=env,stdout=log,stderr=subprocess.STDOUT)
   row['validation_exit']=check.returncode
   if (wd/'validation.json').exists():
    v=json.loads((wd/'validation.json').read_text());row['validated']=sum(a['valid'] for a in v['results']);row['validation']=[{'gamma':a['gamma'],'valid':a['valid'],'period':a.get('period'),'Cmax':a.get('Cmax'),'min_C':a.get('trajectory_min_C')} for a in v['results']]
  except Exception as e:row['summary_error']=str(e)
 (wd/'worker_summary.json').write_text(json.dumps(row,indent=2)+'\n')
 print(f"worker={k:02d} exit={fit.returncode} loss={row.get('loss','-')} validated={row.get('validated','-')} elapsed={row['elapsed_s']}s",flush=True)
 return row
rows=[]
try:
 with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
  tasks=[pool.submit(worker,k) for k in range(workers)]
  for t in concurrent.futures.as_completed(tasks):
   try:rows.append(t.result())
   except Exception as e:print('worker failed:',str(e),flush=True)
finally:
 rows.sort(key=lambda x:(-x.get('validated',-1),x.get('loss',float('inf'))))
 summary={'scope':'independent fits; sort by validated count then training loss; no extrapolation or proof of no other cycles','workers':workers,'random_starts_per_worker':starts,'maxfev_per_worker':maxfev,'results':rows}
 (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 archive=root/'review.zip'
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  z.write(root/'summary.json','summary.json')
  for r in rows:
   wd=root/f"worker_{r['worker']:02d}"
   for name in ['candidate.json','validation.json','worker_summary.json','checkpoint.json']:
    if (wd/name).exists():z.write(wd/name,f'{wd.name}/{name}')
 print('SUMMARY='+str(root/'summary.json'),flush=True);print('REVIEW_ZIP='+str(archive),flush=True)
