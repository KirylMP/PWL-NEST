#!/usr/bin/env python3
"""Fit fixed Taylor/Chebyshev calcium maps with independent finite validation."""
import argparse, concurrent.futures, csv, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
from poly_model import build_map

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'astrocyte_lab/experiments/fm2_amplitude_refine'))
from run_server import stats


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=16)
 p.add_argument('--degrees',type=int,nargs='+',default=[2,3,4,5,6]);p.add_argument('--modes',nargs='+',choices=['am','afm','fm'],default=['am','afm','fm'])
 p.add_argument('--starts',type=int,default=2,help='starts per source seed; mixed and physical sources')
 p.add_argument('--jitter',type=float,default=.025);p.add_argument('--powell-maxfev',type=int,default=6000);p.add_argument('--ls-max-nfev',type=int,default=400)
 p.add_argument('--duration',type=float,default=350);p.add_argument('--validation-duration',type=float,default=450);p.add_argument('--dt',type=float,default=.01)
 p.add_argument('--fm-mixed',type=Path,help='optional final server mixed FM parameters.json')
 a=p.parse_args()
 if min(a.workers,a.starts,a.powell_maxfev,a.ls_max_nfev)<1 or len(set(a.degrees))!=len(a.degrees) or any(x<2 or x>8 for x in a.degrees) or min(a.duration,a.validation_duration)<100 or a.dt<=0 or a.jitter<0:raise ValueError('Invalid options')
 out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
 env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR=str(out/'mpl'),XDG_CACHE_HOME=str(out/'cache'))
 engine=out/'shape_engine.so';subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(HERE/'shape_engine.cpp'),'-o',str(engine)],check=True)
 sources={}
 for mode in a.modes:
  for coord in ('physical','mixed'):
   path=ROOT/f'paper/local_results/models/{mode}_{coord}/parameters.json'
   if mode=='fm' and coord=='mixed' and a.fm_mixed:path=a.fm_mixed.resolve()
   if not path.is_file():raise FileNotFoundError(path)
   sources[mode,coord]=path
 tasks=[];maps=[]
 for mode in a.modes:
  mixed=sources[mode,'mixed'];seed=json.loads(mixed.read_text())
  for family in ('taylor','chebyshev'):
   for degree in a.degrees:
    dest=out/mode/family/f'degree_{degree:02d}';dest.mkdir(parents=True,exist_ok=True)
    try:
     spec=build_map(seed,family,degree);spec.update(source=str(mixed),source_sha256=sha(mixed),case=mode)
     path=dest/'map.json';path.write_text(json.dumps(spec,indent=2,allow_nan=False)+'\n')
     maps.append(dict(mode=mode,family=family,degree=degree,status='monotone',path=str(path),min_derivative=spec['min_derivative'],max_abs_z_error=spec['max_abs_z_error'],rms_z_error=spec['rms_z_error']))
     for coord in ('mixed','physical'):
      for start in range(a.starts):tasks.append((mode,family,degree,path,coord,start))
    except (ValueError,FloatingPointError,OverflowError) as exc:
     maps.append(dict(mode=mode,family=family,degree=degree,status='nonmonotone_or_invalid',reason=str(exc)))
     print('MAP rejected',mode,family,degree,str(exc),flush=True)
 print(f'FIT {len(tasks)} starts, workers={a.workers}, output={out}',flush=True)
 def fit(task):
  mode,family,degree,mapfile,coord,start=task
  dest=mapfile.parent/f'{coord}_start_{start:02d}';dest.mkdir(parents=True,exist_ok=True)
  cmd=[sys.executable,'-u',str(HERE/'refine.py'),'--map-json',str(mapfile),'--seed-model',str(sources[mode,coord]),'--engine',str(engine),'--out',str(dest),'--powell-maxfev',str(a.powell_maxfev),'--ls-max-nfev',str(a.ls_max_nfev),'--duration',str(a.duration),'--dt',str(a.dt),'--peak-weight',str(1 if mode=='fm' else 0),'--jitter',str(0 if start==0 else a.jitter),'--random-seed',str(52000+1000*degree+start)]
  with (dest/'fit.log').open('w') as f:r=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
  path=dest/'parameters.json';data=json.loads(path.read_text()) if path.exists() else {}
  row=dict(mode=mode,family=family,degree=degree,seed=coord,start=start,seed_sha256=sha(sources[mode,coord]),exit_code=r.returncode,status=data.get('status','process_error'),loss=data.get('loss'),path=str(path),log=str(dest/'fit.log'))
  print('FIT RESULT',json.dumps(row),flush=True);return row
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:trials=list(pool.map(fit,tasks))
 candidates=[]
 for m in maps:
  if m['status']!='monotone':continue
  valid=[r for r in trials if all(r[k]==m[k] for k in ('mode','family','degree')) and r['exit_code']==0 and r['status']=='training_cycles_found' and r['loss'] is not None]
  if not valid:m['fit_status']='no_candidate';continue
  winner=min(valid,key=lambda r:r['loss']);m['fit_status']='candidate';m['winner']=winner
  dest=out/m['mode']/m['family']/f"degree_{m['degree']:02d}"/'selected';dest.mkdir(parents=True,exist_ok=True);shutil.copy2(winner['path'],dest/'parameters.json');candidates.append((m,dest))
 def validate(item):
  meta,dest=item
  with (dest/'validate.log').open('w') as f:r=subprocess.run([sys.executable,'-u',str(HERE/'validate.py'),str(dest/'parameters.json'),'--duration',str(a.validation_duration)],stdout=f,stderr=subprocess.STDOUT,env=env)
  path=dest/'validation.json';row=dict(exit_code=r.returncode,path=str(path),plot=str(dest/'comparison.png'),log=str(dest/'validate.log'))
  if r.returncode==0 and path.is_file():row['metrics']=stats(path)
  print('VALIDATION',meta['mode'],meta['family'],meta['degree'],json.dumps(row),flush=True);return row
 with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
  for (m,_),result in zip(candidates,pool.map(validate,candidates)):m['validation']=result
 summary=dict(scope='Fixed polynomial approximation of mixed map, followed by free PWA parameter fitting. No fold condition or exhaustive cycle scan.',settings=dict(workers=a.workers,degrees=a.degrees,modes=a.modes,starts_per_seed=a.starts,jitter=a.jitter,powell_maxfev=a.powell_maxfev,ls_max_nfev=a.ls_max_nfev,duration=a.duration,validation_duration=a.validation_duration,dt=a.dt),sources={f'{mode}_{coord}':dict(path=str(path),sha256=sha(path)) for (mode,coord),path in sources.items()},maps=maps,trials=trials)
 (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
 fields=['mode','family','degree','status','min_derivative','max_abs_z_error','train_loss','validation_exit','train_valid','train_total','train_period','train_shape','holdout_valid','holdout_total','holdout_period','holdout_shape','model']
 with (out/'results.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for m in maps:
   v=m.get('validation',{});metrics=v.get('metrics',{});tr=metrics.get('train',{});ho=metrics.get('holdout',{})
   w.writerow(dict(mode=m['mode'],family=m['family'],degree=m['degree'],status=m.get('fit_status',m['status']),min_derivative=m.get('min_derivative'),max_abs_z_error=m.get('max_abs_z_error'),train_loss=m.get('winner',{}).get('loss'),validation_exit=v.get('exit_code'),train_valid=tr.get('valid'),train_total=tr.get('total'),train_period=tr.get('mean_abs_period'),train_shape=tr.get('mean_abs_shape_rmse'),holdout_valid=ho.get('valid'),holdout_total=ho.get('total'),holdout_period=ho.get('mean_abs_period'),holdout_shape=ho.get('mean_abs_shape_rmse'),model=m.get('winner',{}).get('path')))
 print('SAVED',out/'summary.json','and',out/'results.csv',flush=True)
 missing=[(mode,family) for mode in a.modes for family in ('taylor','chebyshev') if not any(m['mode']==mode and m['family']==family and m.get('validation',{}).get('exit_code')==0 for m in maps)]
 if missing or any(m.get('fit_status')=='no_candidate' or m.get('validation',{}).get('exit_code',0)!=0 for m in maps):raise SystemExit(f'Some families lack successful fits or viable maps failed: missing={missing}; inspect results.csv and logs')
if __name__=='__main__':main()
