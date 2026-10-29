#!/usr/bin/env python3
"""Parallel independent physical-calcium fits with catalogued winners and logs."""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime,timezone
from pathlib import Path

LAB=Path(__file__).resolve().parents[1]
SCRIPTS=LAB/'scripts'
SEEDS={
    'am_identity':'am_2thr_traj_identity_20261029',
    'am_power':'am_2thr_traj_power_20261029',
    'fm_identity':'fm_2thr_traj_identity_20261029',
    'fm_power':'fm_2thr_traj_power_20261029',
}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir',type=Path,required=True);ap.add_argument('--trials',type=int,default=12)
    ap.add_argument('--parallel',type=int,default=16);ap.add_argument('--max-nfev',type=int,default=2000)
    ap.add_argument('--duration',type=float,default=300.);ap.add_argument('--dt',type=float,default=.025)
    ap.add_argument('--cmax-scale',type=float,default=.08);ap.add_argument('--geometry-weight',type=float,default=.02)
    ap.add_argument('--validation-workers',type=int,default=8);ap.add_argument('--validation-duration',type=float,default=300.)
    a=ap.parse_args()
    if min(a.trials,a.parallel,a.max_nfev,a.validation_workers)<1 or a.duration<60 or a.dt<=0 or a.cmax_scale<=0 or a.geometry_weight<0:ap.error('Invalid settings')
    run=a.run_dir.resolve();run.mkdir(parents=True,exist_ok=False);(run/'logs').mkdir();(run/'trials').mkdir()
    gate_min=float(os.environ.get('GATE_POWER_MIN','0.2'));gate_max=float(os.environ.get('GATE_POWER_MAX','2.5'))
    if not 0<gate_min<=1<=gate_max or gate_min>=gate_max:ap.error('Gate power bounds must contain 1')
    config=dict(trials_per_model=a.trials,parallel=a.parallel,max_nfev=a.max_nfev,duration=a.duration,dt=a.dt,cmax_scale=a.cmax_scale,geometry_weight=a.geometry_weight,gate_power_bounds=[gate_min,gate_max],validation_workers=a.validation_workers,validation_duration=a.validation_duration,started_utc=datetime.now(timezone.utc).isoformat(),seeds={k:str((LAB/'models'/v).resolve()) for k,v in SEEDS.items()})
    (run/'config.json').write_text(json.dumps(config,indent=2)+'\n');print(f'RUN_DIR={run}',flush=True)
    for name,path in config['seeds'].items():
        for suffix in ('parameters.json','trajectory_comparison.json'):
            if not (Path(path)/suffix).is_file():raise ValueError(f'Missing {suffix} for {name}: {path}')
    tasks=[(name,j) for name in SEEDS for j in range(a.trials)]
    lock=threading.Lock();done=0;started=time.monotonic();outcomes=[]
    def trial(task):
        nonlocal done
        name,j=task;out=run/'trials'/name/f'trial_{j:03d}';log=run/'logs'/f'{name}_trial_{j:03d}.log';log.parent.mkdir(parents=True,exist_ok=True)
        jitter=0. if j==0 else (.08,.16,.25)[(j-1)%3]
        cmd=[sys.executable,'-u',str(SCRIPTS/'refine_physical_trajectories.py'),config['seeds'][name],'--out',str(out),'--duration',str(a.duration),'--dt',str(a.dt),'--max-nfev',str(a.max_nfev),'--geometry-weight',str(a.geometry_weight),'--cmax-scale',str(a.cmax_scale),'--seed-jitter',str(jitter),'--random-seed',str(20261029+j)]
        env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PHYSICAL_FIELD_WEIGHT='10' if name.startswith('am_') else '1',GATE_POWER_MIN=str(gate_min),GATE_POWER_MAX=str(gate_max))
        with log.open('w') as handle:
            handle.write('COMMAND '+repr(cmd)+'\n');handle.flush();proc=subprocess.run(cmd,stdout=handle,stderr=subprocess.STDOUT,env=env)
        item=dict(model=name,trial=j,jitter=jitter,returncode=proc.returncode,log=str(log),output=str(out));candidate=out/'parameters.json'
        if proc.returncode==0 and candidate.exists():
            try:
                obj=json.loads(candidate.read_text());score=float(obj['trajectory_fit']['loss'])
                if math.isfinite(score) and (not name.startswith('fm_') or obj.get('fold_constraint')):
                    item.update(loss=score,optimizer=obj['trajectory_fit']['optimizer'],fold_verified=bool(obj.get('fold_constraint')))
                else:item['error']='Nonfinite loss or missing FM fold'
            except (ValueError,KeyError,OSError) as e:item['error']=str(e)
        else:item['error']='Trial failed or has no parameter file'
        with lock:
            done+=1;outcomes.append(item)
            (run/'progress.json').write_text(json.dumps(dict(completed=done,total=len(tasks),elapsed_s=time.monotonic()-started,outcomes=outcomes),indent=2)+'\n')
            print(f'FINISHED {done}/{len(tasks)} {name} trial={j:03d} loss={item.get("loss")} rc={proc.returncode} elapsed={(time.monotonic()-started)/60:.1f}m log={log}',flush=True)
        return item
    with ThreadPoolExecutor(max_workers=a.parallel) as pool:
        futures=[pool.submit(trial,t) for t in tasks]
        for future in as_completed(futures):future.result()
    winners={}
    for name in SEEDS:
        valid=[v for v in outcomes if v['model']==name and 'loss' in v]
        if not valid:print(f'NO VALID WINNER {name}',flush=True);continue
        winner=min(valid,key=lambda x:x['loss']);dest=run/'winners'/name;dest.mkdir(parents=True,exist_ok=True)
        shutil.copy2(Path(winner['output'])/'parameters.json',dest/'parameters.json');shutil.copy2(winner['log'],dest/'fit.log')
        (dest/'winner.json').write_text(json.dumps(winner,indent=2)+'\n');winners[name]=dict(winner=winner,path=str(dest))
        print(f'WINNER {name}: trial={winner["trial"]} loss={winner["loss"]:.7g}',flush=True)
    validation={}
    for case in ('am','fm'):
        left,right=(f'{case}_identity',f'{case}_power')
        if left not in winners or right not in winners:continue
        for variant,inputs in [('training',None),('holdout','.37,.44,.52,.60' if case=='am' else '.575,.725,.875,1.05')]:
            dest=run/'validation'/case/variant;log=run/'logs'/f'{case}_validation_{variant}.log'
            cmd=[sys.executable,'-u',str(SCRIPTS/'compare_physical_calcium.py'),winners[left]['path'],winners[right]['path'],'--out',str(dest),'--duration',str(a.validation_duration),'--workers',str(a.validation_workers)]
            if inputs:cmd+=['--inputs',inputs]
            env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR=str(run/'matplotlib_cache'),XDG_CACHE_HOME=str(run/'cache'))
            print(f'VALIDATE {case} {variant} log={log}',flush=True)
            with log.open('w') as handle:proc=subprocess.run(cmd,stdout=handle,stderr=subprocess.STDOUT,env=env)
            validation[f'{case}_{variant}']=dict(returncode=proc.returncode,output=str(dest),log=str(log))
            if proc.returncode:
                print(f'VALIDATION FAILED {case} {variant}; see {log}',flush=True);continue
            prefix='trajectory_comparison' if variant=='training' else 'holdout_comparison'
            for name in (left,right):
                for ext in ('json','png'):shutil.copy2(dest/f'trajectory_comparison.{ext}',Path(winners[name]['path'])/f'{prefix}.{ext}')
    run_label=run.parent.name if run.name=='run' else run.name
    for name,entry in winners.items():
        dest=LAB/'models'/f'{name}_2thr_{run_label}';dest.parent.mkdir(exist_ok=True)
        if dest.exists():raise FileExistsError(dest)
        shutil.copytree(entry['path'],dest);entry['catalog_path']=str(dest)
        print(f'MODEL {name}: {dest}',flush=True)
    summary=dict(config=config,winners=winners,validation=validation,failed_trials=[x for x in outcomes if 'error' in x],note='Best by common finite-trajectory training loss; independent fixed-input DOP853 validation is not a complete cycle or bifurcation analysis. No NEST benchmark.')
    (run/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(f'SUMMARY={run / "summary.json"}',flush=True)
    if len(winners)!=4 or any(v['returncode'] for v in validation.values()):raise SystemExit('Some models or validations failed; inspect summary and logs.')

if __name__=='__main__':main()
