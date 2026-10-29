#!/usr/bin/env python3
"""Run independent free-latent AM/FM fits; save logs and validation."""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path

LAB=Path(__file__).resolve().parents[1];SCRIPTS=LAB/'scripts'
SEEDS={'am':'am_2thr_traj_identity_20261029','fm':'fm_2thr_traj_identity_20261029'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--trials',type=int,default=12);p.add_argument('--parallel',type=int,default=16)
    p.add_argument('--max-nfev',type=int,default=2000);p.add_argument('--duration',type=float,default=300.);p.add_argument('--dt',type=float,default=.025)
    p.add_argument('--cmax-scale',type=float,default=.08);p.add_argument('--equilibrium-weight',type=float,default=.02);p.add_argument('--validation-workers',type=int,default=8)
    a=p.parse_args()
    if min(a.trials,a.parallel,a.max_nfev,a.validation_workers)<1 or a.duration<60 or a.dt<=0 or a.cmax_scale<=0 or a.equilibrium_weight<0:p.error('Invalid settings')
    run=a.run_dir.resolve();run.mkdir(parents=True,exist_ok=False);(run/'logs').mkdir();(run/'trials').mkdir()
    config=dict(trials_per_model=a.trials,parallel=a.parallel,max_nfev=a.max_nfev,duration=a.duration,dt=a.dt,cmax_scale=a.cmax_scale,equilibrium_weight=a.equilibrium_weight,started_utc=datetime.now(timezone.utc).isoformat(),seeds={k:str((LAB/'models'/v).resolve()) for k,v in SEEDS.items()})
    (run/'config.json').write_text(json.dumps(config,indent=2)+'\n');print(f'RUN_DIR={run}',flush=True)
    for seed in config['seeds'].values():
        for suffix in ('parameters.json','trajectory_comparison.json'):
            if not (Path(seed)/suffix).is_file():raise ValueError(f'Missing {suffix} in {seed}')
    tasks=[(case,i) for case in SEEDS for i in range(a.trials)];outcomes=[];lock=threading.Lock();start=time.monotonic()
    def trial(task):
        case,j=task;out=run/'trials'/case/f'trial_{j:03d}';log=run/'logs'/f'{case}_trial_{j:03d}.log';jitter=0. if j==0 else (.08,.16,.25)[(j-1)%3]
        cmd=[sys.executable,'-u',str(SCRIPTS/'fit_free_latent.py'),config['seeds'][case],'--out',str(out),'--duration',str(a.duration),'--dt',str(a.dt),'--max-nfev',str(a.max_nfev),'--cmax-scale',str(a.cmax_scale),'--equilibrium-weight',str(a.equilibrium_weight),'--jitter',str(jitter),'--random-seed',str(20261029+j)]
        env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        with log.open('w') as handle:handle.write('COMMAND '+repr(cmd)+'\n');handle.flush();proc=subprocess.run(cmd,stdout=handle,stderr=subprocess.STDOUT,env=env)
        item=dict(case=case,trial=j,jitter=jitter,returncode=proc.returncode,output=str(out),log=str(log));path=out/'parameters.json'
        if proc.returncode==0 and path.exists():
            try:
                d=json.loads(path.read_text());loss=float(d['trajectory_fit']['loss'])
                if math.isfinite(loss) and (case!='fm' or d.get('fold_constraint')):item.update(loss=loss,optimizer=d['trajectory_fit']['optimizer'])
                else:item['error']='Nonfinite loss or FM fold missing'
            except (OSError,ValueError,KeyError) as e:item['error']=str(e)
        else:item['error']='Trial failed or has no final candidate'
        with lock:
            outcomes.append(item);(run/'progress.json').write_text(json.dumps(dict(completed=len(outcomes),total=len(tasks),elapsed_s=time.monotonic()-start,outcomes=outcomes),indent=2)+'\n')
            print(f'FINISHED {len(outcomes)}/{len(tasks)} {case} trial={j:03d} loss={item.get("loss")} rc={proc.returncode} elapsed={(time.monotonic()-start)/60:.1f}m log={log}',flush=True)
        return item
    with ThreadPoolExecutor(max_workers=a.parallel) as pool:
        for fut in as_completed([pool.submit(trial,t) for t in tasks]):fut.result()
    winners={};validation={}
    for case in SEEDS:
        available=[r for r in outcomes if r['case']==case and 'loss' in r]
        if not available:print(f'NO VALID WINNER {case}',flush=True);continue
        winner=min(available,key=lambda r:r['loss']);out=run/'winners'/case;out.mkdir(parents=True)
        shutil.copy2(Path(winner['output'])/'parameters.json',out/'parameters.json');shutil.copy2(winner['log'],out/'fit.log')
        (out/'winner.json').write_text(json.dumps(winner,indent=2)+'\n');winners[case]=dict(winner=winner,path=str(out))
        print(f'WINNER {case} trial={winner["trial"]} loss={winner["loss"]:.7g}',flush=True)
        for kind,inputs in [('training',None),('holdout','.37,.44,.52,.60' if case=='am' else '.575,.725,.875,1.05')]:
            dest=run/'validation'/case/kind;log=run/'logs'/f'{case}_{kind}_validation.log'
            cmd=[sys.executable,'-u',str(SCRIPTS/'compare_free_latent.py'),str(out),config['seeds'][case],'--out',str(dest),'--duration','300','--workers',str(a.validation_workers)]
            if inputs:cmd+=['--inputs',inputs]
            env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR=str(run/'matplotlib_cache'),XDG_CACHE_HOME=str(run/'cache'))
            print(f'VALIDATE {case} {kind} log={log}',flush=True)
            with log.open('w') as handle:proc=subprocess.run(cmd,stdout=handle,stderr=subprocess.STDOUT,env=env)
            validation[f'{case}_{kind}']=dict(returncode=proc.returncode,output=str(dest),log=str(log))
            if proc.returncode:print(f'VALIDATION FAILED {case} {kind}: {log}',flush=True);continue
            for ext in ('json','png'):shutil.copy2(dest/f'comparison.{ext}',out/f'{kind}_comparison.{ext}')
        label=run.parent.name if run.name=='run' else run.name;catalog=LAB/'models'/f'{case}_2thr_free_{label}'
        if catalog.exists():raise FileExistsError(catalog)
        shutil.copytree(out,catalog);winners[case]['catalog_path']=str(catalog);print(f'MODEL {case}: {catalog}',flush=True)
    summary=dict(config=config,winners=winners,validation=validation,failed_trials=[r for r in outcomes if 'error' in r],note='Finite trajectory loss. Hidden q is unrestricted except numerical runaway guards. No Poincare or NEST; fixed-input DOP853 comparisons are not complete cycle analysis.')
    (run/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(f'SUMMARY={run / "summary.json"}',flush=True)
    if len(winners)!=2 or any(r['returncode'] for r in validation.values()):raise SystemExit('Some models or validations failed; inspect summary and logs.')

if __name__=='__main__':main()
