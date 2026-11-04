"""Parallel, restartable batch of independent robust anchored three-zone fits."""
import argparse
import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',nargs='+',default=['AM','AFM','FM'],choices=['AM','AFM','FM'])
    p.add_argument('--seed-start',type=int,default=0);p.add_argument('--seed-end',type=int,default=31)
    p.add_argument('--workers',type=int,default=16);p.add_argument('--max-nfev',type=int,default=80)
    p.add_argument('--duration',type=float,default=400.);p.add_argument('--bound',type=float,default=4.)
    p.add_argument('--max-init-tries',type=int,default=20);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    if (a.seed_start<0 or a.seed_end<a.seed_start or a.workers<1 or a.max_nfev<1 or a.duration<=0
        or a.max_init_tries<1 or a.bound<=2.1 or len(set(a.cases))!=len(a.cases)):p.error('invalid settings')
    # Interleave cases so AM, AFM and FM start together instead of in blocks.
    tasks=[(case,seed) for seed in range(a.seed_start,a.seed_end+1) for case in a.cases]
    print(f'{len(tasks)} tasks; at most {a.workers} concurrent processes',flush=True)
    if a.dry_run:return
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=True)
    manifest={'cases':a.cases,'seed_start':a.seed_start,'seed_end':a.seed_end,'max_nfev':a.max_nfev,
              'duration':a.duration,'bound':a.bound,'max_init_tries':a.max_init_tries,
              'scope':'warm/fallback fold shooting; three zones; fixed rest, equilibrium crossings and fold'}
    manifest_path=root/'batch_manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())!=manifest:raise SystemExit('Output has different settings; choose a new directory.')
    else:manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    env=os.environ.copy()
    for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='1'
    stopping=threading.Event();lock=threading.Lock();children={}
    def run(task):
        if stopping.is_set():return 'cancelled',task,''
        case,seed=task;parent=root/case/f'seed_{seed}';parent.mkdir(parents=True,exist_ok=True)
        for report in parent.glob('attempt_*/report.json'):
            try:
                d=json.loads(report.read_text())
                if d.get('case')==case and d.get('seed')==seed:return 'skipped',task,str(report)
            except (OSError,ValueError):pass
        n=1
        while (parent/f'attempt_{n:03d}').exists() or (parent/f'attempt_{n:03d}.log').exists():n+=1
        dest=parent/f'attempt_{n:03d}';log=parent/f'attempt_{n:03d}.log'
        cmd=[sys.executable,'-u',str(ROOT/'scripts/fit_robust_anchored.py'),
             '--case',case,'--seed',str(seed),'--bound',str(a.bound),'--max-nfev',str(a.max_nfev),
             '--duration',str(a.duration),'--max-init-tries',str(a.max_init_tries),'--output',str(dest)]
        if stopping.is_set():return 'cancelled',task,''
        print('START',case,seed,'attempt',n,flush=True)
        with log.open('x') as f:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            with lock:children[task]=proc
            if stopping.is_set():
                try:proc.terminate()
                except ProcessLookupError:pass
            code=proc.wait()
            with lock:children.pop(task,None)
        if code==0 and (dest/'report.json').is_file():return 'done',task,str(log)
        return ('cancelled' if stopping.is_set() else f'FAILED({code})'),task,str(log)
    counts={'done':0,'skipped':0,'failed':0,'cancelled':0}
    old_handler=signal.signal(signal.SIGINT,lambda _sig,_frame:stopping.set())
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
            pending=set(pool.submit(run,t) for t in tasks);terminated=False
            while pending:
                if stopping.is_set() and not terminated:
                    print('Stopping: cancelling queued tasks and terminating active workers',flush=True)
                    for future in pending:future.cancel()
                    with lock:active=list(children.values())
                    for proc in active:
                        try:proc.terminate()
                        except ProcessLookupError:pass
                    terminated=True
                done,pending=concurrent.futures.wait(pending,timeout=1,return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    if future.cancelled():counts['cancelled']+=1;continue
                    try:
                        status,task,path=future.result()
                        counts[status if status in ('done','skipped','cancelled') else 'failed']+=1
                        print(status,*task,path,flush=True)
                    except Exception as exc:counts['failed']+=1;print('FAILED runner:',repr(exc),flush=True)
    finally:signal.signal(signal.SIGINT,old_handler)
    print('STOPPED' if stopping.is_set() else 'FINISHED',json.dumps(counts),flush=True)
    if stopping.is_set():raise SystemExit(130)
    if counts['failed']:raise SystemExit(1)


if __name__=='__main__':main()
