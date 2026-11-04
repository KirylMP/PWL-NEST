"""Parallel, restartable batch of independent anchored three-zone fits."""
import argparse
import concurrent.futures
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',nargs='+',default=['AM','AFM','FM'],choices=['AM','AFM','FM'])
    p.add_argument('--seed-start',type=int,default=0);p.add_argument('--seed-end',type=int,default=31)
    p.add_argument('--workers',type=int,default=12);p.add_argument('--max-nfev',type=int,default=20)
    p.add_argument('--duration',type=float,default=400.);p.add_argument('--training',choices=['three','all'],default='all')
    p.add_argument('--max-init-tries',type=int,default=20);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    if (a.seed_start<0 or a.seed_end<a.seed_start or a.workers<1 or a.max_nfev<1 or a.duration<=0
        or a.max_init_tries<1 or len(set(a.cases))!=len(a.cases)):p.error('invalid settings')
    tasks=list(itertools.product(a.cases,range(a.seed_start,a.seed_end+1)))
    print(f'{len(tasks)} tasks; at most {a.workers} concurrent processes',flush=True)
    if a.dry_run:return
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=True)
    manifest={'cases':a.cases,'seed_start':a.seed_start,'seed_end':a.seed_end,'max_nfev':a.max_nfev,
              'training':a.training,'duration':a.duration,'max_init_tries':a.max_init_tries,
              'scope':'three zones; fixed rest, equilibrium crossings and fold'}
    manifest_path=root/'batch_manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())!=manifest:raise SystemExit('Output has different settings; choose a new directory.')
    else:manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    env=os.environ.copy()
    for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='1'
    def run(task):
        case,seed=task;parent=root/case/f'seed_{seed}';parent.mkdir(parents=True,exist_ok=True)
        for report in parent.glob('attempt_*/report.json'):
            try:
                d=json.loads(report.read_text())
                if d.get('case')==case and d.get('seed')==seed:return 'skipped',task,str(report)
            except (OSError,ValueError):pass
        n=1
        while (parent/f'attempt_{n:03d}').exists() or (parent/f'attempt_{n:03d}.log').exists():n+=1
        dest=parent/f'attempt_{n:03d}';log=parent/f'attempt_{n:03d}.log'
        cmd=[sys.executable,'-u',str(ROOT/'scripts/fit_anchored_three_zone.py'),
             '--case',case,'--seed',str(seed),'--training',a.training,'--max-nfev',str(a.max_nfev),
             '--duration',str(a.duration),'--max-init-tries',str(a.max_init_tries),'--output',str(dest)]
        print('START',case,seed,'attempt',n,flush=True)
        with log.open('x') as f:proc=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=False)
        return ('done' if proc.returncode==0 and (dest/'report.json').is_file() else f'FAILED({proc.returncode})'),task,str(log)
    counts={'done':0,'skipped':0,'failed':0}
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        for future in concurrent.futures.as_completed([pool.submit(run,t) for t in tasks]):
            try:
                status,task,path=future.result();counts[status if status in ('done','skipped') else 'failed']+=1
                print(status,*task,path,flush=True)
            except Exception as exc:counts['failed']+=1;print('FAILED runner:',repr(exc),flush=True)
    print('FINISHED',json.dumps(counts),flush=True)
    if counts['failed']:raise SystemExit(1)


if __name__=='__main__':main()
