"""Run independent fixed-fold refits with bounded process concurrency.

Complete reports are skipped on restart. Interrupted/failed attempts are kept,
and a fresh attempt directory is used. This is not optimizer-state resumption.
"""
import argparse
import concurrent.futures
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases', nargs='+', default=['AM','AFM','FM'], choices=['AM','AFM','FM'])
    p.add_argument('--arms', nargs='+', default=['three','four'], choices=['three','four'])
    p.add_argument('--seed-start', type=int, default=2)
    p.add_argument('--seed-end', type=int, default=33, help='inclusive')
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--training', choices=['three','all'], default='all')
    p.add_argument('--max-nfev', type=int, default=20)
    p.add_argument('--duration', type=float, default=400.)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dry-run', action='store_true')
    a=p.parse_args()
    if a.seed_start < 0 or a.seed_end < a.seed_start or a.workers < 1 or a.max_nfev < 1 or a.duration <= 0:
        p.error('invalid seed range, workers, max-nfev or duration')
    if len(set(a.cases))!=len(a.cases) or len(set(a.arms))!=len(a.arms):
        p.error('duplicate case or arm')
    tasks=list(itertools.product(a.cases,a.arms,range(a.seed_start,a.seed_end+1)))
    print(f'{len(tasks)} tasks; at most {a.workers} concurrent processes',flush=True)
    if a.dry_run: return
    root=a.output.resolve(); root.mkdir(parents=True,exist_ok=True)
    manifest={'cases':a.cases,'arms':a.arms,'seed_start':a.seed_start,'seed_end':a.seed_end,
              'training':a.training,'max_nfev':a.max_nfev,'duration':a.duration,
              'note':'one process per task; completed tasks are skipped on restart'}
    path=root/'batch_manifest.json'
    if path.exists():
        if json.loads(path.read_text())!=manifest:
            raise SystemExit('Output contains different settings; choose another --output directory.')
    else: path.write_text(json.dumps(manifest,indent=2)+'\n')
    env=os.environ.copy()
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[key]='1'
    def run(task):
        case,arm,seed=task
        parent=root/case/arm/f'seed_{seed}'
        parent.mkdir(parents=True,exist_ok=True)
        for report in parent.glob('attempt_*/report.json'):
            try:
                d=json.loads(report.read_text())
                if d.get('case')==case and d.get('arm')==arm and d.get('seed')==seed:
                    return 'skipped',task,str(report)
            except (ValueError,OSError): pass
        attempt=1
        while (parent/f'attempt_{attempt:03d}').exists() or (parent/f'attempt_{attempt:03d}.log').exists():
            attempt+=1
        dest=parent/f'attempt_{attempt:03d}'; log=parent/f'attempt_{attempt:03d}.log'
        cmd=[sys.executable,'-u',str(ROOT/'scripts/fit_lower_refit_fixed_fold.py'),
             '--case',case,'--arm',arm,'--seed',str(seed),'--training',a.training,
             '--max-nfev',str(a.max_nfev),'--duration',str(a.duration),'--output',str(dest)]
        print('START',case,arm,seed,'attempt',attempt,flush=True)
        with log.open('x') as f:
            proc=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=False)
        success=proc.returncode==0 and (dest/'report.json').is_file()
        return ('done' if success else f'FAILED({proc.returncode})'),task,str(log)
    counts={'done':0,'skipped':0,'failed':0}
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(run,t) for t in tasks]
        for fut in concurrent.futures.as_completed(futures):
            try:
                status,task,log=fut.result()
                counts[status if status in ('done','skipped') else 'failed']+=1
                print(status,*task,log,flush=True)
            except Exception as exc:
                counts['failed']+=1
                print('FAILED runner:',repr(exc),flush=True)
    print('FINISHED',json.dumps(counts),flush=True)
    if counts['failed']: raise SystemExit(1)


if __name__=='__main__': main()
