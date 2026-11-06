"""Run independent fixed-fold AM reset-constrained fits in parallel."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--starts',type=int,default=16);p.add_argument('--workers',type=int,default=16);p.add_argument('--feas-nfev',type=int,default=120);p.add_argument('--fit-iterations',type=int,default=50);p.add_argument('--jitter',type=float,default=.5);a=p.parse_args()
    if min(a.starts,a.workers,a.feas_nfev,a.fit_iterations)<1 or a.jitter<0:p.error('invalid settings')
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=True)
    settings={'scope':'AM original physical 3-zone fixed-fold; reset positivity hard numerical constraint',
              'starts':a.starts,'feas_nfev':a.feas_nfev,'fit_iterations':a.fit_iterations,'jitter':a.jitter,
              'source_seeds':[465,2311,3080,3420],'high_inputs':[.75,1.,1.2],'hold_s':[20.,40.,60.,80.]}
    manifest=root/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=settings:raise SystemExit('output settings differ: choose another directory')
    manifest.write_text(json.dumps(settings,indent=2)+'\n');env=os.environ.copy()
    for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='1'
    def run(seed):
        dest=root/f'seed_{seed:03}';log=root/f'seed_{seed:03}.log';source=[465,2311,3080,3420][seed%4]
        if (dest/'status.json').exists():return seed,'already completed'
        if dest.exists() or log.exists():return seed,'incomplete; retained; choose another output or explicitly resume'
        cmd=[sys.executable,'-u',str(ROOT/'scripts/fit_am_reset_feasible.py'),'--source-seed',str(source),'--seed',str(seed),
             '--output',str(dest),'--feas-nfev',str(a.feas_nfev),'--fit-iterations',str(a.fit_iterations),'--jitter',str(a.jitter)]
        print('START',seed,'source',source,flush=True)
        with log.open('x') as f:code=subprocess.run(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
        try:status=json.loads((dest/'status.json').read_text())['status']
        except (OSError,KeyError,ValueError):status=f'failed({code}); see {log}'
        return seed,status
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(run,i) for i in range(a.starts)]):print('DONE',*f.result(),flush=True)
    print('Finished. Run summarize_am_reset.py; no feasible start is not proof of impossibility.',flush=True)
if __name__=='__main__':main()
