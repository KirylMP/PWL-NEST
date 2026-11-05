"""Run equal-protocol AM/AFM/FM multistart tasks in separate processes."""
import argparse,concurrent.futures,json,os,signal,subprocess,sys,threading
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);p.add_argument('--cases',nargs='+',default=['AM','AFM','FM'],choices=['AM','AFM','FM']);p.add_argument('--seeds',type=int,default=16);p.add_argument('--workers',type=int,default=16);p.add_argument('--max-nfev',type=int,default=80);p.add_argument('--joint-nfev',type=int,default=250);p.add_argument('--start-sigma',type=float,default=.1);p.add_argument('--shape-scale',type=float,default=.1);a=p.parse_args()
    if min(a.seeds,a.workers,a.max_nfev,a.joint_nfev)<1 or min(a.start_sigma,a.shape_scale)<=0:p.error('invalid settings')
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=True)
    manifest={'cases':a.cases,'seeds':a.seeds,'max_nfev':a.max_nfev,'joint_nfev':a.joint_nfev,'start_sigma':a.start_sigma,'shape_scale':a.shape_scale,'scope':'fixed rest, thresholds, fold; lower stable; middle and upper foci; physical calcium; period/amplitudes/aligned waveform'};path=root/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise SystemExit('Existing output has different settings')
    path.write_text(json.dumps(manifest,indent=2)+'\n');env=os.environ.copy()
    for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='1'
    screen=root/'focus_screen.json'
    if not screen.exists():
        cmd=[sys.executable,str(ROOT/'scripts/check_focus_feasibility.py'),'--cases',*a.cases,'--bounds','40','--starts','8','--margin','.01','--output',str(screen)]
        subprocess.run(cmd,cwd=ROOT,env=env,check=True)
    am_seed=root/'shared_seed_AM'
    if 'AM' in a.cases and not (am_seed/'training.json').exists():
        joint=ROOT/'paper/data/focus_fold_seeds/AM/joint.json'
        if not (am_seed/'report.json').exists():
            subprocess.run([sys.executable,str(ROOT/'scripts/materialize_focus_fold_seed.py'),'--case','AM','--joint',str(joint),'--output',str(am_seed)],cwd=ROOT,env=env,check=True)
        subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_focus_seed.py'),'--case','AM','--candidate',str(am_seed/'candidate.json'),'--report',str(am_seed/'report.json'),'--output',str(am_seed/'training.json')],cwd=ROOT,env=env,check=True)
        data=json.loads((am_seed/'training.json').read_text())['results']
        if any(not x['cycle']['candidates'] for x in data):raise SystemExit('AM shared seed failed physical cycle checks')
    task_list=[(case,i) for i in range(a.seeds) for case in a.cases];stop=threading.Event();lock=threading.Lock();children={}
    def run(task):
        case,seed=task;dest=root/case/f'seed_{seed:03d}'
        if stop.is_set():return task,'cancelled'
        if dest.exists():return task,'exists; skipped'
        dest.parent.mkdir(parents=True,exist_ok=True);log=dest.parent/f'seed_{seed:03d}.log'
        cmd=[sys.executable,'-u',str(ROOT/'scripts/run_focus_fit_one.py'),'--case',case,'--seed',str(seed),'--screen',str(screen),'--am-seed-dir',str(am_seed),'--output',str(dest),'--max-nfev',str(a.max_nfev),'--joint-nfev',str(a.joint_nfev),'--start-sigma',str(a.start_sigma),'--shape-scale',str(a.shape_scale)]
        with log.open('x') as f:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            with lock:children[task]=proc
            code=proc.wait()
            with lock:children.pop(task,None)
        status=json.loads((dest/'status.json').read_text()).get('status') if (dest/'status.json').exists() else f'failed({code})'
        return task,status
    old=signal.signal(signal.SIGINT,lambda *_:stop.set())
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
            pending=set(pool.submit(run,t) for t in task_list);terminated=False
            while pending:
                if stop.is_set() and not terminated:
                    for f in pending:f.cancel()
                    with lock:active=list(children.values())
                    for proc in active:
                        try:os.killpg(proc.pid,signal.SIGTERM)
                        except ProcessLookupError:pass
                    terminated=True
                done,pending=concurrent.futures.wait(pending,timeout=1,return_when=concurrent.futures.FIRST_COMPLETED)
                for f in done:
                    if not f.cancelled():
                        try:print(*f.result(),flush=True)
                        except Exception as e:print('runner error',repr(e),flush=True)
    finally:signal.signal(signal.SIGINT,old)
    if stop.is_set():raise SystemExit(130)
    print('Batch finished; inspect per-start status.json and fit/report.json',flush=True)

if __name__=='__main__':main()
