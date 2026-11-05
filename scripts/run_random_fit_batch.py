"""Generate independent random starting fields until candidate quotas are met."""
import argparse,concurrent.futures,json,os,signal,subprocess,sys,threading
from collections import Counter
from pathlib import Path
from run_until_candidates import eligible,scan
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);p.add_argument('--cases',nargs='+',default=['AM','AFM','FM'],choices=['AM','AFM','FM']);p.add_argument('--target',type=int,default=64);p.add_argument('--workers',type=int,default=16);p.add_argument('--max-nfev',type=int,default=80);p.add_argument('--joint-nfev',type=int,default=300);p.add_argument('--max-field-draws',type=int,default=1000);p.add_argument('--bound',type=float,default=40.);p.add_argument('--shape-scale',type=float,default=.1);p.add_argument('--max-new-attempts',type=int,default=0,help='per-case cap in this invocation; 0 means unlimited');a=p.parse_args()
    if min(a.target,a.workers,a.max_nfev,a.joint_nfev,a.max_field_draws)<1 or min(a.bound,a.shape_scale)<=0 or a.max_new_attempts<0 or len(set(a.cases))!=len(a.cases):p.error('invalid settings')
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=True);manifest_path=root/'manifest.json'
    manifest={'protocol':'independent_random_fields_v1','cases':a.cases,'max_nfev':a.max_nfev,'joint_nfev':a.joint_nfev,'max_field_draws':a.max_field_draws,'bound':a.bound,'shape_scale':a.shape_scale,'loss':'same Cmin Cmax period normalized waveform; physicality hard reject','note':'fixed a,b, thresholds, equilibrium and fold targets; initial x uniform bounded and flight times independently random'}
    if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:raise SystemExit('Output contains different settings; choose a new directory')
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n');env=os.environ.copy()
    for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='1'
    counts={};next_id={};launched=Counter();running=Counter();children={};lock=threading.Lock();stop=threading.Event()
    for case in a.cases:counts[case],next_id[case]=scan(root,case)
    print('Existing valid:',counts,'target:',a.target,'max new attempts per case:',a.max_new_attempts or 'unlimited',flush=True)
    def run(case,seed):
        dest=root/case/f'seed_{seed:03d}';dest.parent.mkdir(parents=True,exist_ok=True);log=dest.parent/f'seed_{seed:03d}.log'
        cmd=[sys.executable,'-u',str(ROOT/'scripts/run_random_fit_one.py'),'--case',case,'--seed',str(seed),'--output',str(dest),'--max-nfev',str(a.max_nfev),'--joint-nfev',str(a.joint_nfev),'--max-field-draws',str(a.max_field_draws),'--bound',str(a.bound),'--shape-scale',str(a.shape_scale)]
        with log.open('x') as f:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            with lock:children[(case,seed)]=proc
            code=proc.wait()
            with lock:children.pop((case,seed),None)
        try:status=json.loads((dest/'status.json').read_text())['status']
        except (OSError,KeyError,ValueError):status=f'failed({code})'
        return case,seed,status,eligible(dest,case)
    old=signal.signal(signal.SIGINT,lambda *_:stop.set());pending={};cursor=0;terminated=False
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
            while True:
                if stop.is_set() and not terminated:
                    for f in pending:f.cancel()
                    with lock:active=list(children.values())
                    for proc in active:
                        try:os.killpg(proc.pid,signal.SIGTERM)
                        except ProcessLookupError:pass
                    terminated=True
                while not stop.is_set() and len(pending)<a.workers:
                    choice=None
                    for n in range(len(a.cases)):
                        k=(cursor+n)%len(a.cases);case=a.cases[k]
                        if counts[case]+running[case]<a.target and (a.max_new_attempts==0 or launched[case]<a.max_new_attempts):choice=case;cursor=(k+1)%len(a.cases);break
                    if choice is None:break
                    seed=next_id[choice];next_id[choice]+=1;launched[choice]+=1;running[choice]+=1;future=pool.submit(run,choice,seed);pending[future]=(choice,seed)
                    print('START',choice,seed,'valid',counts[choice],'running',running[choice],flush=True)
                if not pending:break
                done,_=concurrent.futures.wait(pending,timeout=1,return_when=concurrent.futures.FIRST_COMPLETED)
                for f in done:
                    case,seed=pending.pop(f);running[case]-=1
                    if f.cancelled():continue
                    try:
                        _,_,status,ok=f.result()
                        if ok:counts[case]+=1
                        print('DONE',case,seed,status,'accepted' if ok else 'not accepted',f'{counts[case]}/{a.target}',flush=True)
                    except Exception as e:print('FAILED',case,seed,repr(e),flush=True)
    finally:signal.signal(signal.SIGINT,old)
    print('FINAL',json.dumps({'eligible':counts,'new_attempts':dict(launched),'target':a.target}),flush=True)
    if stop.is_set():raise SystemExit(130)
    if any(counts[c]<a.target for c in a.cases):raise SystemExit('Target not reached; inspect failures or raise attempt cap')

if __name__=='__main__':main()
