"""Continue a compatible batch until each case has a target number of candidates.

A candidate is a completed fit with a valid sampled physical training set,
matching fold residual and required spectra. Holdout is reported separately and
never used to choose or count a candidate. Existing results are never overwritten.
"""
import argparse,concurrent.futures,json,math,os,re,signal,subprocess,sys,threading
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def eligible(folder,case):
    try:
        status=json.loads((folder/'status.json').read_text())
        if status['case']!=case or status['status']!='completed':return False
        r=json.loads((folder/'fit/report.json').read_text());b=r['best'];fold=b['fold'];T=b['trace'];delta=b['discriminant'];cycles=b['cycles']
        candidate=json.loads((folder/'fit/candidate.json').read_text())
        if r['case']!=case or candidate['case'] not in ({'AFM','INTERMEDIATE'} if case=='AFM' else {case}) or len(cycles)!=7:return False
        values=[b['loss'],*T,*delta,fold['matching_max'],fold['logM'],fold['Cmin_sampled']]
        if not all(math.isfinite(x) for x in values):return False
        if not(T[0]<0<T[1] and T[2]<0 and delta[1]<-1e-6 and delta[2]<-1e-6):return False
        if candidate['determinant'][0]<=1e-8 or fold['Cmin_sampled']<=0:return False
        if abs(fold['matching_max'])>2e-7 or abs(fold['logM'])>2e-7:return False
        return all(math.isfinite(v['Cmin']) and v['Cmin']>0 and math.isfinite(v['logM']) and v['logM']<0 and len(v['shape'])==64 for v in cycles)
    except (OSError,KeyError,TypeError,ValueError,IndexError):return False


def scan(root,case):
    directories=list((root/case).glob('seed_*')) if (root/case).is_dir() else []
    ids=[int(m.group(1)) for p in directories if p.is_dir() and (m:=re.fullmatch(r'seed_(\d+)',p.name))]
    logs=[int(m.group(1)) for p in (root/case).glob('seed_*.log') if (m:=re.fullmatch(r'seed_(\d+)\.log',p.name))] if (root/case).is_dir() else []
    count=sum(eligible(root/case/f'seed_{i:03d}',case) for i in ids)
    return count,max(ids+logs,default=-1)+1


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);p.add_argument('--cases',nargs='+',default=['AM','AFM','FM'],choices=['AM','AFM','FM']);p.add_argument('--target',type=int,default=16);p.add_argument('--workers',type=int,default=16);p.add_argument('--max-new-attempts',type=int,default=128,help='maximum new starts per case in this invocation');a=p.parse_args()
    if min(a.target,a.workers,a.max_new_attempts)<1 or len(set(a.cases))!=len(a.cases):p.error('invalid settings')
    root=a.output.resolve();manifest_path=root/'manifest.json';screen=root/'focus_screen.json';am_seed=root/'shared_seed_AM'
    if not manifest_path.is_file() or not screen.is_file():p.error('existing batch manifest and focus_screen.json required; run the initial batch first')
    manifest=json.loads(manifest_path.read_text())
    if any(c not in manifest['cases'] for c in a.cases):p.error('requested case absent from original batch')
    if 'AM' in a.cases and not (am_seed/'training.json').exists():p.error('AM shared training seed missing')
    configs=['max_nfev','joint_nfev','start_sigma','shape_scale'];settings={k:manifest[k] for k in configs}
    if manifest.get('scope')!='fixed rest, thresholds, fold; lower stable; middle and upper foci; physical calcium; period/amplitudes/aligned waveform':p.error('batch scope differs from lower-stable protocol')
    env=os.environ.copy()
    for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[k]='1'
    counts={};next_id={};launched=Counter();running=Counter();children={};lock=threading.Lock();stop=threading.Event()
    for case in a.cases:counts[case],next_id[case]=scan(root,case)
    print('Existing valid:',counts,'target:',a.target,'maximum new starts per case:',a.max_new_attempts,flush=True)
    def run(case,seed):
        dest=root/case/f'seed_{seed:03d}';dest.parent.mkdir(parents=True,exist_ok=True);log=dest.parent/f'seed_{seed:03d}.log'
        cmd=[sys.executable,'-u',str(ROOT/'scripts/run_focus_fit_one.py'),'--case',case,'--seed',str(seed),'--screen',str(screen),'--am-seed-dir',str(am_seed),'--output',str(dest),'--max-nfev',str(settings['max_nfev']),'--joint-nfev',str(settings['joint_nfev']),'--start-sigma',str(settings['start_sigma']),'--shape-scale',str(settings['shape_scale'])]
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
                        if counts[case]+running[case]<a.target and launched[case]<a.max_new_attempts:
                            choice=case;cursor=(k+1)%len(a.cases);break
                    if choice is None:break
                    seed=next_id[choice];next_id[choice]+=1;launched[choice]+=1;running[choice]+=1
                    future=pool.submit(run,choice,seed);pending[future]=(choice,seed)
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
    if any(counts[c]<a.target for c in a.cases):raise SystemExit('Target not reached; increase --max-new-attempts or inspect failures')

if __name__=='__main__':main()
