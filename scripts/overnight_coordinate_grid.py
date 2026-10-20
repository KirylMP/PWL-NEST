"""Resumable, bounded C x h search. No server credentials or remote shell code."""
import argparse,copy,hashlib,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from pwl_nest.models import ROOT
from pwl_nest.strict_pwa import C_MAX,lr_field,equilibrium_h
from pwl_nest.calcium_coordinates import CalciumTransformPWA,NativeCoordinateDynamic,simulate_coordinates
from pwl_nest.protocols import simulate
from pwl_nest.rest_protocols import stable_rest,native_events,event_residual
from fit_calcium_coordinates import training_data
from fit_positive_coordinates import CalibrationResidual,calibration
import fit_positive_coordinates as fitting

C_KINDS=['identity','sqrt','cbrt','sixth_root','log','square','cube','sixth','exp','inverse','inverse_sqrt','hill','bounded_logit','sqrt_tails']
H_KINDS=['identity','logit','log','exp','cube','sixth']
VERSION=1


def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def source_rest(case,baseline):
    c=brentq(lambda c:lr_field([c,equilibrium_h(c,baseline),baseline],case)[0],1e-10,C_MAX,xtol=1e-14)
    return [float(c),float(equilibrium_h(c,baseline)),baseline]


def prepare(out,case):
    directory=out/'cache'/case;directory.mkdir(parents=True,exist_ok=True)
    if (directory/'data.json').exists():return
    data=training_data(case);seed=CalciumTransformPWA.load(ROOT/f'models/positive_coordinates/sqrt_tails/{case}.json')
    fixed=[p for p in calibration(case,data)[0] if p['name'].startswith('fixed')]
    baseline=.3 if case=='AM' else .45;rest=source_rest(case,baseline);pulse=.35 if case=='AM' else .7
    def decay(tau):return dict(name=f'pulse_tau_{tau}',duration=200.,tau=tau,baseline=baseline,jumps=[[20.,pulse]])
    train=[decay(20.),decay(100.),dict(name='periodic',duration=220.,tau=20.,baseline=baseline,
        jumps=[[t,.16 if case=='AM' else .3] for t in [20.,40.,60.,80.,100.]])]
    rng=np.random.default_rng(20261022)
    test=[decay(tau) for tau in [7.,50.,65.]]+[dict(name='irregular',duration=240.,tau=12.,baseline=baseline,
        jumps=[[float(t),.16 if case=='AM' else .3] for t in 20.+np.r_[0.,np.cumsum(rng.uniform(10,19,6))]])]
    refs={};points=[data[2]]
    for p in fixed:
        trace=data[1].get(p['name'])
        if trace is None:trace=simulate(case,**{k:v for k,v in p.items() if k!='name'},dt=.2)
        refs[p['name']]=trace
    for split,protocols in [('train',train),('test',test)]:
        for p in protocols:
            trace=simulate(case,rest,**{k:v for k,v in p.items() if k!='name'},dt=.25 if split=='train' else .05,
                rtol=2e-11,atol=2e-13,max_step=.05)
            if np.max(abs(trace[1][trace[0]<20]-rest))>1e-8:raise AssertionError('LR does not remain at prepared rest')
            refs[p['name']]=trace;points.append(trace[1][::max(1,len(trace[0])//150)])
    for name,(t,x,_) in refs.items():np.savez_compressed(directory/f'{name}.npz',t=t,x=x)
    np.savez_compressed(directory/'field.npz',points=np.vstack(points))
    atomic(directory/'data.json',dict(case=case,fixed=fixed,train=train,test=test,rest=rest,
        C_lower=data[4],C_upper=data[5],pre_input_delay_s=20.,source_initial_conditions_optimized=False))


class RestObjective:
    peak_weight=25.;explicit_status=True
    def __init__(self,out,case,seed):
        directory=out/'cache'/case;self.spec=json.loads((directory/'data.json').read_text());self.refs={}
        for p in self.spec['fixed']+self.spec['train']+self.spec['test']:
            with np.load(directory/f'{p["name"]}.npz') as data:self.refs[p['name']]=(data['t'],data['x'],{})
        self.static=CalibrationResidual(case,self.spec['fixed'],self.refs,seed)
        self.dynamic=NativeCoordinateDynamic();self.case=case
        self.lengths={p['name']:len(event_residual(*self.refs[p['name']][:2],*self.refs[p['name']][:2])) for p in self.spec['train']}
    def paths(self,model,protocols,precise=False):
        rest,count=stable_rest(model,self.spec['rest'][2],self.spec['rest']);paths={}
        atol=np.array([min(2e-13,float(model.coordinate.derivative(.005))*2e-13),
            min(2e-13,float(model.gate.derivative(.03))*2e-13),2e-13])
        for p in protocols:
            kwargs={k:v for k,v in p.items() if k!='name'}
            if precise:paths[p['name']]=simulate_coordinates(model,rest,**kwargs,dt=.25 if p in self.spec['train'] else .05,
                rtol=2e-11,atol=atol,max_step=.05)
            else:paths[p['name']]=native_events(model,rest,**kwargs,native=self.dynamic)
        return rest,count,paths
    def __call__(self,model,precise=False,parts=False):
        groups={}
        if precise:
            replay={}
            for p in self.spec['fixed']:
                replay[p['name']]=simulate_coordinates(model,**{k:v for k,v in p.items() if k!='name'},dt=.2,
                    rtol=2e-11,atol=np.array([min(2e-13,float(model.coordinate.derivative(.005))*2e-13),2e-14,2e-13]),max_step=.25)
            r=self.static(model,replay=replay)
        else:r=self.static(model)
        groups['fixed_and_spectral_anchors']=float(r@r)
        try:
            rest,count,paths=self.paths(model,self.spec['train'],precise)
            anchor=np.r_[(rest[:2]-np.array(self.spec['rest'])[:2])/[.01,.05],10*(count-1)]
            for p in self.spec['train']:
                t,x,_=paths[p['name']];tr,ref,_=self.refs[p['name']]
                part=event_residual(t,x,tr,ref);r=np.r_[r,part];groups[p['name']]=float(part@part)
        except (ValueError,RuntimeError,FloatingPointError,OverflowError,np.linalg.LinAlgError):
            if precise:raise
            anchor=np.full(3,100.)
            for p in self.spec['train']:
                part=np.full(self.lengths[p['name']],100.);r=np.r_[r,part];groups[p['name']]=float(part@part)
        r=np.r_[r,anchor];groups['own_rest']=float(anchor@anchor)
        return (r,groups) if parts else r


def trial(args):
    out=Path(args.output).resolve();directory=Path(args.trial_dir);directory.mkdir(parents=True,exist_ok=True)
    spec=json.loads((out/'cache'/args.case/'data.json').read_text())
    seed=CalciumTransformPWA.load(ROOT/f'models/positive_coordinates/sqrt_tails/{args.case}.json')
    best_path=out/'best'/args.case/f'{args.c_kind}__{args.h_kind}.json'
    parent_model=None
    if best_path.exists():
        previous_best=json.loads(best_path.read_text());parent_model=previous_best['model']
        seed=CalciumTransformPWA.load(out/parent_model)
    objective=RestObjective(out,args.case,seed)
    with np.load(out/'cache'/args.case/'field.npz') as data:points=data['points']
    data=([],{},points,{},spec['C_lower'],spec['C_upper'])
    fitting.OUT=directory/'fit';fitting.MODELS=directory/'models'
    fitting.OUT.mkdir(parents=True,exist_ok=True);fitting.MODELS.mkdir(exist_ok=True)
    rng=np.random.default_rng(args.seed)
    if args.restart and parent_model is None:
        seed.metadata['parameters']=(np.array(seed.metadata['parameters'])+rng.normal(0,.04,len(seed.metadata['parameters']))).tolist()
    model=fitting.fit(args.case,'cell',seed,data,objective,args.budget,coordinate_kind=args.c_kind,gate_kind=args.h_kind,
        rng_seed=args.seed,checkpoint_path=directory/'unverified_checkpoint.json')
    model.metadata.update(experiment_variant=f'{args.c_kind}__{args.h_kind}',accepted=False,
        source_seed=f'models/positive_coordinates/sqrt_tails/{args.case}.json',initialization='own stable physical equilibrium; 20 s before IP3 pulse',
        fit_seed=args.seed,fit_restart=args.restart,objective='static regimes + own rest + peak heights/widths/intervals/latency + weak waveform')
    model.metadata['parent_model']=parent_model
    model.save(directory/'candidate.json')
    # Model selection is based only on TRAIN, independently recomputed by DOP.
    r,parts=objective(model,precise=True,parts=True);loss=float(r@r)
    previous=json.loads(best_path.read_text())['verified_train_loss'] if best_path.exists() else float('inf')
    result=dict(case=args.case,C=args.c_kind,h=args.h_kind,restart=args.restart,seed=args.seed,
        verified_train_loss=loss,train_parts=parts,physical_field_PWA=False,regional_matrices_constant=True,
        regions=model.latent.region_count*(len(model.input_thresholds)+1),active_parameters=model.metadata['active_fit_parameters'],
        model=str((directory/'candidate.json').relative_to(out)),status='verified',test=None)
    initial_rest,rest_count=stable_rest(model,objective.spec['rest'][2],objective.spec['rest'])
    result.update(PWA_rest=initial_rest.tolist(),LR_rest=objective.spec['rest'],stable_rest_count=rest_count)
    if loss<previous:
        from check_rest_and_phase import metrics
        tests=[]
        for p in objective.spec['test']:
            try:
                _,_,paths=objective.paths(model,[p],precise=True)
                t,x,meta=paths[p['name']];tr,truth,_=objective.refs[p['name']]
                np.testing.assert_allclose(t,tr,atol=1e-10,rtol=0)
                tests.append(dict(name=p['name'],completed=True,**metrics(t,truth,x,20.)))
                np.savez_compressed(directory/f'test_{p["name"]}.npz',t=t,lr=truth,candidate=x)
            except (ValueError,RuntimeError,FloatingPointError,OverflowError,np.linalg.LinAlgError) as exc:
                tests.append(dict(name=p['name'],completed=False,error=str(exc)))
        result['test']=tests
        # This status is observable on the output grid, not a global invariance proof.
        complete=all(r['completed'] for r in tests)
        result['test_C_RMSE']=float(np.mean([r['raw_full_RMSE'][0] for r in tests])) if complete else None
        result['test_phase_aligned_C_RMSE']=float(np.mean([r['global_shift']['aligned_eval_rmse'] for r in tests])) if complete else None
        result['test_all_physical']=all(r.get('physical',False) for r in tests)
        result['test_peak_count_matches']=sum(r.get('peaks',{}).get('counts',[0,-1])[0]==r.get('peaks',{}).get('counts',[0,-1])[1] for r in tests)
        atomic(best_path,result)
    atomic(directory/'result.json',result)


def code_hash():
    paths=sorted((ROOT/'src/pwl_nest').glob('*.py'))+sorted((ROOT/'native').glob('*.cpp'))
    paths += [ROOT/'scripts'/p for p in ['overnight_coordinate_grid.py','fit_positive_coordinates.py',
        'fit_calcium_coordinates.py','fit_strict.py','check_rest_and_phase.py']]
    return hashlib.sha256(b''.join(p.read_bytes() for p in paths)).hexdigest()


def summarize(out):
    config=json.loads((out/'config.json').read_text());rows=[]
    text=['# Coordinate grid: results available so far','',
        'Cells select models by independently verified TRAIN loss only. Test errors are diagnostic; failed/unrun cells stay empty. Different numbers of completed restarts must be reported with any table.','']
    for case in config['cases']:
        text += [f'## {case}: held-out C RMSE (uM)','', '| C / h | '+' | '.join(config['h_kinds'])+' |',
            '|---|'+'---:|'*len(config['h_kinds'])]
        for c in config['c_kinds']:
            cells=[]
            for h in config['h_kinds']:
                path=out/'best'/case/f'{c}__{h}.json'
                r=json.loads(path.read_text()) if path.exists() else None
                cells.append(f'{r["test_C_RMSE"]:.5g}' if r and r.get('test_C_RMSE') is not None else 'FAIL' if r else '—')
                if r:
                    folder=out/'trials'/case/f'{c}__{h}'
                    r['verified_restarts']=len(list(folder.glob('r*/result.json')))
                    rows.append(r)
            text.append('| '+c+' | '+' | '.join(cells)+' |')
        text.append('')
    text+=['Own stable rest, 20 s input delay; C/h never clipped. Raw causal and phase-aligned errors are stored separately. No full acceptance, speed benchmark or NEST claim.','',
        'See best/ for fitted models and metrics; trials/ for every restart including failures; status.json for process state.']
    (out/'TABLE.md').write_text('\n'.join(text)+'\n');atomic(out/'summary.json',rows)
    import csv
    with (out/'table.csv').open('w',newline='') as f:
        fields=['case','C','h','verified_restarts','restart','verified_train_loss','test_C_RMSE','test_phase_aligned_C_RMSE','test_peak_count_matches','test_all_physical','regions','active_parameters','model']
        writer=csv.DictWriter(f,fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)


def master(args):
    import fcntl,platform
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    lock=(out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config=dict(version=VERSION,code_sha256=code_hash(),cases=args.cases,c_kinds=args.c_kinds,h_kinds=args.h_kinds,budget=args.budget,
        libraries=dict(python=f'{sys.version_info.major}.{sys.version_info.minor}',numpy=np.__version__,scipy=__import__('scipy').__version__),
        seed=args.seed,train_selection='independent DOP; no held-out selection',pre_input_delay_s=20.,
        source_models={c:hashlib.sha256((ROOT/f'models/positive_coordinates/sqrt_tails/{c}.json').read_bytes()).hexdigest() for c in args.cases})
    if (out/'config.json').exists():
        if json.loads((out/'config.json').read_text())!=config:raise RuntimeError('Resume configuration/code changed; use a new output directory')
    else:atomic(out/'config.json',config)
    atomic(out/'runtime.json',dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,
        scipy=__import__('scipy').__version__,workers=args.workers,master_pid=os.getpid()))
    (out/'master.pid').write_text(str(os.getpid())+'\n')
    stop=False
    def requested(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,requested);signal.signal(signal.SIGINT,requested)
    started=time.monotonic();deadline=started+args.hours*3600
    for case in args.cases:prepare(out,case)
    cells=[(case,c,h) for c in args.c_kinds for h in args.h_kinds for case in args.cases]
    preferred=[('sqrt_tails','cube'),('sqrt','logit'),('sqrt','log'),('bounded_logit','log'),('sqrt_tails','log')]
    cells.sort(key=lambda x:preferred.index((x[1],x[2])) if (x[1],x[2]) in preferred else len(preferred))
    # Resume after every attempted restart, including interrupted/failed trials.
    next_run={cell:0 for cell in cells}
    for cell in cells:
        folder=out/'trials'/cell[0]/f'{cell[1]}__{cell[2]}'
        if folder.exists():next_run[cell]=max([int(p.name[1:])+1 for p in folder.glob('r[0-9]*')]+[0])
    running={};completed=0;manifest=[];last_plot=0.
    def export_plots():
        try:
            result=subprocess.run([sys.executable,str(ROOT/'scripts/plot_coordinate_grid.py'),str(out)],
                capture_output=True,text=True,timeout=60)
            if result.returncode:print('Plot export failed:',result.stderr[-500:],flush=True)
        except subprocess.TimeoutExpired:print('Plot export timed out; numerical results are saved.',flush=True)
    def cleanup():
        for proc,log,_,_ in running.values():
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=3)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            log.close()
    __import__('atexit').register(cleanup)
    while (time.monotonic()<deadline and not stop) or running:
        now=time.monotonic()
        if now>=deadline:stop=True
        eligible=[c for c in cells if c not in running and (args.rounds==0 or next_run[c]<args.rounds)]
        while not stop and eligible and len(running)<args.workers:
            cell=min(eligible,key=lambda c:(next_run[c],cells.index(c)));eligible.remove(cell)
            case,c,h=cell;restart=next_run[cell];next_run[cell]+=1
            directory=out/'trials'/case/f'{c}__{h}'/f'r{restart:04d}';directory.mkdir(parents=True,exist_ok=True)
            seed=args.seed+restart*100000+cells.index(cell)
            command=[sys.executable,str(Path(__file__).resolve()),'--worker','--output',str(out),'--trial-dir',str(directory),
                '--case',case,'--c-kind',c,'--h-kind',h,'--restart',str(restart),'--seed',str(seed),'--budget',str(args.budget),
                '--trial-timeout',str(args.trial_timeout)]
            log=(directory/'worker.log').open('w')
            proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            running[cell]=(proc,log,time.monotonic(),directory)
            atomic(directory/'attempt.json',dict(case=case,C=c,h=h,restart=restart,seed=seed,status='running',pid=proc.pid))
        for cell,(proc,log,begin,directory) in list(running.items()):
            timed_out=time.monotonic()-begin>args.trial_timeout or (stop and time.monotonic()>=deadline)
            if timed_out and proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM)
            if proc.poll() is not None or timed_out:
                try:proc.wait(timeout=3)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                log.close();completed+=1
                result_path=directory/'result.json'
                status='verified' if result_path.exists() else 'timeout' if timed_out else 'failed'
                record=dict(cell=list(cell),restart=int(directory.name[1:]),status=status,exit_code=proc.returncode,
                    seconds=time.monotonic()-begin,path=str(directory.relative_to(out)))
                atomic(directory/'attempt.json',record);manifest.append(record);del running[cell]
                print(json.dumps(record),flush=True);summarize(out)
                if time.monotonic()-last_plot>300:export_plots();last_plot=time.monotonic()
        atomic(out/'status.json',dict(state='stopping' if stop else 'running',master_pid=os.getpid(),
            elapsed_s=time.monotonic()-started,completed_this_session=completed,
            running=[dict(cell=list(c),pid=v[0].pid) for c,v in running.items()],next_restart={'/'.join(c):n for c,n in next_run.items()}))
        if not running and (stop or (args.rounds and all(n>=args.rounds for n in next_run.values()))):break
        time.sleep(.5)
    summarize(out);export_plots();atomic(out/'status.json',dict(state='stopped' if stop else 'completed',master_pid=os.getpid(),
        elapsed_s=time.monotonic()-started,completed_this_session=completed))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--hours',type=float,default=8.);parser.add_argument('--workers',type=int,default=min(4,max(1,(os.cpu_count() or 2)//2)))
    parser.add_argument('--budget',type=int,default=1500);parser.add_argument('--trial-timeout',type=float,default=900.)
    parser.add_argument('--rounds',type=int,default=0);parser.add_argument('--cases',nargs='+',default=['AM','FM'],choices=['AM','FM'])
    parser.add_argument('--c-kinds',nargs='+',default=C_KINDS,choices=C_KINDS);parser.add_argument('--h-kinds',nargs='+',default=H_KINDS,choices=H_KINDS)
    parser.add_argument('--seed',type=int,default=20261022);parser.add_argument('--worker',action='store_true')
    parser.add_argument('--case',choices=['AM','FM']);parser.add_argument('--c-kind',choices=C_KINDS);parser.add_argument('--h-kind',choices=H_KINDS)
    parser.add_argument('--trial-dir');parser.add_argument('--restart',type=int,default=0);parser.add_argument('--summarize',action='store_true')
    args=parser.parse_args()
    if args.hours<=0 or args.workers<1 or args.budget<1 or args.trial_timeout<=0 or args.rounds<0:parser.error('Invalid run limits')
    if args.summarize:summarize(Path(args.output).resolve())
    elif args.worker:
        signal.signal(signal.SIGALRM,signal.SIG_DFL);signal.alarm(max(1,int(args.trial_timeout)))
        trial(args)
    else:master(args)
