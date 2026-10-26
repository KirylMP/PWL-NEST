"""Wall-clock progress only; never evaluates the model or changes solver state."""
import datetime
import json
import os
import sys
import threading
import time


def duration(seconds):
    if seconds is None:return 'warming-up'
    seconds=max(0.,seconds)
    if seconds<60:return f'{seconds:.0f}s'
    if seconds<3600:return f'{seconds/60:.1f}m'
    return f'{seconds/3600:.1f}h'


def optimizer_plan(cfg,epsilon_schedule,max_nfev):
    first=max(1,max_nfev//2)
    plans=[dict(name='precycle_fixed_transform',kind='geometry',budget=first,units=1)]
    if max_nfev>1:plans.append(dict(name='precycle_joint',kind='geometry',budget=max_nfev-first,units=1))
    budget=cfg['max_nfev_per_epsilon'];recovery=cfg.get('rescue_evaluations',6)
    for index,epsilon in enumerate(epsilon_schedule):
        coarse=index<min(2,len(epsilon_schedule)-1)
        levels=cfg.get('coarse_I_nodes',cfg['I_nodes']) if coarse else cfg['I_nodes']
        plans.append(dict(name=f'cycles_{epsilon}',kind='cycles',budget=budget,
            recovery=recovery if budget>3 else 0,units=len(levels)*cfg['max_stage_nodes']))
    budget=cfg.get('common_polish_nfev',budget)
    plans.append(dict(name='common_grid_joint_polish',kind='cycles',budget=budget,
        recovery=recovery if budget>3 else 0,units=len(cfg['I_nodes'])*cfg['selection_nodes']))
    return plans


class TrialProgress:
    """One reporter per worker, created after multiprocessing dispatch.

    Budget ETAs extrapolate wall time and grid size; they exclude final audits,
    analysis and validation. Unknown future phase costs remain unknown.
    """
    def __init__(self,log_dir,label,interval=30.):
        self.log_dir=log_dir;self.label=label;self.interval=interval
        self.lock=threading.RLock();self.stop=threading.Event()
        self.started=time.perf_counter();self.stage_started=self.started
        self.plans=[];self.index=None;self.rates={};self.work=None
        self.state=dict(trial=label,pid=os.getpid(),status='running',phase='starting',evaluation=0)
        self.last_emit=0.;self.last_evaluation=self.started

    def __enter__(self):
        if self.log_dir is not None:self.log_dir.mkdir(parents=True,exist_ok=True)
        self.emit('started')
        self.thread=threading.Thread(target=self._heartbeat,daemon=True)
        self.thread.start()
        return self

    def __exit__(self,kind,error,traceback):
        self.stop.set();self.thread.join()
        with self.lock:
            self.state['status']='failed' if kind else 'finished'
            if error is not None:self.state['error']=repr(error)
            self.emit(self.state['status'])

    def _heartbeat(self):
        while not self.stop.wait(self.interval):self.emit('heartbeat')

    def configure(self,plans):
        with self.lock:self.plans=[dict(p) for p in plans]

    def stage(self,index,units=None):
        with self.lock:
            self.index=index;self.stage_started=time.perf_counter();self.last_evaluation=self.stage_started
            self.work=None
            if units is not None:self.plans[index]['units']=units
            p=self.plans[index]
            for key in ['last_evaluation_wall_seconds','last_objective_seconds']:self.state.pop(key,None)
            self.state.pop('admissible',None)
            self.state.update(phase=p['name'],stage=index+1,stages=len(self.plans),evaluation=0,
                main_budget=p['budget'],recovery_budget=p.get('recovery',0),activity='stage_setup',details={},loss=None,
                stage_finished=False)
            self.emit('stage_started')

    def units(self,value):
        with self.lock:
            if self.index is not None:self.plans[self.index]['units']=value

    def phase(self,name):
        with self.lock:
            self.index=None;self.stage_started=time.perf_counter();self.work=None
            for key in ['stage','stages','main_budget','recovery_budget','last_evaluation_wall_seconds','last_objective_seconds']:
                self.state.pop(key,None)
            self.state.pop('admissible',None);self.state.pop('stage_finished',None)
            self.state.update(phase=name,evaluation=0,activity=name,details={},loss=None)
            self.emit('phase_started')

    def activity(self,name,**details):
        with self.lock:
            self.work=None;self.state['activity']=name;self.state['details']=details

    def start_work(self,name,total):
        with self.lock:
            self.work=dict(name=name,total=int(total),completed=0,started=time.perf_counter())
            self.state['activity']=name;self.state['details']={}

    def advance(self,completed,**details):
        with self.lock:
            if self.work is not None:self.work['completed']=int(completed)
            self.state['details']=details

    def evaluation(self,number,loss,admissible,seconds):
        with self.lock:
            now=time.perf_counter()
            self.state.update(evaluation=number,loss=float(loss),admissible=bool(admissible),
                last_objective_seconds=seconds,last_evaluation_wall_seconds=now-self.last_evaluation)
            self.last_evaluation=now
            if number<=2 or now-self.last_emit>=self.interval:self.emit('evaluation')

    def stage_done(self):
        with self.lock:
            if self.index is not None and self.state['evaluation']:
                p=self.plans[self.index]
                self.rates[p['kind']]=(time.perf_counter()-self.stage_started)/(self.state['evaluation']*p['units'])
            self.state['stage_finished']=True
            self.emit('stage_finished')

    def snapshot(self):
        with self.lock:
            now=time.perf_counter();data=dict(self.state)
            data.update(timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                trial_elapsed_seconds=now-self.started,phase_elapsed_seconds=now-self.stage_started,
                eta_scope='optimizer budget only; excludes final audits, analysis and validation',
                eta_assumptions='current wall throughput and linear I*u scaling; no convergence-time prediction')
            work_rate=None
            if self.work is not None:
                w=self.work;elapsed=now-w['started'];done=w['completed']
                data['work']=dict(name=w['name'],completed=done,total=w['total'],elapsed_seconds=elapsed,
                    remaining_seconds=elapsed/done*(w['total']-done) if done else None)
                if done and w['name']=='return_maps':work_rate=elapsed/done*w['total']
            data.update(stage_budget_remaining_seconds=None,optimizer_budget_remaining_seconds=None)
            if self.index is not None:
                p=self.plans[self.index];done=self.state['evaluation'];rates=dict(self.rates)
                if done:
                    per_eval=(now-self.stage_started)/done
                    # A continuation stage can reuse the preceding map at its
                    # first evaluation; do not extrapolate that cache hit alone.
                    if done<2 and p['kind']=='cycles':per_eval=max(per_eval,rates.get('cycles',0)*p['units'])
                else:per_eval=work_rate or (rates.get(p['kind'],0)*p['units']) or None
                if per_eval is not None:
                    rates[p['kind']]=per_eval/p['units']
                    remaining=0. if self.state.get('stage_finished') else max(0,p['budget']+p.get('recovery',0)-done)*per_eval
                    data['stage_budget_remaining_seconds']=remaining
                    future=self.plans[self.index+1:]
                    missing=sorted(set(q['kind'] for q in future if q['kind'] not in rates))
                    data['unmeasured_phase_kinds']=missing
                    if not missing:
                        data['optimizer_budget_remaining_seconds']=remaining+sum(
                            (q['budget']+q.get('recovery',0))*q['units']*rates[q['kind']] for q in future)
            return data

    def emit(self,event):
        with self.lock:
            data=self.snapshot();data['event']=event;self.last_emit=time.perf_counter()
            text=(f"[{data['timestamp_utc']}] {self.label} {event} phase={data['phase']} "
                  f"elapsed={duration(data['trial_elapsed_seconds'])}")
            if 'stage' in data:
                text+=(f" stage={data['stage']}/{data['stages']} eval={data['evaluation']}/{data['main_budget']}"
                       f"(+{data['recovery_budget']} recovery)")
            if data.get('loss') is not None:text+=f" loss={data['loss']:.7g} admissible={data['admissible']}"
            if 'last_evaluation_wall_seconds' in data:text+=f" last_eval_wall={duration(data['last_evaluation_wall_seconds'])}"
            text+=f" activity={data['activity']}"
            if data.get('details'):text+=' '+','.join(f'{k}={v}' for k,v in data['details'].items())
            if 'work' in data:
                w=data['work'];text+=f" work={w['completed']}/{w['total']} work_ETA~{duration(w['remaining_seconds'])}"
            if self.index is not None:
                text+=(' stage_budget_ETA~'+duration(data['stage_budget_remaining_seconds'])+
                       ' optimizer_budget_ETA~'+duration(data['optimizer_budget_remaining_seconds']))
            if self.log_dir is not None:
                with (self.log_dir/f'{self.label}.log').open('a') as stream:stream.write(text+'\n')
                path=self.log_dir/f'{self.label}.progress.json';temporary=path.with_suffix('.tmp')
                temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temporary.replace(path)
            sys.stdout.write(text+'\n');sys.stdout.flush()
