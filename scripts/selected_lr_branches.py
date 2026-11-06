"""Numerically inspect the selected physical-coordinate LR reductions.

Continuation uses a both-threshold shooting itinerary and finite sampled
first-crossing checks. Forward simulations probe attracting cycles only.
Neither is a completeness proof. Existing manuscript figures are untouched.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from continue_physical_fixed import Orbit, continue_branch, nondegeneracy

ROOT=Path(__file__).resolve().parents[1]
MODELS=ROOT/'paper/data/selected_lr_candidates'
OUT=ROOT/'results/selected_lr_branches'
CASES=('AM','AFM','FM')
WINDOW={'AM':(.29,.70),'AFM':(.27,.98),'FM':(.46,1.20)}
INITIAL={'AM':.35,'AFM':.7,'FM':1.0}


def load_model(case):
    manifest=json.loads((MODELS/'manifest.json').read_text())['models'][case]
    path=ROOT/manifest['candidate_path'];raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=manifest['candidate_sha256']:raise ValueError(f'{case} hash mismatch')
    return json.loads(raw),manifest


def branch_task(task):
    case,sign,steps=task
    model,manifest=load_model(case);report=json.loads((ROOT/manifest['report_path']).read_text())
    if hashlib.sha256((ROOT/manifest['report_path']).read_bytes()).hexdigest()!=manifest['report_sha256']:
        raise ValueError('report hash mismatch')
    fold=report['fold'];orb=Orbit(model);seed=np.array([*fold['times'],fold['I']])
    try:
        neutral=orb.point(seed[:4],seed[4],1e-10,require_physical=False)
        neutral.pop('physical',None);neutral.pop('stable',None)
        neutral['nondegeneracy']=nondegeneracy(orb,seed)
        branch=continue_branch(orb,seed,sign,steps,require_physical=False,max_I=1.3,
                               initial_step=.006,max_step=.015)
        for point in branch['points']:
            point.pop('physical',None)
            point['sampled_positive_C']=point['Cmin']>0
        return {'case':case,'candidate_sha256':manifest['candidate_sha256'],'neutral':neutral,'branch':branch}
    except Exception as exc:
        return {'case':case,'candidate_sha256':manifest['candidate_sha256'],'error':f'{type(exc).__name__}: {exc}',
                'branch':{'direction':sign,'points':[],'reason':'seed unresolved'}}


def forward_task(task):
    case,I,duration=task;d,_=load_model(case);orb=Orbit(d);C=INITIAL[case]
    q=float(np.dot(orb.N,np.r_[1.,C,I,np.maximum(C-orb.cuts,0.)])); initial=[C,q,0.]
    def field(t,y):return orb.rhs(t,y,I)
    def peak(t,y):return field(t,y)[0]
    peak.direction=-1
    try:
        sol=solve_ivp(field,(0,duration),initial,method='DOP853',rtol=2e-8,atol=2e-10,
                      max_step=.3,events=peak,dense_output=True)
        ts=sol.t_events[0];ts=ts[ts>duration/2]
        if not sol.success:return {'I':I,'valid':False,'reason':'solver_failed'}
        if len(ts)<3:return {'I':I,'valid':False,'reason':'insufficient_peaks'}
        t0,t1=ts[-2:];period=t1-t0;prev=ts[-2]-ts[-3]
        z=sol.sol(np.linspace(t0,t1,801));amp=float(np.ptp(z[0]));delta=float(abs(period-prev))
        if amp<1e-4 or delta>max(.1,.01*period):return {'I':I,'valid':False,'reason':'not_converged_or_tiny'}
        minimum=float(np.min(z[0]));maximum=float(np.max(z[0]));logm=float(z[2,-1]-z[2,0])
        if minimum<=0:return {'I':I,'valid':False,'reason':'negative_C','Cmin':minimum}
        return {'I':I,'valid':True,'Cmin':minimum,'Cmax':maximum,'period':float(period),
                'period_change':delta,'logM':logm,'stable':bool(logm<0),
                'initial_C':C,'initial_q':q,'duration':duration,'scope':'finite forward observation'}
    except (ValueError,OverflowError,FloatingPointError) as exc:
        return {'I':I,'valid':False,'reason':type(exc).__name__}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=6);p.add_argument('--steps',type=int,default=120)
    p.add_argument('--grid',type=int,default=55);p.add_argument('--duration',type=float,default=550.)
    p.add_argument('--skip-forward',action='store_true');a=p.parse_args()
    if min(a.workers,a.steps,a.grid)<1 or a.duration<=0:p.error('invalid settings')
    OUT.mkdir(parents=True,exist_ok=True)
    tasks=[(c,s,a.steps) for c in CASES for s in (-1,1)];branches=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for fut in as_completed([pool.submit(branch_task,t) for t in tasks]):
            r=fut.result();branches.append(r);pnt=r['branch']['points']
            print('branch',r['case'],r['branch']['direction'],'points',len(pnt),
                  'reason',r['branch']['reason'],flush=True)
    branch_data={'scope':'numerical, four-segment both-threshold itinerary; finite sampling, no global completeness',
                 'settings':{'steps':a.steps},'results':branches}
    (OUT/'branches.json').write_text(json.dumps(branch_data,indent=2,allow_nan=False)+'\n')
    if a.skip_forward:return
    plan=[(c,float(I),a.duration) for c in CASES for I in np.linspace(*WINDOW[c],a.grid)]
    scans={c:[] for c in CASES}
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for task,r in zip(plan,pool.map(forward_task,plan,chunksize=1)):
            scans[task[0]].append(r)
            if len(scans[task[0]])%10==0:print('forward',task[0],len(scans[task[0]]),flush=True)
    for c in CASES:
        print('forward',c,'cycles',sum(x['valid'] for x in scans[c]),'/',len(scans[c]),flush=True)
    (OUT/'forward.json').write_text(json.dumps({'scope':'finite forward only, initial state C and q on C-nullcline; failed detection is not a proof of absence',
         'settings':{'duration_s':a.duration,'grid':a.grid,'windows':WINDOW},'results':scans},indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
