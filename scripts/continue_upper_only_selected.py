"""Continue cycles crossing only the upper threshold after lower grazing.

Uses sampled segment checks and independently integrated affine segments.
No claim is made that all routes/branches have been found.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import least_squares, minimize_scalar
from physical_fold_routes import Route
from selected_lr_branches import load_model

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/selected_lr_branches'
SCALE=np.array([10.,10.,.1])


def mid_minimum(orb,t,I):
    _,z,q,mats=orb.match(t,I,True);y=np.array([z[1],q[1],1.]);A=mats[1]
    def C(s):return float((expm(A*s)@y)[0])
    grid=np.linspace(0,t[1],101);idx=int(np.argmin([C(x) for x in grid]));a=grid[max(idx-1,0)];b=grid[min(idx+1,len(grid)-1)]
    val=min(C(0),C(t[1]))
    if b>a:
        result=minimize_scalar(C,bounds=(a,b),method='bounded',options={'xatol':1e-11})
        val=min(val,float(result.fun))
    return val


def grazing(orb,seed):
    def f(w):
        t=w[:2]*SCALE[:2];I=w[2]*SCALE[2]
        try:return np.r_[orb.match(t,I),mid_minimum(orb,t,I)-orb.cut[0]]
        except (ValueError,OverflowError,FloatingPointError):return np.full(3,1e4)
    lower=np.r_[np.full(2,1e-6)/SCALE[:2],.001/SCALE[2]];upper=np.r_[np.full(2,120.)/SCALE[:2],1.3/SCALE[2]]
    sol=least_squares(f,np.array(seed)/SCALE,bounds=(lower,upper),max_nfev=300,
                      xtol=1e-11,ftol=1e-11,gtol=1e-11)
    if max(abs(f(sol.x)))>1e-7:raise ValueError('grazing solve unresolved')
    return sol.x*SCALE


def inspect(orb,t,I):
    r,z,q,mats=orb.match(t,I,True);speeds=[];values=[];segments=[]
    for j,(region,A) in enumerate(zip(orb.regions,mats)):
        y=np.array([z[j],q[j],1.]);speed=float((A@y)[0]);speeds.append(speed)
        if (1 if j==0 else -1)*speed<1e-8:raise ValueError('upper section grazing')
        states=np.array([expm(A*s)@y for s in np.linspace(0,t[j],201)]);lo=orb.cut[region-1];hi=np.inf if region==2 else orb.cut[region]
        if np.any(states[:,0]<lo-1e-8) or np.any(states[:,0]>hi+1e-8):raise ValueError('sampled upper itinerary violation')
        values.extend(states[:,0]);sol=solve_ivp(lambda _,v:A@v,(0,t[j]),y,method='DOP853',rtol=1e-11,atol=1e-13,max_step=.05)
        segments.append(float(np.max(abs(sol.y[:,-1]-states[-1]))))
        if not sol.success or segments[-1]>1e-7:raise ValueError('independent segment unresolved')
    initial=np.array([z[0],q[0],0.]);logm=float(orb.T[list(orb.regions)]@t);direction=-1 if logm>0 else 1
    def rhs(_,y):
        C,Q=y[:2];v=np.r_[1,C,I,np.maximum(C-orb.cut,0)];j=np.searchsorted(orb.cut,C,side='right')
        return [orb.a*(Q-orb.N@v),orb.b*(orb.H@v-Q),orb.T[j]]
    sol=solve_ivp(rhs,(0,direction*sum(t)),initial,method='DOP853',rtol=5e-10,atol=5e-12,max_step=.04)
    closure=float(np.max(abs(sol.y[:2,-1]-initial[:2])))
    if not sol.success or closure>1e-5:raise ValueError('independent full cycle unresolved')
    return {'I':float(I),'times':t.tolist(),'period':float(sum(t)),'Cmin':float(min(values)),
            'Cmax':float(max(values)),'stable':bool(logm<0),'logM':logm,
            'matching':float(max(abs(r))),'closure':closure,'independent_segment_error':max(segments),
            'crossing_speeds':speeds,'sampled_positive_C':bool(min(values)>0)}


def continuation(orb,seed,sign,steps):
    def fun(w):
        try:return orb.match(w[:2]*SCALE[:2],w[2]*SCALE[2])
        except (ValueError,OverflowError,FloatingPointError):return np.full(2,1e5)
    def jac(w):
        eps=1e-5;return np.column_stack([(fun(w+eps*e)-fun(w-eps*e))/(2*eps) for e in np.eye(3)])
    w=seed/SCALE;tangent=np.linalg.svd(jac(w),full_matrices=True)[2][-1]*sign;points=[];step=.002;reason='step limit'
    lower=np.r_[np.full(2,1e-7)/SCALE[:2],0.];upper=np.r_[np.full(2,120.)/SCALE[:2],1.3/SCALE[2]]
    for _ in range(steps):
        v=w+step*tangent
        if np.any(v<=lower) or np.any(v>=upper):reason='bounds';break
        def objective(x):return np.r_[fun(x),np.dot(x-v,tangent)]
        sol=least_squares(objective,v,bounds=(lower,upper),max_nfev=90,ftol=1e-10,xtol=1e-10,gtol=1e-10)
        if max(abs(objective(sol.x)))>1e-8:
            step*=.5
            if step<2e-5:reason='corrector unresolved';break
            continue
        new=sol.x;z=new*SCALE
        try:points.append(inspect(orb,z[:2],z[2]))
        except ValueError as exc:reason=str(exc);break
        tan=np.linalg.svd(jac(new),full_matrices=True)[2][-1]
        if tan@tangent<0:tan=-tan
        w,tangent=new,tan;step=min(step*1.05,.012)
    return {'direction':sign,'points':points,'reason':reason}


def task(t):
    case,steps=t;model,manifest=load_model(case);report=json.loads((ROOT/manifest['report_path']).read_text());f=report['fold'];t=f['times'];I=f['I'];orb=Route(model,'upper')
    try:
        seed=grazing(orb,[t[1],t[0]+t[2],I]);branches=[continuation(orb,seed,s,steps) for s in (-1,1)]
        return {'case':case,'candidate_sha256':manifest['candidate_sha256'],'grazing':{'I':float(seed[2]),'times':seed[:2].tolist(),'Cmin':mid_minimum(orb,seed[:2],seed[2])},'branches':branches}
    except (ValueError,OverflowError,FloatingPointError) as exc:return {'case':case,'candidate_sha256':manifest['candidate_sha256'],'error':str(exc),'branches':[]}


def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=160);p.add_argument('--workers',type=int,default=3)
    p.add_argument('--cases',nargs='+',choices=('AM','AFM','FM'),default=('AM','AFM','FM'))
    p.add_argument('--output',type=Path,default=OUT/'upper_only.json');a=p.parse_args();out=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(task,(c,a.steps)) for c in a.cases]):
            r=f.result();out.append(r);print(r['case'],r.get('error'),[(len(b['points']),b['reason']) for b in r['branches']],flush=True)
            a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'scope':'finite continuation of two-segment itinerary; no branch completeness','results':out},indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
