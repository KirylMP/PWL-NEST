"""Finite-difference return-map diagnostic near selected neutral cycles.

Root finding follows a given itinerary and validates sampled segment residence.
Nonzero derivative estimates are numerical evidence, not a proof.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.optimize import newton
from continue_physical_fixed import Orbit, REGIONS
from selected_lr_branches import load_model

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/selected_lr_branches/return_map_check.json'


def mapped(orb,times,I,q):
    cuts=orb.cuts
    bounds=(cuts[0],cuts[1],cuts[1],cuts[0],cuts[0])
    state=np.array([cuts[0],q,1.]); phases=[]
    for j,r in enumerate(REGIONS):
        n=orb.N[1]+sum(orb.N[3:3+r]); h=orb.H[1]+sum(orb.H[3:3+r])
        ni=orb.N[0]+orb.N[2]*I-orb.N[3:3+r]@cuts[:r]
        hi=orb.H[0]+orb.H[2]*I-orb.H[3:3+r]@cuts[:r]
        A=np.array([[-orb.a*n,orb.a,-orb.a*ni],[orb.b*h,-orb.b,orb.b*hi],[0.,0.,0.]])
        def f(t):return (expm(A*t)@state)[0]-bounds[j+1]
        def fp(t):return (A@expm(A*t)@state)[0]
        t=float(newton(f,times[j],fprime=fp,tol=1e-11,maxiter=50))
        if t<=1e-10 or not np.isfinite(t) or abs(f(t))>1e-9:
            raise ValueError('unresolved crossing')
        lo=-np.inf if r==0 else cuts[r-1]; hi_bound=np.inf if r==2 else cuts[r]
        C=np.array([(expm(A*x)@state)[0] for x in np.linspace(0,t,201)])
        if np.min(C)<lo-1e-9 or np.max(C)>hi_bound+1e-9:
            raise ValueError('sampled itinerary violation')
        if (1 if j<2 else -1)*fp(0)<=0:raise ValueError('wrong entry direction')
        if (1 if j in (0,3) else -1)*fp(t)<=0:raise ValueError('wrong exit direction')
        state=expm(A*t)@state;phases.append(t)
    return float(state[1]),phases


def main():
    result={}
    for case in ('AM','AFM','FM'):
        model,manifest=load_model(case);orb=Orbit(model)
        report=json.loads((ROOT/manifest['report_path']).read_text());fold=report['fold']
        t=np.array(fold['times']);I=float(fold['I']);_,_,q,_=orb.match(t,I,True);q0=float(q[0])
        base,phases=mapped(orb,t,I,q0); record={'candidate_sha256':manifest['candidate_sha256'],'I':I,'q0':q0,'return_residual':base-q0,'phases':phases,'logM':float(orb.T[list(REGIONS)]@t),'q_steps':[],'I_steps':[]}
        for h in (1e-5,3e-6,1e-6,3e-7,1e-7,3e-8):
            try:
                p,_=mapped(orb,t,I,q0+h);m,_=mapped(orb,t,I,q0-h)
                record['q_steps'].append({'h':h,'P_q':(p-m)/(2*h),'P_qq':(p-2*base+m)/(h*h)})
            except (ValueError,RuntimeError,OverflowError) as exc:record['q_steps'].append({'h':h,'error':str(exc)})
        for h in (1e-5,1e-6,1e-7,1e-8,1e-9):
            try:
                p,_=mapped(orb,t,I+h,q0);m,_=mapped(orb,t,I-h,q0)
                record['I_steps'].append({'h':h,'P_I':(p-m)/(2*h)})
            except (ValueError,RuntimeError,OverflowError) as exc:record['I_steps'].append({'h':h,'error':str(exc)})
        result[case]=record;print(case,json.dumps(record),flush=True)
    OUT.write_text(json.dumps({'scope':'finite differences at given four-segment itinerary; sampled residence checks; not a proof of nondegeneracy', 'results':result},indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
