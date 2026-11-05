"""Three-focus necessary-condition screening under fixed equilibrium anchors.

Does not impose the fold or prove cycle existence. Use before a costly fit.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from anchored_lower_fold import FREE

ROOT=Path(__file__).resolve().parents[1]


def anchored(base,x):
    n=np.array(base['alpha'],float);h=np.array(base['beta'],float)
    scales=np.array([.05*max(abs(base[k][j]),.3) for k,j in FREE])
    for (key,j),v,scale in zip(FREE,x,scales):
        (n if key=='alpha' else h)[j]+=v*scale
    cuts=np.array(base['physical_thresholds_C']);C0,q0=base['rest_target'];I1,I2=base['fixed_equilibrium_threshold_I']
    def v(c,I):return np.r_[1.,c,I,np.maximum(c-cuts,0.)]
    u0,u1,u2=v(C0,0.),v(cuts[0],I1),v(cuts[1],I2)
    def eq():return np.array([n@u0-q0,h@u0-q0,(n-h)@u1,(n-h)@u2])
    mat=np.array([[1,0,u0[2],u0[3]],[0,1,0,0],[1,-1,u1[2],u1[3]],[1,-1,u2[2],u2[3]]])
    delta=np.linalg.solve(mat,-eq());n[0]+=delta[0];h[0]+=delta[1];n[2]+=delta[2];n[3]+=delta[3]
    a,b=base['a'],base['b'];T=[];D=[]
    for j in range(3):
        ns=n[1]+sum(n[3:3+j]);hs=h[1]+sum(h[3:3+j]);T.append(-a*ns-b);D.append(a*b*(ns-hs))
    return n,h,np.array(T),np.array(D),eq()


def constraints(base,x,margin):
    n,h,T,D,_=anchored(base,x)
    # Focus plus sign trace and increasing equilibrium; strict margins are numerical choices.
    return np.r_[-T[0]-margin,T[1]-margin,-T[2]-margin,4*D-T*T-margin, h[2]-n[2]-margin]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cases',nargs='+',default=['AM','AFM','FM']);p.add_argument('--bounds',type=float,default=40.);p.add_argument('--starts',type=int,default=6);p.add_argument('--seed',type=int,default=20261105);p.add_argument('--margin',type=float,default=0.01);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.bounds<=0 or a.starts<1 or a.margin<=0:p.error('bounds and starts must be positive')
    rng=np.random.default_rng(a.seed);out={}
    for case in a.cases:
        path=ROOT/'paper/data/physical_models/fixed_fold'/case/'parameters.json';base=json.loads(path.read_text());attempts=[]
        for k in range(a.starts):
            x=np.zeros(len(FREE)) if k==0 else rng.uniform(-min(a.bounds,8),min(a.bounds,8),len(FREE))
            res=minimize(lambda x:float(x@x),x,method='SLSQP',bounds=[(-a.bounds,a.bounds)]*len(FREE),
                         constraints=[{'type':'ineq','fun':lambda x:constraints(base,x,a.margin)}],
                         options={'maxiter':2000,'ftol':1e-11})
            n,h,T,D,err=anchored(base,res.x);cons=constraints(base,res.x,a.margin);feasible=bool(np.min(cons)>-1e-7 and np.max(abs(err))<1e-9)
            attempts.append({'start':k,'solver_success':bool(res.success),'message':str(res.message),'feasible':feasible,
                'x':res.x.tolist(),'objective':float(res.fun),'trace':T.tolist(),'determinant':D.tolist(),
                'discriminant':(T*T-4*D).tolist(),'anchor_max_abs':float(np.max(abs(err))),'min_constraint':float(np.min(cons)),
                'alpha':n.tolist(),'beta':h.tolist()})
        good=[z for z in attempts if z['feasible']];best=min(good,key=lambda z:z['objective']) if good else None
        out[case]={'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'best':best,'attempts':attempts}
        print(case,'feasible',len(good),'/',len(attempts),'best',None if best is None else round(best['objective'],4),flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'scope':'necessary screening only: 3 foci and fixed rest/threshold equilibria; NO fold or cycle conditions','settings':vars(a)|{'output':str(a.output)},'results':out},indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
