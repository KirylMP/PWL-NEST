#!/usr/bin/env python3
"""Two-threshold C oscillator with an unrestricted, non-biological latent q.

The gauge q*=0 at I=0 and coefficient of q in Cdot=1 removes an affine
scale/shift redundancy; it is not a constraint on physical h.
"""
import argparse
import ctypes
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares,minimize
from scipy.integrate import solve_ivp
from fit_physical_calcium import PhysicalModel
from fold import BoundaryFold

ROOT=Path(__file__).resolve().parent

class FreeModel(PhysicalModel):
    def __init__(self,case):
        super().__init__(case,'identity')
        self.active=np.array([1,2,3,4,5,6,7]);self.gate='free'
    def constrain_events(self,x):
        x=np.asarray(x,float).copy();x[0]=0.;x[8]=0.;return x
    def unpack(self,x):
        m=super().unpack(x);m['N']=m['N'].copy();m['H']=m['H'].copy()
        # Translate the arbitrary auxiliary coordinate so q(rest)=0.
        m['N'][0]-=self.rest[1];m['H'][0]-=self.rest[1]
        m['co']=m['co'].copy();m['co'][0,0]=-m['N'][0];m['co'][1,0]=m['b']*m['H'][0]
        return m
    def physically_valid_events(self,x):
        return bool(np.all(np.isfinite(self.unpack(x)['co'])))
    def hidden_state_admissible(self,q):return bool(np.all(np.isfinite(q)))
    def equilibrium(self,I,m):
        n=m['N'][1]+np.r_[0,np.cumsum(m['N'][3:])];h=m['H'][1]+np.r_[0,np.cumsum(m['H'][3:])]
        nc=m['N'][0]-np.r_[0,np.cumsum(m['N'][3:]*self.cuts)];hc=m['H'][0]-np.r_[0,np.cumsum(m['H'][3:]*self.cuts)]
        c=(hc-nc+(m['H'][2]-m['N'][2])*I)/(n-h);q=nc+n*c+m['N'][2]*I
        valid=(c>np.r_[-np.inf,self.cuts])&(c<=np.r_[self.cuts,np.inf])&(c>0)&(c<self.CMAX)&np.isfinite(q)
        return [dict(C=float(c[r]),z=[float(c[r]),float(q[r])],region=int(r),stable=bool(m['T'][r]<0 and m['D'][r]>0)) for r in np.flatnonzero(valid)]

def from_identity(model,seed):
    if seed.get('gate_coordinate')!='identity':raise ValueError('Free latent initialization requires an identity-gate seed')
    x=np.asarray(seed['parameters'],float).copy();x[7]*=float(seed['a']);x[0]=0.;x[8]=0.
    return model.constrain_events(x)

def check_fold(model,x,fold,info):
    times=np.asarray(info['crossing_times']);_,m,z,q,_=fold.matching(x,times,True);period=float(sum(times))
    initial=np.array([z[0],q[0],0.])
    def rhs(t,y):
        flow=m['co']@np.r_[1.,y[0],y[1],fold.I,np.maximum(y[0]-m['padded'],0.)]
        region=int(np.searchsorted(m['th'],y[0],side='right'))
        return np.r_[flow,m['T'][region]]
    sol=solve_ivp(rhs,(0,period),initial,method='DOP853',rtol=2e-10,atol=2e-12,max_step=.01)
    if not sol.success:raise ValueError('Independent fixed-period fold integration failed: '+sol.message)
    dc=float(sol.y[0,-1]-initial[0]);dq=float(sol.y[1,-1]-initial[1]);logm=float(sol.y[2,-1])
    if max(abs(dc),abs(dq))>1e-6 or abs(logm)>1e-4:raise ValueError(f'Fold check failed closure_C={dc:.3g} closure_q={dq:.3g} log_multiplier={logm:.3g}')
    return dict(period=period,closure_C=dc,closure_q=dq,log_multiplier=logm,method='Independent DOP853 integration over boundary-matched period; no event-root search')

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('seed',type=Path);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--duration',type=float,default=240.);ap.add_argument('--dt',type=float,default=.04);ap.add_argument('--max-nfev',type=int,default=1000)
    ap.add_argument('--cmax-scale',type=float,default=.08);ap.add_argument('--equilibrium-weight',type=float,default=.02)
    ap.add_argument('--jitter',type=float,default=0.);ap.add_argument('--random-seed',type=int,default=20261029)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if a.duration<60 or a.dt<=0 or a.cmax_scale<=0 or a.equilibrium_weight<0 or a.jitter<0:raise ValueError('Invalid fitting settings')
    seed=json.loads((a.seed/'parameters.json').read_text());model=FreeModel(seed['case'].lower());x0=from_identity(model,seed);m0=model.unpack(x0)
    ref=json.loads((a.seed/'trajectory_comparison.json').read_text());samples=np.load(model.root/'orbit_samples.npz');rows=[]
    for row in ref['rows']:
        q=row['LR'];I=float(row['I']);idx=np.flatnonzero(np.isclose(samples['I'],I,rtol=0,atol=1e-10))
        if len(idx) and q['period'] is not None:
            # Use the observed point nearest the calcium peak. This is still
            # specified by C and Cdot, not by any hidden-state value.
            j=idx[np.argmax(samples['C'][idx])];rows.append((I,float(samples['C'][j]),float(samples['vel'][j,0]),float(q['Cmin']),float(q['Cmax']),float(q['period'])))
    if not rows:raise ValueError('No matching reference inputs')
    anchors=[];lo,hi=(.25,.75) if model.case=='am' else (.4,1.2)
    for I in np.linspace(lo,hi,25):
        for y in model.lr.eqs(float(I)):
            J=model.lr.jac(y,float(I));anchors.append((float(I),float(y[0]),float(np.trace(J)),float(np.linalg.det(J))))
    libpath=a.out.resolve()/'free_latent_engine.so';subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(ROOT/'free_latent_engine.cpp'),'-o',str(libpath)],check=True)
    lib=ctypes.CDLL(str(libpath));ptr=ctypes.POINTER(ctypes.c_double);lib.simulate.argtypes=[ptr,ptr,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_int,ptr];lib.simulate.restype=ctypes.c_int
    steps=round(a.duration/a.dt);calls=0;best=np.inf;start=time.monotonic();last_checkpoint=0.
    def metric(x,details=False):
        nonlocal calls,best,last_checkpoint
        calls+=1;m=model.unpack(x);co=np.ascontiguousarray(m['co'],dtype=np.float64);cuts=np.ascontiguousarray(model.cuts,dtype=np.float64);r=[];report=[]
        for I,C,cdot,cmin,cmax,period in rows:
            q0=float(model.basis(C,I,m)@m['N']+cdot);out=np.zeros(6,dtype=np.float64)
            status=lib.simulate(co.ctypes.data_as(ptr),cuts.ctypes.data_as(ptr),I,C,q0,a.dt,steps,out.ctypes.data_as(ptr))
            rr=[(out[0]-cmin)/.025,(out[1]-cmax)/a.cmax_scale,(out[2]-period)/2.] if status and out[3]>=2 and out[1]-out[0]>.01 else [8.,8.,8.]
            r.extend(rr);report.append(dict(I=I,initial_C=C,initial_Cdot=cdot,initial_q=q0,valid=bool(status),peaks=int(out[3]),Cmin=float(out[0]),Cmax=float(out[1]),period=float(out[2]) if out[3]>=2 else None,target_Cmin=cmin,target_Cmax=cmax,target_period=period,q_min=float(out[4]),q_max=float(out[5])))
        extra=[]
        for I,C,tr,det in anchors:
            eq=model.equilibrium(I,m)
            if not eq:extra.extend([5.,0.,0.]);continue
            nearest=min(eq,key=lambda e:abs(e['C']-C));region=nearest['region']
            extra.extend([(nearest['C']-C)/.05,.2*(m['T'][region]-tr),.2*(m['D'][region]-det)/.15])
        result=np.r_[np.asarray(r)/np.sqrt(len(rows)),np.sqrt(a.equilibrium_weight)*np.asarray(extra)/np.sqrt(len(anchors))];cost=float(result@result)
        if cost<best:
            best=cost;now=time.monotonic()
            if now-last_checkpoint>5 or calls==1:
                tmp=a.out/'checkpoint_unverified.tmp';tmp.write_text(json.dumps(dict(parameters=np.asarray(x).tolist(),loss=cost,eval=calls,note='Unverified intermediate; fold may not hold.'),indent=2)+'\n');tmp.replace(a.out/'checkpoint_unverified.json');last_checkpoint=now
        if calls==1 or calls%100==0:print(f'FREE {model.case} eval={calls} best={best:.6g} elapsed={time.monotonic()-start:.1f}s',flush=True)
        return (result,report) if details else result
    active=model.active;lower=model.lo[active]+1e-7;upper=model.hi_bounds[active]-1e-7;rng=np.random.default_rng(a.random_seed)
    startx=np.clip(x0[active]+rng.normal(0,a.jitter,len(active)),lower+1e-9,upper-1e-9)
    if model.case=='am':
        def expand(v):x=x0.copy();x[active]=v;return model.constrain_events(x)
        sol=least_squares(lambda v:metric(expand(v)),startx,bounds=(lower,upper),max_nfev=a.max_nfev,diff_step=2e-4,x_scale='jac',ftol=1e-7,xtol=1e-7,gtol=1e-7)
        xx=expand(sol.x);rx=metric(xx);r0=metric(x0);x=xx if rx@rx<r0@r0 else x0;fold_info=None
        opt=dict(success=bool(sol.success),message=str(sol.message),nfev=int(sol.nfev),optimality=float(sol.optimality),selected_seed=bool(np.array_equal(x,x0)))
    else:
        original=seed.get('fold_constraint')
        if not original:raise ValueError('FM seed must contain a verified fold')
        fold=BoundaryFold(model,original['I']);times=np.asarray(original['crossing_times'],float);n=len(times);v0=np.r_[startx,times];bounds=list(zip(np.r_[lower,np.full(n,1e-5)],np.r_[upper,np.full(n,30.)]))
        def expand(v):x=x0.copy();x[active]=v[:len(active)];return model.constrain_events(x),v[len(active):]
        def constraint(v):
            try:
                x,t=expand(v);m=model.unpack(x);return np.r_[fold.matching(x,t),sum(m['T'][r]*dt for r,dt in zip(fold.regions,t))/10]
            except (ValueError,FloatingPointError,OverflowError):return np.full(n+1,1e5)
        def objective(v):
            try:r=metric(expand(v)[0]);return float(r@r)
            except (ValueError,FloatingPointError,OverflowError):return 1e12
        promising=[]
        def callback(v):
            err=float(max(abs(constraint(v))))
            if err<1e-5:
                score=objective(v);promising.append((score,np.asarray(v).copy(),err));promising.sort(key=lambda row:row[0]);del promising[5:]
        sol=minimize(objective,v0,method='SLSQP',bounds=bounds,constraints=[dict(type='eq',fun=constraint)],callback=callback,options=dict(maxiter=a.max_nfev,ftol=1e-7,eps=1e-4))
        errors=[];candidates=[]
        for _,v,err in promising+[(float(sol.fun),sol.x,float(max(abs(constraint(sol.x)))) )]:
            raw,t=expand(v)
            for dep in active:
                try:
                    y,info=fold.solve(raw,t,int(dep),max_nfev=500);info['independent_check']=check_fold(model,y,fold,info);r=metric(y);candidates.append((float(r@r),y,info));break
                except (ValueError,FloatingPointError,OverflowError,np.linalg.LinAlgError) as e:errors.append(f'{dep}: {e}')
        fallback=None
        for dep in active:
            try:
                y,info=fold.solve(x0,times,int(dep),max_nfev=500);info['independent_check']=check_fold(model,y,fold,info);r=metric(y);fallback=(float(r@r),y,info);break
            except (ValueError,FloatingPointError,OverflowError,np.linalg.LinAlgError) as e:errors.append(f'fallback {dep}: {e}')
        if fallback is not None:candidates.append(fallback)
        if not candidates:raise ValueError('Could not verify even the transformed original fold: '+'; '.join(errors))
        _,x,fold_info=min(candidates,key=lambda t:t[0]);opt=dict(success=bool(sol.success),message=str(sol.message),nfev=int(sol.nfev),nit=int(sol.nit),constraint_residual=float(max(abs(constraint(sol.x)))),fold_errors=errors)
    rr,report=metric(x,True);m=model.unpack(x)
    obj=dict(case=model.case.upper(),threshold_count_m=2,model_family='physical_C_free_latent',gate_coordinate='free',hidden_variable='q is scaled by gauge, not physical LR h',gauge='q(rest)=0, coefficient of q in Cdot is 1',hidden_bounds=None,physical_thresholds_C=model.cuts.tolist(),fixed_equilibrium_threshold_I=model.events.tolist(),rest_C_target=float(model.rest[0]),rest_actual=model.equilibrium(0.,m),parameters=x.tolist(),alpha=m['N'].tolist(),beta=m['H'].tolist(),a=1.,b=float(m['b']),trace=m['T'].tolist(),determinant=m['D'].tolist(),discriminant=(m['T']**2-4*m['D']).tolist(),fold_constraint=fold_info,trajectory_fit=dict(loss=float(rr@rr),duration=a.duration,dt=a.dt,cmax_scale=a.cmax_scale,equilibrium_weight=a.equilibrium_weight,jitter=a.jitter,random_seed=a.random_seed,targets=report,optimizer=opt,initialization='For each reference trajectory use its observed near-peak C0 and Cdot; q0=N(C0,I)+Cdot. No fit to h or h nullclines.',method='Finite RK4 trajectories, no Poincare; equilibrium calcium and spectra regularizer, no hidden-state targets. Independent DOP853 required.'),accepted=False)
    (a.out/'parameters.json').write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    print(f'SAVED {a.out / "parameters.json"} loss={float(rr@rr):.7g} fold_verified={fold_info is not None}',flush=True)

if __name__=='__main__':main()
