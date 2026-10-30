#!/usr/bin/env python3
"""Fit AM or FM with physical calcium and identity or power gate coordinate.

All residuals compare physical h and physical derivatives. No Poincare maps.
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.special import expit

ROOT=Path(__file__).resolve().parents[2]
EXPERIMENT=ROOT/'experiments/fm_am_minimal_zones'
sys.path.insert(0,str(EXPERIMENT/'fm_threshold_refinement'))
from fold import BoundaryFold
from run import independent_fold

def load_source(case):
    root=EXPERIMENT/('am_two_thresholds' if case=='am' else 'fm_four_zone_fit')
    spec=importlib.util.spec_from_file_location(f'lr_{case}',root/'support/lr_reference.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module,root

class PhysicalModel:
    def __init__(self,case,gate):
        self.case=case;self.gate=gate;self.lr,self.root=load_source(case)
        self.events=np.array([.35453381865657685,.6368818163953428] if case=='am' else [.5290700389163683,.8600205557570602])
        self.cuts=np.array([self.lr.eq(I)[0] for I in self.events] if case=='am' else [.05214116272147648,.3926251934101712])
        self.rest=np.asarray(self.lr.eqs(0)[0],float);self.n=3;self.f=self
        # log(a,b), first-region stable margin, log(T1), log(-T2), log excess D1,D2, alpha_I, log(gamma)
        self.gamma_min=float(os.environ.get('GATE_POWER_MIN','0.5'));self.gamma_max=float(os.environ.get('GATE_POWER_MAX','2.5'))
        if not 0<self.gamma_min<=1<=self.gamma_max or self.gamma_min>=self.gamma_max:raise ValueError('Gate power bounds must contain 1')
        self.size=9;self.lo=np.array([-5.,-5.,-6.,-6.,-6.,-10.,-10.,-4.,np.log(self.gamma_min)])
        self.hi_bounds=np.array([4.,1.,4.,2.,2.,2.,2.,4.,np.log(self.gamma_max)])
        self.active=np.arange(9 if gate=='power' else 8)

    @staticmethod
    def phi(c,shape=None):return np.asarray(c,float)

    @property
    def CMAX(self):return self.lr.CMAX

    def constrain_events(self,x):
        x=np.asarray(x,float).copy()
        if self.gate=='identity':x[8]=0.
        return x

    def unpack(self,x):
        x=self.constrain_events(x);a,b=np.exp(x[:2]);T1=np.exp(x[3]);T2=-np.exp(x[4])
        D1=T1*T1/4+np.exp(x[5]);D2=T2*T2/4+np.exp(x[6])
        D0=self.events[0]*D1*(self.cuts[1]-self.cuts[0])/((self.events[1]-self.events[0])*(self.cuts[0]-self.rest[0]))
        T0=-(2*np.sqrt(D0)+np.exp(x[2])) if self.case=='fm' else -2*np.sqrt(D0)*expit(x[2])
        T=np.array([T0,T1,T2]);D=np.array([D0,D1,D2]);ns=-(T+b)/a;hs=ns-D/(a*b)
        gamma=np.exp(x[8]);q0=self.rest[1]**gamma;s=(ns[0]-hs[0])*(self.cuts[0]-self.rest[0])/self.events[0]
        N=np.r_[q0-ns[0]*self.rest[0],ns[0],x[7],np.diff(ns)]
        H=np.r_[q0-hs[0]*self.rest[0],hs[0],x[7]+s,np.diff(hs)]
        co=np.zeros((2,11));idx=[0,1,3,4,5];co[0,idx]=-a*N;co[0,2]=a;co[1,idx]=b*H;co[1,2]=-b
        return dict(a=a,b=b,T=T,D=D,N=N,H=H,gamma=gamma,th=self.cuts.copy(),tc=self.cuts.copy(),co=co,padded=np.r_[self.cuts,np.full(5,1e100)],shape=None)

    def basis(self,c,I,m):
        c,I=np.broadcast_arrays(np.asarray(c,float),I)
        return np.stack([np.ones_like(c),c,I,*[np.maximum(c-t,0) for t in self.cuts]],axis=-1)

    def physically_valid_events(self,x):
        m=self.unpack(x);h=self.basis(self.cuts,self.events,m)@m['N']
        return bool(np.all(np.isfinite(h)) and np.all((0<h)&(h<1)))

    def equilibrium(self,I,m):
        slopes=m['N'][1]+np.r_[0,np.cumsum(m['N'][3:])];gs=m['H'][1]+np.r_[0,np.cumsum(m['H'][3:])]
        n0=m['N'][0]-np.r_[0,np.cumsum(m['N'][3:]*self.cuts)];h0=m['H'][0]-np.r_[0,np.cumsum(m['H'][3:]*self.cuts)]
        c=(h0-n0+(m['H'][2]-m['N'][2])*I)/(slopes-gs);q=n0+slopes*c+m['N'][2]*I
        valid=(c>np.r_[-np.inf,self.cuts])&(c<=np.r_[self.cuts,np.inf])&(q>0)&(q<1)&(c>0)&(c<self.CMAX)
        return [dict(C=float(c[r]),z=[float(c[r]),float(q[r])],h=float(q[r]**(1/m['gamma'])),region=int(r),stable=bool(m['T'][r]<0 and m['D'][r]>0)) for r in np.flatnonzero(valid)]

def data(model):
    case=model.case;lo,hi=(.25,.75) if case=='am' else (.4,1.2)
    C,I=np.meshgrid(np.linspace(.03,1.15,55),np.linspace(lo,hi,11),indexing='ij')
    pump=(.9*C*C/(C*C+(.01 if case=='am' else .0025))-.11*(2-1.185*C))
    raw=np.cbrt(pump/(6*(I/(I+.13))**3*(C/(C+.08234))**3*(2-1.185*C)))
    gate=1.049*(I+.13)/(I+.9434);other=gate/(gate+C)
    targets=np.stack([raw,other],axis=-1);mask=(targets>1e-4)&(targets<1-1e-4)
    anchors=[]
    for I0 in np.linspace(lo,hi,30):
        for e in model.lr.eqs(float(I0)):
            J=model.lr.jac(e,float(I0));anchors.append((float(I0),float(e[0]),float(e[1]),float(np.trace(J)),float(np.linalg.det(J))))
    orbit=np.load(model.root/'orbit_samples.npz')
    return C,I,targets,mask,np.asarray(anchors),{k:orbit[k] for k in ('C','h','I','vel')}

def residual(model,x,data):
    C,I,target,mask,anchors,orb=data;m=model.unpack(x);gamma=m['gamma'];bb=model.basis(C,I,m);outputs=np.stack([bb@m['N'],bb@m['H']],axis=-1)
    q=np.maximum(target,1e-6)**gamma;dq=gamma*np.maximum(target,1e-6)**(gamma-1)
    geom=np.where(mask,(outputs-q)/dq,0.)
    b=model.basis(anchors[:,1],anchors[:,0],m);qa=anchors[:,2]**gamma;da=gamma*anchors[:,2]**(gamma-1)
    eq=np.column_stack([(b@m['N']-qa)/da,(b@m['H']-qa)/da]);regions=np.searchsorted(model.cuts,anchors[:,1])
    spectrum=np.column_stack([m['T'][regions]-anchors[:,3],(m['D'][regions]-anchors[:,4])/.15])
    b=model.basis(orb['C'],orb['I'],m);qorb=orb['h']**gamma;der=gamma*orb['h']**(gamma-1)
    pred=np.column_stack([m['a']*(qorb-b@m['N']),m['b']*(b@m['H']-qorb)/der]);field=(pred-orb['vel'])/[.035,.006]
    heights=model.basis(model.cuts,model.events,m)@m['N'];domain=100*np.r_[np.maximum(-heights,0),np.maximum(heights-1,0)]
    field_weight=float(os.environ.get('PHYSICAL_FIELD_WEIGHT','1'))
    if not np.isfinite(field_weight) or field_weight<=0:raise ValueError('PHYSICAL_FIELD_WEIGHT must be positive')
    return np.r_[(geom/.025/np.sqrt(C.size)).ravel(),(eq/.04/np.sqrt(len(anchors))).ravel(),(.35*spectrum/np.sqrt(len(anchors))).ravel(),(np.sqrt(field_weight)*field/np.sqrt(len(field))).ravel(),domain]

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('case',choices=['am','fm']);ap.add_argument('gate',choices=['identity','power']);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--starts',type=int,default=8);ap.add_argument('--max-nfev',type=int,default=1500);ap.add_argument('--fold-seed',type=Path);ap.add_argument('--fold-starts',type=int,default=1)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True);model=PhysicalModel(args.case,args.gate);d=data(model)
    rng=np.random.default_rng(21129);base=np.array([np.log(3.),np.log(.2),-1.,np.log(.2),np.log(.4),np.log(.1),np.log(.2),-.2,0.]);best=np.inf;bestx=None;runs=[]
    for j in range(args.starts):
        x=base.copy() if j==0 else base+rng.normal(0,[.5,.4,1.,.6,.5,.7,.7,.15,.3]);x=np.clip(x,model.lo+1e-7,model.hi_bounds-1e-7)
        def expand(v):y=x.copy();y[model.active]=v;return model.constrain_events(y)
        sol=least_squares(lambda v:residual(model,expand(v),d),x[model.active],bounds=(model.lo[model.active],model.hi_bounds[model.active]),max_nfev=args.max_nfev,x_scale='jac')
        y=expand(sol.x);cost=float(sol.fun@sol.fun);runs.append(dict(start=j+1,loss=cost,nfev=int(sol.nfev),optimality=float(sol.optimality),success=bool(sol.success)))
        print(f'{args.case.upper()} {args.gate} start={j+1}/{args.starts} loss={cost:.7g} nfev={sol.nfev} gamma={np.exp(y[8]):.6g}',flush=True)
        if cost<best and model.physically_valid_events(y):best=cost;bestx=y
    if bestx is None:raise ValueError('No admissible geometry candidate')
    (args.out/'geometry_seed.json').write_text(json.dumps(dict(case=args.case,gate=args.gate,parameters=bestx.tolist(),loss=best,starts=runs),indent=2)+'\n')
    fold_info=None;fold_error=None;fold_opt=None
    if args.case=='fm' and args.fold_seed:
        seed=json.loads(args.fold_seed.read_text());target=float(seed['fold_constraint']['I']);fold=BoundaryFold(model,target);times=np.asarray(seed['fold_constraint']['crossing_times'],float);n=fold.legs
        if len(times)!=n:raise ValueError('Fold seed itinerary mismatch')
        fold_start=bestx.copy()
        if len(seed.get('parameters',[]))==model.size and seed.get('observable_calcium')=='physical C':
            fold_start=np.asarray(seed['parameters'],float).copy()
            if args.gate=='identity' and abs(fold_start[8])>1e-12:raise ValueError('Cannot lift nonlinear gate seed to identity')
            print(f'FOLD using physical-calcium seed {args.fold_seed}',flush=True)
        initial=np.r_[fold_start[model.active],times];lower=np.r_[model.lo[model.active]+1e-8,np.full(n,1e-5)];upper=np.r_[model.hi_bounds[model.active]-1e-8,np.full(n,30.)];initial=np.clip(initial,lower+1e-8,upper-1e-8)
        def expand(v):x=bestx.copy();x[model.active]=v[:len(model.active)];return model.constrain_events(x),v[len(model.active):]
        def constraints(v):
            try:
                x,t=expand(v);m=model.unpack(x);return np.r_[fold.matching(x,t),sum(m['T'][r]*dt for r,dt in zip(fold.regions,t))/10]
            except (ValueError,FloatingPointError,OverflowError):return np.full(n+1,1e5)
        trials=[]
        for j in range(args.fold_starts):
            start=initial.copy()
            if j:
                start[:len(model.active)]+=rng.normal(0,.65,len(model.active))
                start[len(model.active):]*=np.exp(rng.normal(0,.3,n))
                start=np.clip(start,lower+1e-7,upper-1e-7)
            init=least_squares(lambda v:np.r_[constraints(v),1e-7*(v-start)],start,bounds=(lower,upper),max_nfev=1800,x_scale='jac',ftol=1e-11,xtol=1e-11,gtol=1e-11)
            xi,ti=expand(init.x);err=float(max(abs(constraints(init.x))));valid=model.physically_valid_events(xi)
            print(f'FOLD INIT {args.gate} start={j+1}/{args.fold_starts} residual={err:.4g} physical_events={valid} nfev={init.nfev}',flush=True)
            trials.append((err if valid else err+1e3,init))
            if err<1e-8 and valid:break
        init=min(trials,key=lambda p:p[0])[1]
        calls=0
        def objective(v):
            nonlocal calls
            calls+=1;x,_=expand(v);r=residual(model,x,d);value=float(r@r)
            if calls%200==0:print(f'FOLD OPT {args.gate} eval={calls} loss={value:.6g} residual={max(abs(constraints(v))):.3g}',flush=True)
            return value
        def physical(v):
            x,_=expand(v);m=model.unpack(x);h=model.basis(model.cuts,model.events,m)@m['N'];return np.r_[h-1e-6,1-1e-6-h]
        def verify(v):
            xx,tt=expand(v);errors=[]
            for idx in model.active:
                try:
                    yy,info=fold.solve(xx,tt,int(idx),max_nfev=500);info['independent_check']=independent_fold(model,yy,fold,info)
                    return yy,info,errors
                except (ValueError,FloatingPointError,OverflowError,np.linalg.LinAlgError) as e:errors.append(f'{idx}: {e}')
            return None,None,errors
        init_x,init_info,init_errors=verify(init.x)
        print(f'FOLD initial verification={init_info is not None}',flush=True)
        sol=minimize(objective,init.x,method='SLSQP',bounds=list(zip(lower,upper)),constraints=[dict(type='eq',fun=constraints),dict(type='ineq',fun=physical)],options=dict(maxiter=1000,ftol=1e-8))
        fold_opt=dict(success=bool(sol.success),message=str(sol.message),nfev=int(sol.nfev),constraint_residual=float(max(abs(constraints(sol.x)))))
        final_x,final_info,errors=verify(sol.x)
        choices=[(float(residual(model,x,d)@residual(model,x,d)),x,info) for x,info in ((init_x,init_info),(final_x,final_info)) if info is not None]
        if choices:
            _,bestx,fold_info=min(choices,key=lambda q:q[0])
        else:fold_error='initial: '+'; '.join(init_errors)+'; final: '+'; '.join(errors)
    m=model.unpack(bestx);heights=model.basis(model.cuts,model.events,m)@m['N'];eq=model.equilibrium(0,m);loss=float(residual(model,bestx,d)@residual(model,bestx,d))
    result=dict(case=args.case.upper(),threshold_count_m=2,physical_thresholds_C=model.cuts.tolist(),fixed_equilibrium_threshold_I=model.events.tolist(),event_q=heights.tolist(),rest_target=model.rest.tolist(),rest_actual=eq,gate_coordinate=args.gate,gate_power=float(m['gamma']),gate_power_bounds=[model.gamma_min,model.gamma_max],observable_calcium='physical C',parameters=bestx.tolist(),alpha=m['N'].tolist(),beta=m['H'].tolist(),a=float(m['a']),b=float(m['b']),trace=m['T'].tolist(),determinant=m['D'].tolist(),discriminant=(m['T']**2-4*m['D']).tolist(),loss=loss,field_weight=float(os.environ.get('PHYSICAL_FIELD_WEIGHT','1')),geometry_optimizer=runs,fold_constraint=fold_info,fold_optimizer=fold_opt,fold_error=fold_error,method='Rest and equilibrium crossings exact; nullcline, equilibrium, spectrum and LR-orbit vector-field loss; no Poincare or cycle count loss.',accepted=False)
    (args.out/'parameters.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(f'SAVED {args.out / "parameters.json"} loss={loss:.6g} fold_verified={fold_info is not None}',flush=True)
    if args.case=='fm' and args.fold_seed and fold_info is None:raise SystemExit('Fold not verified; geometry-only candidate retained')

if __name__=='__main__':main()
