#!/usr/bin/env python3
"""Refine physical-calcium models from fixed-input trajectories, no Poincare."""
import argparse
import ctypes
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares, minimize
from fit_physical_calcium import PhysicalModel, data, residual
from fold import BoundaryFold
from run import independent_fold

ROOT=Path(__file__).resolve().parent

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('seed',type=Path);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--duration',type=float,default=240);ap.add_argument('--dt',type=float,default=.04)
    ap.add_argument('--max-nfev',type=int,default=1800);ap.add_argument('--geometry-weight',type=float,default=.02)
    ap.add_argument('--cmax-scale',type=float,default=.08,help='Cmax residual scale in micromolar; lower means higher weight')
    ap.add_argument('--seed-jitter',type=float,default=0.,help='Gaussian perturbation of active parameters before optimization')
    ap.add_argument('--random-seed',type=int,default=20261029)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    seed=json.loads((args.seed/'parameters.json').read_text());model=PhysicalModel(seed['case'].lower(),seed['gate_coordinate']);x0=np.asarray(seed['parameters'],float);m0=model.unpack(x0);d=data(model)
    reference=json.loads((args.seed/'trajectory_comparison.json').read_text());samples=np.load(model.root/'orbit_samples.npz');rows=[]
    for row in reference['rows']:
        target=row['LR'];I=float(row['I']);inds=np.flatnonzero(np.isclose(samples['I'],I,rtol=0,atol=1e-10))
        if len(inds) and target['period'] is not None:
            j=inds[0];rows.append((I,float(samples['C'][j]),float(samples['h'][j]),float(target['Cmin']),float(target['Cmax']),float(target['period'])))
    if not rows:raise ValueError('No matching reference trajectory targets')
    if args.duration<60 or args.dt<=0 or args.geometry_weight<0 or args.cmax_scale<=0 or args.seed_jitter<0:raise ValueError('Invalid settings')
    libpath=args.out/'trajectory_engine.so';src=ROOT/'physical_trajectory_engine.cpp'
    subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(src),'-o',str(libpath)],check=True)
    lib=ctypes.CDLL(str(libpath));pointer=ctypes.POINTER(ctypes.c_double)
    lib.simulate.argtypes=[pointer,pointer,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_int,pointer];lib.simulate.restype=ctypes.c_int
    steps=round(args.duration/args.dt);counts=0;best=np.inf;start=time.monotonic();last_checkpoint=0.
    def metric(x,details=False):
        nonlocal counts,best,last_checkpoint
        counts+=1;m=model.unpack(x);co=np.ascontiguousarray(m['co'],dtype=np.float64);cuts=np.ascontiguousarray(model.cuts,dtype=np.float64);rr=[];report=[]
        for I,C,h,cmin,cmax,period in rows:
            output=np.zeros(6,dtype=np.float64)
            status=lib.simulate(co.ctypes.data_as(pointer),cuts.ctypes.data_as(pointer),I,C,h**m['gamma'],args.dt,steps,output.ctypes.data_as(pointer))
            if status and output[3]>=2 and output[1]-output[0]>.01:
                r=[(output[0]-cmin)/.025,(output[1]-cmax)/args.cmax_scale,(output[2]-period)/2.]
            else:r=[8.,8.,8.]
            rr.extend(r);report.append(dict(I=I,valid=bool(status),peaks=int(output[3]),Cmin=float(output[0]),Cmax=float(output[1]),period=float(output[2]) if output[3]>=2 else None,target_Cmin=cmin,target_Cmax=cmax,target_period=period,gate_min=float(output[4]),gate_max=float(output[5])))
        r=np.r_[np.asarray(rr)/np.sqrt(len(rows)),np.sqrt(args.geometry_weight)*residual(model,x,d)];cost=float(r@r)
        if cost<best:
            best=cost;now=time.monotonic()
            if now-last_checkpoint>=5. or counts==1:
                checkpoint=dict(case=model.case,gate=model.gate,parameters=np.asarray(x).tolist(),loss=cost,eval=counts,elapsed_s=now-start,scope='Unverified intermediate parameters; FM fold and physical admissibility are not guaranteed. Not a final candidate.')
                tmp=args.out/'checkpoint_unverified.tmp';tmp.write_text(json.dumps(checkpoint,indent=2,allow_nan=False)+'\n');tmp.replace(args.out/'checkpoint_unverified.json');last_checkpoint=now
        if counts==1 or counts%100==0:print(f'FIT {model.case} {model.gate} eval={counts} best={best:.6g} elapsed={time.monotonic()-start:.1f}s',flush=True)
        return (r,report) if details else r
    active=model.active;lo=model.lo[active]+1e-7;hi=model.hi_bounds[active]-1e-7
    rng=np.random.default_rng(args.random_seed)
    start_active=np.clip(x0[active]+rng.normal(0,args.seed_jitter,len(active)),lo+1e-9,hi-1e-9)
    if model.case=='am':
        def expand(v):x=x0.copy();x[active]=v;return model.constrain_events(x)
        sol=least_squares(lambda v:metric(expand(v)),start_active,bounds=(lo,hi),max_nfev=args.max_nfev,diff_step=2e-4,x_scale='jac',ftol=1e-7,xtol=1e-7,gtol=1e-7)
        candidate=expand(sol.x);r_candidate=metric(candidate);r_original=metric(x0)
        x=candidate if model.physically_valid_events(candidate) and r_candidate@r_candidate<r_original@r_original else x0
        info=None;opt=dict(success=bool(sol.success),message=str(sol.message),nfev=int(sol.nfev),optimality=float(sol.optimality),selected_seed=bool(np.array_equal(x,x0)))
    else:
        if not seed.get('fold_constraint'):raise ValueError('FM seed must have a verified fold')
        info0=seed['fold_constraint'];fold=BoundaryFold(model,info0['I']);times=np.asarray(info0['crossing_times'],float);n=len(times)
        v0=np.r_[start_active,times];bounds=list(zip(np.r_[lo,np.full(n,1e-5)],np.r_[hi,np.full(n,30.)]))
        def expand(v):x=x0.copy();x[active]=v[:len(active)];return model.constrain_events(x),v[len(active):]
        def constraint(v):
            try:
                x,t=expand(v);m=model.unpack(x);return np.r_[fold.matching(x,t),sum(m['T'][r]*dt for r,dt in zip(fold.regions,t))/10]
            except (ValueError,FloatingPointError,OverflowError):return np.full(n+1,1e5)
        def physical(v):
            x,_=expand(v);m=model.unpack(x);h=model.basis(model.cuts,model.events,m)@m['N'];return np.r_[h-1e-6,1-1e-6-h]
        def objective(v):
            try:r=metric(expand(v)[0]);return float(r@r)
            except (ValueError,FloatingPointError,OverflowError):return 1e12
        sol=minimize(objective,v0,method='SLSQP',bounds=bounds,constraints=[dict(type='eq',fun=constraint),dict(type='ineq',fun=physical)],options=dict(maxiter=args.max_nfev,ftol=1e-7,eps=1e-4))
        raw,t=expand(sol.x);opt=dict(success=bool(sol.success),message=str(sol.message),nfev=int(sol.nfev),nit=int(sol.nit),constraint_residual=float(max(abs(constraint(sol.x)))))
        candidates=[];errors=[]
        for idx in active:
            try:
                xx,fi=fold.solve(raw,t,int(idx),max_nfev=500);fi['independent_check']=independent_fold(model,xx,fold,fi);candidates.append((float(metric(xx)@metric(xx)),xx,fi));break
            except (ValueError,FloatingPointError,OverflowError,np.linalg.LinAlgError) as e:errors.append(f'{idx}: {e}')
        info0=dict(info0);info0['independent_check']=independent_fold(model,x0,fold,info0);r0=metric(x0);candidates.append((float(r0@r0),x0,info0))
        _,x,info=min(candidates,key=lambda k:k[0]);opt['fold_solver_errors']=errors
    r,report=metric(x,True);m=model.unpack(x);final=dict(seed)
    final.update(parameters=x.tolist(),alpha=m['N'].tolist(),beta=m['H'].tolist(),a=float(m['a']),b=float(m['b']),gate_power=float(m['gamma']),gate_power_bounds=[model.gamma_min,model.gamma_max],trace=m['T'].tolist(),determinant=m['D'].tolist(),discriminant=(m['T']**2-4*m['D']).tolist(),event_q=(model.basis(model.cuts,model.events,m)@m['N']).tolist(),rest_actual=model.equilibrium(0.,m),fold_constraint=info,trajectory_fit=dict(loss=float(r@r),duration=args.duration,dt=args.dt,geometry_weight=args.geometry_weight,cmax_scale=args.cmax_scale,seed_jitter=args.seed_jitter,random_seed=args.random_seed,targets=report,optimizer=opt,method='RK4 fixed-input extrema and period over the second half of a finite trajectory; no Poincare map. Target inputs used for training; independent DOP853 validation still required.'),accepted=False)
    (args.out/'parameters.json').write_text(json.dumps(final,indent=2,allow_nan=False)+'\n')
    print(f'SAVED {args.out / "parameters.json"} loss={float(r@r):.7g} fold_verified={info is not None}',flush=True)

if __name__=='__main__':main()
