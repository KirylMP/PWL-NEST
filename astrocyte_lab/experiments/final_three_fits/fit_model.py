#!/usr/bin/env python3
"""Constrained two-threshold fold and stable-cycle period/waveform fit."""
import argparse,ctypes,json,sys,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares,minimize
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'astrocyte_lab/experiments/high_k_family'))
from fit_high_k import configure
from fit_k import lr_rhs
from fold import BoundaryFold
from run import independent_fold

def main():
 p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--engine',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int,default=1);p.add_argument('--maxiter',type=int,default=1000);p.add_argument('--duration',type=float,default=350);p.add_argument('--dt',type=float,default=.02);p.add_argument('--jitter',type=float,default=.1);p.add_argument('--init-max-nfev',type=int,default=1500);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 ref=json.loads(a.reference.read_text());K=ref['K'];model,fixed,_=configure(K)
 if not fixed:raise ValueError('Both LR equilibrium crossings are required')
 fold=BoundaryFold(model,ref['fold']['I'],first=0,last=1);n=fold.legs
 rows=json.loads((ROOT/'astrocyte_lab/experiments/joint_linear_family/reference.json').read_text())['models'];r=min(rows,key=lambda q:abs(q['K']-K));base=np.array(r['parameters'],float);active=model.active
 seed_file=json.loads((ROOT/'astrocyte_lab/models/fm_2thr_traj_identity_20261029/parameters.json').read_text());seed_t=seed_file['fold_constraint']['crossing_times']
 rng=np.random.default_rng(a.seed);start=np.r_[base[active]+rng.normal(0,a.jitter,len(active)),np.array(seed_t)*np.exp(rng.normal(0,a.jitter,n))]
 lo=np.r_[model.lo[active]+1e-7,np.full(n,1e-5)];hi=np.r_[model.hi_bounds[active]-1e-7,np.full(n,30.)];start=np.clip(start,lo+1e-8,hi-1e-8)
 lib=ctypes.CDLL(str(a.engine.resolve()));ptr=ctypes.POINTER(ctypes.c_double);lib.simulate_shape.argtypes=[ptr,ptr,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_int,ctypes.c_int,ptr];lib.simulate_shape.restype=ctypes.c_int
 cycles=ref['cycles'];train=[z for z in cycles if z['split']=='train'];nphase=ref['phase_count'];steps=round(a.duration/a.dt);cuts=np.ascontiguousarray(model.cuts,dtype=np.float64);calls=0;best=1e100;started=time.monotonic();checkpoint_time=0.
 def expand(v):
  x=base.copy();x[active]=v[:len(active)];return model.constrain_events(x),v[len(active):]
 def constraints(v):
  try:
   x,t=expand(v);m=model.unpack(x);f=np.r_[fold.matching(x,t),sum(m['T'][j]*dt for j,dt in zip(fold.regions,t))/10];return f if np.isfinite(f).all() else np.full(n+1,1e3)
  except (ValueError,OverflowError,FloatingPointError,np.linalg.LinAlgError):return np.full(n+1,1e3)
 def physical(v):
  x,_=expand(v);m=model.unpack(x);q=model.basis(model.cuts,model.events,m)@m['N'];return np.r_[q-1e-6,1-1e-6-q]
 def evaluate(x,subset):
  m=model.unpack(x);co=np.ascontiguousarray(m['co'],dtype=np.float64);res=[];detail=[]
  for q in subset:
   o=np.zeros(4+nphase);ok=bool(lib.simulate_shape(co.ctypes.data_as(ptr),cuts.ctypes.data_as(ptr),q['I'],*q['initial'],a.dt,steps,nphase,o.ctypes.data_as(ptr)))
   if ok:
    shape=np.asarray(q['shape']);pred=o[4:];res.extend(((pred-shape)/.08/np.sqrt(nphase)).tolist());res.append((o[2]-q['period'])/2.)
    detail.append(dict(I=q['I'],valid=True,Cmin=float(o[0]),Cmax=float(o[1]),period=float(o[2]),shape=pred.tolist(),target_Cmin=q['Cmin'],target_Cmax=q['Cmax'],target_period=q['period'],target_shape=q['shape']))
   else:
    res.extend([8./np.sqrt(nphase)]*nphase+[8.]);detail.append(dict(I=q['I'],valid=False))
  return np.asarray(res)/np.sqrt(len(subset)),detail
 def objective(v):
  nonlocal calls,best,checkpoint_time
  calls+=1;x,_=expand(v);res,_=evaluate(x,train);cost=float(res@res);err=float(max(abs(constraints(v))))
  if cost<best and err<1e-5:
   best=cost;now=time.monotonic()
   if now-checkpoint_time>20:
    tmp=a.out/'checkpoint.tmp';tmp.write_text(json.dumps(dict(status='unverified_checkpoint',K=K,parameters=x.tolist(),fold_times=expand(v)[1].tolist(),loss=cost,constraint_residual=err),indent=2));tmp.replace(a.out/'checkpoint.json');checkpoint_time=now
  if calls==1 or calls%100==0:print(f'FIT K={K:.3f} eval={calls} loss={cost:.6g} best={best:.6g} fold_res={err:.3g} elapsed={time.monotonic()-started:.1f}s',flush=True)
  return cost
 init_calls=0
 def init_residual(v):
  nonlocal init_calls
  init_calls+=1;f=constraints(v)
  if init_calls==1 or init_calls%200==0:print(f'FOLD INIT K={K:.3f} eval={init_calls} residual={max(abs(f)):.4g} elapsed={time.monotonic()-started:.1f}s',flush=True)
  return np.r_[10*f,10*np.maximum(-physical(v),0),1e-4*(v-start)]
 init=least_squares(init_residual,start,bounds=(lo,hi),max_nfev=a.init_max_nfev,x_scale='jac',ftol=1e-10,xtol=1e-10,gtol=1e-10)
 print(f'FOLD INIT K={K:.3f} residual={max(abs(constraints(init.x))):.3g} nfev={init.nfev}',flush=True)
 def project(v):
  x,t=expand(v);errors=[]
  for idx in active:
   try:
    print(f'FOLD VERIFY K={K:.3f} parameter={idx} elapsed={time.monotonic()-started:.1f}s',flush=True)
    y,info=fold.solve(x,t,int(idx),max_nfev=500);info['independent_check']=independent_fold(model,y,fold,info);return y,info,errors
   except (ValueError,FloatingPointError,OverflowError,np.linalg.LinAlgError) as e:errors.append(f'{idx}: {e}')
  return None,None,errors
 candidates=[];x0,fi0,errors0=project(init.x)
 if fi0 is not None:candidates.append((float(evaluate(x0,train)[0]@evaluate(x0,train)[0]),x0,fi0,'initial'))
 sol=minimize(objective,init.x,method='SLSQP',bounds=list(zip(lo,hi)),constraints=[dict(type='eq',fun=constraints),dict(type='ineq',fun=physical)],options=dict(maxiter=a.maxiter,ftol=1e-7,eps=1e-4))
 x1,fi1,errors1=project(sol.x)
 if fi1 is not None:candidates.append((float(evaluate(x1,train)[0]@evaluate(x1,train)[0]),x1,fi1,'optimized'))
 result=dict(K=K,status='fold_unresolved',scope='Fixed LR rest and LR equilibrium crossing inputs; fold matching M=1 on both-threshold itinerary at LR fold I. No Poincare loss; stable-cycle C waveform and period fit. Fold local nondegeneracy checked; global branch completeness unproved.',reference_fold=ref['fold'],fixed_equilibrium_threshold_I=model.events.tolist(),physical_thresholds_C=model.cuts.tolist(),rest_target=model.rest.tolist(),optimizer=dict(success=bool(sol.success),message=str(sol.message),nfev=int(sol.nfev),initial_residual=float(max(abs(constraints(init.x))),),final_residual=float(max(abs(constraints(sol.x)))),initial_errors=errors0,final_errors=errors1))
 if candidates:
  loss,x,info,selected=min(candidates,key=lambda z:z[0]);m=model.unpack(x);tr,trd=evaluate(x,train);ho,hod=evaluate(x,[z for z in cycles if z['split']=='holdout']);valid=all(q['valid'] for q in trd)
  result.update(status='fold_verified' if valid else 'fold_verified_training_cycles_missing',loss=loss,selected=selected,case='FM' if abs(K-.05)<1e-9 else ('AM' if abs(K-.1)<1e-9 else 'INTERMEDIATE'),observable_calcium='physical C',gate_coordinate='identity',gate_power=1.,parameters=x.tolist(),alpha=m['N'].tolist(),beta=m['H'].tolist(),a=float(m['a']),b=float(m['b']),trace=m['T'].tolist(),determinant=m['D'].tolist(),event_q=(model.basis(model.cuts,model.events,m)@m['N']).tolist(),fold_constraint=info,training_cycles=trd,holdout_cycles=hod)
 (a.out/'parameters.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print('SAVED',a.out/'parameters.json',result['status'],flush=True)
if __name__=='__main__':main()
