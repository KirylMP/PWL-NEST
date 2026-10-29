#!/usr/bin/env python3
"""Fit stable cycle period and physical-C shape, with no fold constraints."""
import argparse,ctypes,json,sys,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'astrocyte_lab/experiments/mixed_three_fits'))
from mixed_model import MixedModel

def main():
 p=argparse.ArgumentParser();p.add_argument('--K',type=float,required=True);p.add_argument('--engine',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--max-nfev',type=int,default=1200);p.add_argument('--duration',type=float,default=350);p.add_argument('--dt',type=float,default=.02);p.add_argument('--seed',type=int,default=1);p.add_argument('--init-tries',type=int,default=500);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 if a.duration<100 or a.dt<=0 or a.max_nfev<1 or a.init_tries<1:raise ValueError('Invalid fit configuration')
 model=MixedModel(a.K);old=json.loads((HERE/f'seeds/K_{a.K:.3f}_validation.json').read_text());cycles=[q['LR'] for q in old['cycles']];train=[q for q in cycles if q['split']=='train'];hold=[q for q in cycles if q['split']=='holdout'];nphase=len(train[0]['shape']);active=model.active;base=np.zeros(model.size);rng=np.random.default_rng(a.seed)
 lo=model.lo[active]+1e-7;hi=model.hi_bounds[active]-1e-7
 lib=ctypes.CDLL(str(a.engine.resolve()));ptr=ctypes.POINTER(ctypes.c_double);lib.simulate_shape.argtypes=[ptr,ptr,ptr,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_int,ctypes.c_int,ptr];lib.simulate_shape.restype=ctypes.c_int;steps=round(a.duration/a.dt)
 def expand(v):x=base.copy();x[active]=v;return model.constrain_events(x)
 def samples(x,subset):
  m=model.unpack(x);co=np.ascontiguousarray(m['co'],dtype=np.float64);cuts=np.ascontiguousarray(m['th'],dtype=np.float64);sh=np.ascontiguousarray(m['shape'],dtype=np.float64);res=[];report=[]
  for q in subset:
   o=np.zeros(4+nphase);ok=bool(lib.simulate_shape(co.ctypes.data_as(ptr),cuts.ctypes.data_as(ptr),sh.ctypes.data_as(ptr),q['I'],*q['initial'],a.dt,steps,nphase,o.ctypes.data_as(ptr)))
   if ok:
    v=o[4:];res.extend(((v-np.asarray(q['shape']))/.08/np.sqrt(nphase)).tolist());res.append((o[2]-q['period'])/2.)
    report.append(dict(I=q['I'],valid=True,Cmin=float(o[0]),Cmax=float(o[1]),period=float(o[2]),shape=v.tolist(),target_Cmin=q['Cmin'],target_Cmax=q['Cmax'],target_period=q['period']))
   else:res.extend([8./np.sqrt(nphase)]*nphase+[8.]);report.append(dict(I=q['I'],valid=False))
  return np.asarray(res)/np.sqrt(len(subset)),report
 # Independent bounded random draws, followed by a finite feasibility screen.
 # None of the optimized parameters or transformation from previous models is used.
 prior_lo=np.maximum(lo,np.array([np.log(.3),np.log(.05),-2.,-2.7,-3.,-6.,-6.,-1.5,-4.,-6.,.15,-4.]))
 prior_hi=np.minimum(hi,np.array([np.log(12.),np.log(.7),2.,1.,.7,.5,.5,1.5,5.,3.,2.5,4.]))
 probe=[train[0],train[len(train)//2],train[-1]];chosen=None;physical_count=0;started_init=time.monotonic()
 for j in range(a.init_tries):
  v=rng.uniform(prior_lo,prior_hi);x=expand(v)
  try:
   if not model.physically_valid_events(x):continue
   physical_count+=1;r,report=samples(x,probe);coverage=sum(q['valid'] for q in report);score=float(r@r)
   if chosen is None or (coverage,-score)>(chosen[0],-chosen[1]):chosen=(coverage,score,v.copy(),j+1)
   if physical_count==1 or physical_count%50==0:print(f'INITIAL K={a.K:.3f} draws={j+1} physical={physical_count} best_cycles={chosen[0]}/3 elapsed={time.monotonic()-started_init:.1f}s',flush=True)
   if coverage==3:break
  except (ValueError,OverflowError,FloatingPointError,np.linalg.LinAlgError):continue
 if chosen is None or chosen[0]==0:
  result=dict(K=a.K,status='random_initialization_failed',seed=a.seed,draws=a.init_tries,physical_draws=physical_count,best_cycles=0 if chosen is None else chosen[0],scope='Random screen did not find a candidate with a cycle; this does not prove infeasibility.')
  (a.out/'parameters.json').write_text(json.dumps(result,indent=2)+'\n');print('SAVED',a.out/'parameters.json',result['status'],flush=True);return
 start=chosen[2];initialization=dict(method='independent random draws plus finite screening',seed=a.seed,draws_used=chosen[3],physical_draws=physical_count,probe_inputs=[q['I'] for q in probe],cycles_detected=chosen[0],probe_loss=chosen[1],prior_lower=prior_lo.tolist(),prior_upper=prior_hi.tolist(),starting_parameters=expand(start).tolist())
 print(f'INITIAL K={a.K:.3f} selected cycles={chosen[0]}/3 draws={chosen[3]} physical={physical_count}',flush=True)
 count=0;best=float('inf');started=time.monotonic();checkpoint_time=0.
 def residual(v):
  nonlocal count,best,checkpoint_time
  count+=1;x=expand(v);m=model.unpack(x);r,_=samples(x,train);q=model.basis(model.cuts,model.events,m)@m['N'];domain=100*np.r_[np.maximum(1e-6-q,0),np.maximum(q-(1-1e-6),0)];value=np.r_[r,domain];loss=float(value@value)
  if loss<best:
   best=loss;now=time.monotonic()
   if now-checkpoint_time>=20 or count==1:
    temp=a.out/'checkpoint.tmp';temp.write_text(json.dumps(dict(status='unverified_checkpoint',K=a.K,parameters=x.tolist(),loss=loss,evaluations=count),indent=2));temp.replace(a.out/'checkpoint.json');checkpoint_time=now
  if count==1 or count%100==0:print(f'K={a.K:.3f} eval={count} loss={loss:.6g} best={best:.6g} elapsed={time.monotonic()-started:.1f}s',flush=True)
  return value
 sol=least_squares(residual,start,bounds=(lo,hi),max_nfev=a.max_nfev,x_scale='jac',diff_step=2e-4,ftol=1e-7,xtol=1e-7,gtol=1e-7)
 x=expand(sol.x);m=model.unpack(x);r,training=samples(x,train);_,holding=samples(x,hold);domain_valid=model.physically_valid_events(x);passed=domain_valid and all(q['valid'] for q in training)
 result=dict(K=a.K,status='training_cycles_found' if passed else 'training_unresolved',scope='Finite stable-cycle shape and period fit; no fold constraint, calculation, target or search; no unstable-cycle search; rest and two equilibrium threshold crossing constraints exact.',loss=float(r@r),initialization=initialization,optimizer=dict(success=bool(sol.success),message=str(sol.message),nfev=int(sol.nfev),optimality=float(sol.optimality)),physical_thresholds_C=model.cuts.tolist(),fixed_equilibrium_threshold_I=model.events.tolist(),rest_target=model.rest.tolist(),event_q=(model.basis(model.cuts,model.events,m)@m['N']).tolist(),case='FM' if abs(a.K-.05)<1e-9 else ('AM' if abs(a.K-.1)<1e-9 else 'INTERMEDIATE'),observable_calcium='inverse mixed transform',coordinate_family='mixed',transform=dict(zip(['p','e','n','k'],m['shape'].tolist())),gate_coordinate='identity',gate_power=1.,parameters=x.tolist(),alpha=m['N'].tolist(),beta=m['H'].tolist(),a=float(m['a']),b=float(m['b']),trace=m['T'].tolist(),determinant=m['D'].tolist(),training_cycles=training,holdout_cycles=holding)
 (a.out/'parameters.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print('SAVED',a.out/'parameters.json',result['status'],flush=True)
if __name__=='__main__':main()
