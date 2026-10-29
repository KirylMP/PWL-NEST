#!/usr/bin/env python3
"""Refine existing randomly generated no-fold candidate with robust stages."""
import argparse,ctypes,json,sys,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares,minimize
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'astrocyte_lab/experiments/mixed_three_fits'))
from mixed_model import MixedModel

def main():
 p=argparse.ArgumentParser();p.add_argument('--seed-model',type=Path,required=True);p.add_argument('--engine',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--powell-maxfev',type=int,default=2500);p.add_argument('--ls-max-nfev',type=int,default=300);p.add_argument('--fd-relative',type=float,default=.001);p.add_argument('--duration',type=float,default=350);p.add_argument('--dt',type=float,default=.01);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 if min(a.powell_maxfev,a.ls_max_nfev)<1 or a.duration<100 or a.dt<=0 or a.fd_relative<=0:raise ValueError('Invalid settings')
 seed=json.loads(a.seed_model.read_text());K=seed['K'];model=MixedModel(K);active=model.active;base=np.asarray(seed['parameters'],float);old=json.loads((ROOT/f'astrocyte_lab/experiments/no_fold_three_fits/seeds/K_{K:.3f}_validation.json').read_text());cycles=[q['LR'] for q in old['cycles']];train=[q for q in cycles if q['split']=='train'];hold=[q for q in cycles if q['split']=='holdout'];nphase=len(train[0]['shape']);lo=model.lo[active]+1e-7;hi=model.hi_bounds[active]-1e-7;start=np.clip(base[active],lo+1e-8,hi-1e-8)
 lib=ctypes.CDLL(str(a.engine.resolve()));ptr=ctypes.POINTER(ctypes.c_double);lib.simulate_shape.argtypes=[ptr,ptr,ptr,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_int,ctypes.c_int,ptr];lib.simulate_shape.restype=ctypes.c_int;steps=round(a.duration/a.dt)
 def expand(v):x=base.copy();x[active]=v;return model.constrain_events(x)
 def samples(x,subset):
  m=model.unpack(x);co=np.ascontiguousarray(m['co'],dtype=np.float64);cuts=np.ascontiguousarray(m['th'],dtype=np.float64);sh=np.ascontiguousarray(m['shape'],dtype=np.float64);res=[];report=[]
  for q in subset:
   o=np.zeros(4+nphase);ok=bool(lib.simulate_shape(co.ctypes.data_as(ptr),cuts.ctypes.data_as(ptr),sh.ctypes.data_as(ptr),q['I'],*q['initial'],a.dt,steps,nphase,o.ctypes.data_as(ptr)))
   if ok:
    shape=o[4:];res.extend(((shape-np.asarray(q['shape']))/.08/np.sqrt(nphase)).tolist());res.append((o[2]-q['period'])/2.)
    report.append(dict(I=q['I'],valid=True,Cmin=float(o[0]),Cmax=float(o[1]),period=float(o[2]),shape=shape.tolist(),target_Cmin=q['Cmin'],target_Cmax=q['Cmax'],target_period=q['period']))
   else:res.extend([8./np.sqrt(nphase)]*nphase+[8.]);report.append(dict(I=q['I'],valid=False))
  return np.asarray(res)/np.sqrt(len(subset)),report
 calls=0;stage='initial';best=float('inf');best_v=None;starttime=time.monotonic();last_checkpoint=0.
 def residual(v):
  nonlocal calls,best,best_v,last_checkpoint
  calls+=1;x=expand(v);m=model.unpack(x);r,report=samples(x,train);q=model.basis(model.cuts,model.events,m)@m['N'];domain=100*np.r_[np.maximum(1e-6-q,0),np.maximum(q-(1-1e-6),0)];full=np.r_[r,domain];loss=float(full@full);valid=bool(model.physically_valid_events(x) and all(z['valid'] for z in report))
  if valid and loss<best:
   best=loss;best_v=v.copy();now=time.monotonic()
   if now-last_checkpoint>=20 or calls==1:
    temp=a.out/'checkpoint.tmp';temp.write_text(json.dumps(dict(status='unverified_checkpoint',K=K,parameters=x.tolist(),loss=loss,evaluations=calls,stage=stage),indent=2));temp.replace(a.out/'checkpoint.json');last_checkpoint=now
  if calls==1 or calls%100==0:print(f'{stage} K={K:.3f} eval={calls} loss={loss:.7g} best_valid={best:.7g} elapsed={time.monotonic()-starttime:.1f}s',flush=True)
  return full
 def finite_jac(v):
  f0=residual(v);cols=[]
  for j in range(len(v)):
   step=a.fd_relative*max(1.,abs(v[j]));vp=v.copy();vm=v.copy();vp[j]=min(hi[j],v[j]+step);vm[j]=max(lo[j],v[j]-step)
   cols.append((residual(vp)-residual(vm))/(vp[j]-vm[j]) if vp[j]>vm[j] else np.zeros_like(f0))
  return np.column_stack(cols)
 r0=residual(start);initial_loss=float(r0@r0);print(f'INITIAL K={K:.3f} loss={initial_loss:.7g}',flush=True)
 def objective(v):
  rr=residual(v);return float(rr@rr)
 stage='powell';ps=minimize(objective,start,method='Powell',bounds=list(zip(lo,hi)),options=dict(maxfev=a.powell_maxfev,xtol=1e-4,ftol=1e-6))
 print(f'POWELL DONE K={K:.3f} nfev={ps.nfev} best={best:.7g} success={ps.success}',flush=True)
 if best_v is None:
  result=dict(K=K,status='no_valid_candidate',initial_loss=initial_loss,scope='No physically valid candidate had all training cycles; fold not calculated.')
  (a.out/'parameters.json').write_text(json.dumps(result,indent=2)+'\n');return
 stage='least_squares';ls=least_squares(residual,best_v,jac=finite_jac,bounds=(lo,hi),max_nfev=a.ls_max_nfev,x_scale='jac',ftol=1e-9,xtol=1e-10,gtol=1e-8)
 print(f'LEAST_SQUARES DONE K={K:.3f} nfev={ls.nfev} best={best:.7g} success={ls.success} optimality={ls.optimality:.6g}',flush=True)
 x=expand(best_v);m=model.unpack(x);r,training=samples(x,train);_,holding=samples(x,hold);valid=model.physically_valid_events(x) and all(q['valid'] for q in training)
 result=dict(K=K,status='training_cycles_found' if valid else 'training_unresolved',scope='No fold condition or fold analysis. Training period and waveform; best physically valid evaluation retained. Powell then central finite-difference least squares. Optimizer termination does not prove stationarity.',seed_path=str(a.seed_model),initial_loss=initial_loss,loss=float(r@r),evaluations=calls,optimizers=dict(powell=dict(success=bool(ps.success),message=str(ps.message),nfev=int(ps.nfev)),least_squares=dict(success=bool(ls.success),message=str(ls.message),nfev=int(ls.nfev),optimality=float(ls.optimality),finite_difference_relative=a.fd_relative)),physical_thresholds_C=model.cuts.tolist(),fixed_equilibrium_threshold_I=model.events.tolist(),rest_target=model.rest.tolist(),event_q=(model.basis(model.cuts,model.events,m)@m['N']).tolist(),case='FM' if abs(K-.05)<1e-9 else ('AM' if abs(K-.1)<1e-9 else 'INTERMEDIATE'),observable_calcium='inverse mixed transform',coordinate_family='mixed',transform=dict(zip(['p','e','n','k'],m['shape'].tolist())),gate_coordinate='identity',gate_power=1.,parameters=x.tolist(),alpha=m['N'].tolist(),beta=m['H'].tolist(),a=float(m['a']),b=float(m['b']),trace=m['T'].tolist(),determinant=m['D'].tolist(),training_cycles=training,holdout_cycles=holding)
 (a.out/'parameters.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print('SAVED',a.out/'parameters.json',result['status'],flush=True)
if __name__=='__main__':main()
