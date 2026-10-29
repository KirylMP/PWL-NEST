#!/usr/bin/env python3
"""Fit physical-C two-threshold models at LR equilibrium transitions for K=0.01–0.11."""
import argparse, ctypes, json, sys, time
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, brentq
from scipy.special import expit, logit
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'astrocyte_lab/experiments/phenomenological_k_fit'))
from fit_k import setup,lr_rhs,lr_trace,gate,hopf
from fit_physical_calcium import PhysicalModel

class FlexibleModel(PhysicalModel):
 def unpack(self,x):
  m=super().unpack(x); x=self.constrain_events(x);a,b=m['a'],m['b'];D=m['D']
  T=m['T'].copy();T[0]=-2*np.sqrt(D[0])*np.exp(x[2]);ns=-(T+b)/a;hs=ns-D/(a*b)
  q0=self.rest[1];shift=(ns[0]-hs[0])*(self.cuts[0]-self.rest[0])/self.events[0]
  N=np.r_[q0-ns[0]*self.rest[0],ns[0],x[7],np.diff(ns)]
  H=np.r_[q0-hs[0]*self.rest[0],hs[0],x[7]+shift,np.diff(hs)]
  co=np.zeros((2,11));idx=[0,1,3,4,5];co[0,idx]=-a*N;co[0,2]=a;co[1,idx]=b*H;co[1,2]=-b
  m.update(T=T,N=N,H=H,co=co);return m

def configure(K):
 model=FlexibleModel('am','identity');c0=brentq(lambda c:lr_rhs(c,gate(c,0),0,K)[0],1e-8,.3);model.rest=np.array([c0,gate(c0,0)])
 upper=np.array([.3926251934101712,.8600205557570602])
 for k in np.linspace(.05,K,max(2,int(abs(K-.05)/.001)+1)):upper=hopf(k,upper)
 lower=None
 for c0_guess in [max(.01,K),.05,.1]:
  for i0_guess in [.4,.7,1.,2.,5.]:
   try:
    x=hopf(K,[c0_guess,i0_guess])
    if x[0]<upper[0]-1e-5 and x[1]>0:lower=x;break
   except ValueError:pass
  if lower is not None:break
 fixed=bool(lower is not None and lower[1]<upper[1] and lower[0]>model.rest[0])
 c1=lower[0] if fixed else max(model.rest[0]+.005,min(1.05*K,upper[0]-.01))
 i1=lower[1] if fixed else .5*upper[1]
 model.cuts=np.array([c1,upper[0]]);model.events=np.array([i1,upper[1]])
 return model,fixed,(lower.tolist() if lower is not None else None)

def main():
 p=argparse.ArgumentParser();p.add_argument('--K',type=float,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--engine',type=Path,required=True);p.add_argument('--starts',type=int,default=4);p.add_argument('--max-nfev',type=int,default=1000);p.add_argument('--duration',type=float,default=400.);p.add_argument('--dt',type=float,default=.02);p.add_argument('--input-count',type=int,default=7);p.add_argument('--seed',type=int,default=4108);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 if not .01<=a.K<=.11 or a.starts<1 or a.max_nfev<1 or a.duration<100 or a.dt<=0 or a.input_count<2:raise ValueError('Invalid fit configuration')
 model,fixed_lower,lr_lower=configure(a.K); targets=[]
 scan_low=model.events[0]+.08*(model.events[1]-model.events[0]) if fixed_lower else .05
 scan_high=min(2.,1.5*model.events[1]) if not fixed_lower else model.events[1]-.08*(model.events[1]-scan_low)
 for I in np.linspace(scan_low,scan_high,a.input_count):
  I=float(I)
  def rhs(t,y):return lr_rhs(y[0],y[1],I,a.K)
  def peak(t,y):return rhs(t,y)[0]
  peak.direction=-1;peak.terminal=False
  def trough(t,y):return rhs(t,y)[0]
  trough.direction=1;trough.terminal=False
  peak_count=0
  for initial in ([.7,.8],model.rest.copy()) if a.K<.06 else (model.rest.copy(),[.7,.8]):
   s=solve_ivp(rhs,(0,a.duration),initial,method='DOP853',events=[peak,trough],rtol=1e-8,atol=1e-10,max_step=.3)
   peaks=s.t_events[0];peaks=peaks[peaks>a.duration*.5];mins=s.t_events[1];mins=mins[mins>a.duration*.5];peak_count=len(peaks)
   if s.success and len(peaks)>=3 and len(mins)>=2:
    y=s.y_events[0][s.t_events[0]>a.duration*.5,0];z=s.y_events[1][s.t_events[1]>a.duration*.5,0];periods=np.diff(peaks)
    if y[-1]-z[-1]>.001 and periods[-1]>1. and abs(periods[-1]-periods[-2])<max(.3,.03*periods[-1]) and abs(y[-1]-y[-2])<.005:
     targets.append(dict(I=I,Cmin=float(z[-1]),Cmax=float(y[-1]),period=float(periods[-1]),initial=np.asarray(initial).tolist(),peaks=len(peaks)));break
  print(f'LR K={a.K:.5f} I={I:.6f} peaks={peak_count} accepted={bool(targets and targets[-1]["I"]==I)}',flush=True)
 if len(targets)<2:
  result=dict(K=a.K,status='reference_cycles_unresolved',rest_target=model.rest.tolist(),LR_lower_hopf_C_I=lr_lower,LR_upper_hopf_C_I=[float(model.cuts[1]),float(model.events[1])],lower_transition_fixed_to_lr=fixed_lower,reference_cycles=targets,reason='Fewer than two resolved finite-amplitude cycles in finite scans from the chosen initial conditions; this does not prove oscillations absent.')
  (a.out/'parameters.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print('SAVED',a.out/'parameters.json',result['status'],flush=True);return
 lib=ctypes.CDLL(str(a.engine.resolve()));ptr=ctypes.POINTER(ctypes.c_double)
 lib.simulate.argtypes=[ptr,ptr,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_int,ptr];lib.simulate.restype=ctypes.c_int
 steps=round(a.duration/a.dt)
 cgrid=np.linspace(max(.01,model.rest[0]*.5),max(r['Cmax'] for r in targets)*1.05,30);igrid=np.linspace(scan_low,model.events[1],7);hgrid=np.linspace(.35,.9,5);C,I,H=np.meshgrid(cgrid,igrid,hgrid,indexing='ij');reference=np.moveaxis(lr_rhs(C,H,I,a.K),0,-1)
 seedname='fm' if a.K<.07 else 'am';seed=json.loads((ROOT/f'astrocyte_lab/models/{seedname}_2thr_traj_identity_20261029/parameters.json').read_text());base=np.array(seed['parameters'],float);active=model.active
 if seedname=='am':base[2]=np.log(expit(base[2]))
 else:
  previous=PhysicalModel('fm','identity').unpack(base);base[2]=np.log(-previous['T'][0]/(2*np.sqrt(previous['D'][0])))
 lo=model.lo[active]+1e-7;hi=model.hi_bounds[active]-1e-7
 if not fixed_lower:
  f=(model.cuts[0]-model.rest[0])/(model.cuts[1]-model.rest[0]);g=(model.events[0]-.01)/(model.events[1]-.01);base_extra=np.array([logit(np.clip(f,.01,.99)),logit(np.clip(g,.01,.99))]);lo=np.r_[lo,-5.,-5.];hi=np.r_[hi,5.,5.]
 else:base_extra=np.array([])
 rng=np.random.default_rng(a.seed);runs=[];best=None;count=0;start=time.monotonic()
 def evaluate(v,report=False):
  nonlocal count
  count+=1;x=base.copy();x[active]=v[:len(active)]
  if not fixed_lower:
   model.cuts[0]=model.rest[0]+expit(v[-2])*(model.cuts[1]-model.rest[0]);model.events[0]=.01+expit(v[-1])*(model.events[1]-.01)
  cuts=np.ascontiguousarray(model.cuts,dtype=np.float64);m=model.unpack(x);co=np.ascontiguousarray(m['co'],dtype=np.float64); B=model.basis(C,I,m)
  pred=np.stack([m['a']*(H-B@m['N']),m['b']*(B@m['H']-H)],axis=-1);field=((pred-reference)/[.5,.08]).ravel()/np.sqrt(C.size)
  q=model.basis(model.cuts,model.events,m)@m['N']; domain=100*np.r_[np.maximum(1e-6-q,0),np.maximum(q-(1-1e-6),0)];rr=[];actual=[]
  for target in targets:
   y=np.zeros(6,dtype=np.float64);ok=lib.simulate(co.ctypes.data_as(ptr),cuts.ctypes.data_as(ptr),target['I'],*target['initial'],a.dt,steps,y.ctypes.data_as(ptr));good=bool(ok and y[3]>=3 and y[1]-y[0]>1e-4)
   rr.extend([(y[0]-target['Cmin'])/.03,(y[1]-target['Cmax'])/.03,(y[2]-target['period'])/1.5] if good else [10.,10.,10.]);actual.append(dict(I=target['I'],valid=good,Cmin=float(y[0]) if good else None,Cmax=float(y[1]) if good else None,period=float(y[2]) if good else None))
  res=np.r_[field,np.asarray(rr)/np.sqrt(len(targets)),domain]
  if count==1 or count%200==0:print(f'FIT K={a.K:.5f} eval={count} loss={res@res:.6g} elapsed={time.monotonic()-start:.1f}s',flush=True)
  return (res,actual) if report else res
 for j in range(a.starts):
  initial0=np.r_[base[active],base_extra];initial=np.clip(initial0+(rng.normal(0,.15,len(initial0)) if j else 0),lo+1e-8,hi-1e-8)
  sol=least_squares(evaluate,initial,bounds=(lo,hi),max_nfev=a.max_nfev,x_scale='jac',diff_step=2e-4,ftol=1e-7,xtol=1e-7,gtol=1e-7)
  x=base.copy();x[active]=sol.x[:len(active)];res,actual=evaluate(sol.x,True);loss=float(res@res);valid=model.physically_valid_events(x) and all(r['valid'] for r in actual)
  record=dict(start=j,loss=loss,valid=valid,nfev=sol.nfev,success=bool(sol.success),message=str(sol.message),optimality=float(sol.optimality));runs.append(record);print('START_DONE',json.dumps(record),flush=True)
  if valid and (best is None or loss<best[0]):best=(loss,x,actual,model.cuts.copy(),model.events.copy())
 result=dict(K=a.K,status='no_valid_fit',rest_target=model.rest.tolist(),lower_transition_fixed_to_lr=fixed_lower,LR_lower_hopf_C_I=lr_lower,reference_cycles=targets,runs=runs,scope='LR upper Hopf and rest fixed; lower Hopf fixed where ordered, otherwise effective first threshold and input fitted. Finite forward trajectories; no fold or unstable-cycle constraint and no global completeness check.')
 if best:
  loss,x,actual,best_cuts,best_events=best;model.cuts=best_cuts;model.events=best_events;m=model.unpack(x);result.update(physical_thresholds_C=best_cuts.tolist(),fixed_equilibrium_threshold_I=best_events.tolist(),status='fit_candidate',case='AM',observable_calcium='physical C',gate_coordinate='identity',gate_power=1.,parameters=x.tolist(),a=m['a'],b=m['b'],alpha=m['N'].tolist(),beta=m['H'].tolist(),trace=m['T'].tolist(),determinant=m['D'].tolist(),loss=loss,pwa_training_cycles=actual)
  validation=[]
  for target,fast in zip(targets,actual):
   def pwa(t,y):
    B=model.basis(y[0],target['I'],m);return np.array([m['a']*(y[1]-B@m['N']),m['b']*(B@m['H']-y[1])])
   def peak(t,y):return pwa(t,y)[0]
   peak.direction=-1;peak.terminal=False
   def trough(t,y):return pwa(t,y)[0]
   trough.direction=1;trough.terminal=False
   s=solve_ivp(pwa,(0,a.duration),target['initial'],method='DOP853',events=[peak,trough],rtol=2e-9,atol=2e-11,max_step=.1)
   mask=s.t_events[0]>a.duration*.5;mt=s.t_events[1]>a.duration*.5
   if s.success and sum(mask)>=3 and sum(mt)>=2:
    yy=s.y_events[0][mask,0];zz=s.y_events[1][mt,0];tt=s.t_events[0][mask];precise=dict(Cmin=float(zz[-1]),Cmax=float(yy[-1]),period=float(np.mean(np.diff(tt))))
    diff={k:abs(precise[k]-fast[k]) for k in precise};passed=diff['Cmin']<.003 and diff['Cmax']<.003 and diff['period']<.15
   else:precise=None;diff=None;passed=False
   validation.append(dict(I=target['I'],passed=passed,independent=precise,absolute_difference=diff));print(f'VALIDATE K={a.K:.5f} I={target["I"]:.6f} passed={passed}',flush=True)
  result['independent_validation']=validation
  if not all(v['passed'] for v in validation):result['status']='validation_failed'
 (a.out/'parameters.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print('SAVED',a.out/'parameters.json',result['status'],flush=True)
 if best:
  import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
  fig,ax=plt.subplots(1,2,figsize=(10,4),layout='constrained');Is=[r['I'] for r in targets]
  for key in ('Cmin','Cmax'):ax[0].plot(Is,[t[key] for t in targets],'o-',label='LR '+key);ax[0].plot(Is,[t[key] for t in actual],'x--',label='PWA '+key)
  ax[1].plot(Is,[t['period'] for t in targets],'o-',label='LR');ax[1].plot(Is,[t['period'] for t in actual],'x--',label='PWA');ax[0].set_ylabel('C (µM)');ax[1].set_ylabel('Period (s)')
  for axes in ax:axes.set_xlabel('I (µM)');axes.grid(alpha=.25);axes.legend()
  fig.suptitle(f'K_ER={a.K:.5f}: finite training trajectories');fig.savefig(a.out/'comparison.png',dpi=150)
if __name__=='__main__':main()
