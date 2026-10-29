#!/usr/bin/env python3
"""Joint physical-C PWA family: affine latent parameters in K, exact events."""
import argparse,ctypes,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'astrocyte_lab/experiments/high_k_family'))
from fit_high_k import configure
from fit_k import lr_rhs

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--starts',type=int,default=4);p.add_argument('--max-nfev',type=int,default=600);p.add_argument('--duration',type=float,default=350.);p.add_argument('--dt',type=float,default=.02);p.add_argument('--seed',type=int,default=8321);p.add_argument('--jitter',type=float,default=0.);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 if a.starts<1 or a.max_nfev<1 or a.duration<100 or a.dt<=0:raise ValueError('Invalid settings')
 rows=json.loads((HERE/'reference.json').read_text())['models'];models=[]
 for row in rows:
  model,fixed,_=configure(row['K'])
  if not fixed or not row['lower_transition_fixed_to_lr']:raise ValueError('Training reference must have both LR events fixed')
  models.append(model)
 ks=np.array([r['K'] for r in rows]);t=(ks-.0775)/.0325;known=np.array([r['parameters'][:8] for r in rows]);A=np.column_stack([np.ones(len(t)),t]);init=np.linalg.lstsq(A,known,rcond=None)[0].ravel()
 # Fit endpoint vectors rather than unconstrained slope: keeps parameters within bounds throughout the interval.
 endpoints=np.r_[init[:8]-init[8:],init[:8]+init[8:]];lo=models[0].lo[:8]+1e-7;hi=models[0].hi_bounds[:8]-1e-7;lower=np.tile(lo,2);upper=np.tile(hi,2);endpoints=np.clip(endpoints,lower+1e-6,upper-1e-6)
 engine=a.out/'trajectory_engine.so';subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(ROOT/'astrocyte_lab/scripts/physical_trajectory_engine.cpp'),'-o',str(engine)],check=True)
 lib=ctypes.CDLL(str(engine.resolve()));ptr=ctypes.POINTER(ctypes.c_double);lib.simulate.argtypes=[ptr,ptr,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_double,ctypes.c_int,ptr];lib.simulate.restype=ctypes.c_int;steps=round(a.duration/a.dt)
 inputs=[];hold=[]
 for i,row in enumerate(rows):
  for j,target in enumerate(row['reference_cycles']):
   (hold if j%3==1 else inputs).append((i,target))
 if not hold or not inputs:raise ValueError('Insufficient reference inputs for train/holdout split')
 def expand(v,i):
  left,right=v[:8],v[8:];u=(t[i]+1)/2;return np.r_[(1-u)*left+u*right,0.]
 def simulate(model,x,target):
  m=model.unpack(x);co=np.ascontiguousarray(m['co'],dtype=np.float64);cuts=np.ascontiguousarray(model.cuts,dtype=np.float64);o=np.zeros(6)
  ok=lib.simulate(co.ctypes.data_as(ptr),cuts.ctypes.data_as(ptr),target['I'],*target['initial'],a.dt,steps,o.ctypes.data_as(ptr));good=bool(ok and o[3]>=3 and o[1]-o[0]>1e-4)
  return good,o
 calls=0;best=float('inf');start=time.monotonic();checkpoint_time=0.
 def residual(v):
  nonlocal calls,best,checkpoint_time
  calls+=1;xs=[expand(v,i) for i in range(len(rows))];rr=[]
  for model,x in zip(models,xs):
   m=model.unpack(x);q=model.basis(model.cuts,model.events,m)@m['N'];rr.extend((30*np.maximum(1e-6-q,0)).tolist());rr.extend((30*np.maximum(q-(1-1e-6),0)).tolist())
  for i,r in inputs:
   good,o=simulate(models[i],xs[i],r)
   rr.extend([(o[0]-r['Cmin'])/.05,(o[1]-r['Cmax'])/.08,(o[2]-r['period'])/2.] if good else [8.,8.,8.])
  arr=np.r_[np.asarray(rr[:4*len(rows)])/np.sqrt(len(rows)),np.asarray(rr[4*len(rows):])/np.sqrt(len(inputs)),.005*(v-endpoints)]
  score=float(arr@arr)
  if score<best:
   best=score;now=time.monotonic()
   if now-checkpoint_time>=20 or calls==1:
    temp=a.out/'checkpoint.tmp';temp.write_text(json.dumps(dict(status='unverified_checkpoint',loss=score,eval=calls,left=v[:8].tolist(),right=v[8:].tolist()),indent=2));temp.replace(a.out/'checkpoint.json');checkpoint_time=now
  if calls==1 or calls%100==0:print(f'EVAL {calls} loss={score:.6g} best={best:.6g} elapsed={time.monotonic()-start:.1f}s',flush=True)
  return arr
 rng=np.random.default_rng(a.seed);solutions=[];attempts=[]
 for j in range(a.starts):
  jitter=a.jitter if j==0 else max(a.jitter,.06)
  v=np.clip(endpoints+rng.normal(0,jitter,len(endpoints)),lower+1e-8,upper-1e-8)
  sol=least_squares(residual,v,bounds=(lower,upper),max_nfev=a.max_nfev,x_scale='jac',diff_step=2e-4,ftol=1e-6,xtol=1e-6,gtol=1e-6)
  cost=float(sol.fun@sol.fun);attempts.append(dict(start=j,nfev=sol.nfev,success=bool(sol.success),message=str(sol.message),optimality=float(sol.optimality),loss=cost));solutions.append((cost,sol.x.copy()));print('START_DONE',json.dumps(attempts[-1]),flush=True)
 cost,v=min(solutions,key=lambda z:z[0]);train_report=[];hold_report=[];outputs=[]
 for i,row in enumerate(rows):
  x=expand(v,i);m=models[i].unpack(x);q=models[i].basis(models[i].cuts,models[i].events,m)@m['N'];outputs.append(dict(K=row['K'],physical_thresholds_C=models[i].cuts.tolist(),fixed_equilibrium_threshold_I=models[i].events.tolist(),rest_target=models[i].rest.tolist(),parameters=x.tolist(),alpha=m['N'].tolist(),beta=m['H'].tolist(),a=float(m['a']),b=float(m['b']),trace=m['T'].tolist(),determinant=m['D'].tolist(),event_q=q.tolist(),observable_calcium='physical C',gate_coordinate='identity',gate_power=1.))
 for group,dest in ((inputs,train_report),(hold,hold_report)):
  for i,r in group:
   valid,o=simulate(models[i],expand(v,i),r);dest.append(dict(K=rows[i]['K'],I=r['I'],valid=valid,LR=r,PWA=dict(Cmin=float(o[0]),Cmax=float(o[1]),period=float(o[2])) if valid else None))
 data=dict(status='candidate_unverified',scope='Joint linear endpoint parameter family; exact rest and both equilibrium crossings at each K; no fold or unstable cycle constraint. Train and input holdouts are finite RK4 trajectories. Independently verify before use.',K_range=[float(ks.min()),float(ks.max())],coordinate='free_x(K)=(1-u)*left+u*right; u=(K-.045)/.065; x[8]=0; remaining coefficients derived from rest/events at each K.',left=v[:8].tolist(),right=v[8:].tolist(),train_loss=cost,attempts=attempts,models=outputs,train=train_report,input_holdout=hold_report,settings=dict(duration=a.duration,dt=a.dt))
 (a.out/'family.json').write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
 # Plots show measured performance, not just parameter interpolation.
 fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
 for reports,label,style in ((train_report,'training','o'),(hold_report,'held-out I','x')):
  valid=[r for r in reports if r['valid']]
  for ax,key in zip(axes,('Cmin','Cmax','period')):
   ax.scatter([r['K'] for r in valid],[r['PWA'][key]-r['LR'][key] for r in valid],label=label,marker=style,s=18)
 for ax,label in zip(axes,('Cmin error (µM)','Cmax error (µM)','period error (s)')):ax.axhline(0,color='grey',lw=.8);ax.set_ylabel(label);ax.set_xlabel('K_ER (µM)');ax.grid(alpha=.25);ax.legend()
 fig.suptitle('Joint family: errors at fixed training K');fig.savefig(a.out/'trajectory_errors.png',dpi=150)
 print('SAVED',a.out/'family.json','training',len(inputs),'held_out_I',len(hold),flush=True)
if __name__=='__main__':main()
