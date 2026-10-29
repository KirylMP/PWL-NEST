#!/usr/bin/env python3
"""Search neutral cycles of fixed mixed-coordinate models by affine matching."""
import argparse,concurrent.futures,json,sys
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'astrocyte_lab/experiments/mixed_three_fits'))
sys.path.insert(0,str(ROOT/'experiments/fm_am_minimal_zones/fm_four_zone_refinement'))
from mixed_model import MixedModel,inverse,phi
from fold import BoundaryFold

def solve(task):
 name,params,ref,first,number,seed=task
 model=MixedModel(params['K']);x=np.asarray(params['parameters'],float);m=model.unpack(x);target=ref['I'];fold=BoundaryFold(model,target,first=first,last=1);n=fold.legs
 rng=np.random.default_rng(seed);refperiod=ref['period'];t=np.full(n,refperiod/n)*np.exp(rng.normal(0,.65,n));initial=np.r_[np.clip(t,.05,59),np.clip(target+rng.normal(0,.08),.01,1.99)]
 qscale=max(1.,max(abs(float(v)) for v in params.get('event_q',[1.])))
 def equations(v):
  try:
   fold.I=v[-1];res=fold.matching(x,v[:n]);val=np.r_[res/qscale,sum(m['T'][r]*t for r,t in zip(fold.regions,v[:n]))/10.]
   return val if np.isfinite(val).all() else np.full(n+1,1e5)
  except (ValueError,OverflowError,np.linalg.LinAlgError,FloatingPointError):return np.full(n+1,1e5)
 sol=least_squares(equations,initial,bounds=(np.r_[np.full(n,.001),.01],np.r_[np.full(n,60.),2.]),max_nfev=600,ftol=1e-10,xtol=1e-10,gtol=1e-10,x_scale='jac')
 v=sol.x;res=equations(v);record={'model':name,'itinerary':'both_thresholds' if first==0 else 'upper_threshold_only','start':number,'I':float(v[-1]),'times':v[:n].tolist(),'residual':float(max(abs(res))),'nfev':int(sol.nfev),'status':'unresolved'}
 if max(abs(res))>1e-7:return record
 try:
  fold.I=v[-1];rr,_,z,q,mat=fold.matching(x,v[:n],True);states=[];speeds=[];valid=True;reasons=[]
  for j,(r,M) in enumerate(zip(fold.regions,mat)):
   y=np.r_[z[j],q[j],1.];speed=float((M@y)[0]);speeds.append(speed)
   if (1 if j<n//2 else -1)*speed<=1e-8:valid=False;reasons.append('wrong or grazing threshold crossing')
   samples=np.array([expm(M*t)@y for t in np.linspace(0,v[j],101)]);low=-np.inf if r==0 else m['th'][r-1];high=np.inf if r>=len(m['th']) else m['th'][r]
   if min(samples[:,0])<low-1e-7 or max(samples[:,0])>high+1e-7:valid=False;reasons.append('sampled trajectory outside itinerary')
   states.extend(samples[:,:2])
  states=np.asarray(states);zlo=float(phi(1e-12,m['shape']));zhi=float(phi(model.CMAX,m['shape']))
  if np.min(states[:,0])<=zlo or np.max(states[:,0])>=zhi:valid=False;reasons.append('outside physical C domain')
  if valid:
   c=inverse(states[:,0],m['shape']);cmin=float(np.min(c));cmax=float(np.max(c))
   if cmax-cmin<.001:valid=False;reasons.append('amplitude too small')
  else:cmin=cmax=None
  # Local fold tests: matching Jacobian rank and nonzero quadratic and input terms.
  dt=1e-4;tv=v[:n];eye=np.eye(n);fold.I=v[-1]
  J=np.column_stack([(fold.matching(x,tv+dt*e)-fold.matching(x,tv-dt*e))/(2*dt) for e in eye]);U,S,V=np.linalg.svd(J);left=U[:,-1];right=V[-1];base=fold.matching(x,tv);curv=float(left@(fold.matching(x,tv+dt*right)-2*base+fold.matching(x,tv-dt*right))/dt**2)
  fold.I=v[-1]+dt;plus=fold.matching(x,tv);fold.I=v[-1]-dt;minus=fold.matching(x,tv);unfold=float(left@(plus-minus)/(2*dt))
  nondegenerate=bool(S[-1]<max(1e-7,1e-5*S[0]) and (n==1 or S[-2]>1e-8*max(S[0],1e-12)) and abs(curv)>1e-7 and abs(unfold)>1e-7)
  ivp_closure=None;ivp_log_multiplier=None
  if valid and nondegenerate:
   def rhs(_,state):
    zz,qq=state[:2];region=int(np.searchsorted(m['th'],zz,side='right'));field=m['co']@np.r_[1.,zz,qq,v[-1],np.maximum(zz-m['padded'],0.)];return [field[0],field[1],m['T'][region]]
   y0=[float(z[0]),float(q[0]),0.];ivp=solve_ivp(rhs,(0,float(sum(tv))),y0,method='DOP853',rtol=1e-9,atol=1e-11,max_step=.04)
   if ivp.success:
    ivp_closure=float(np.linalg.norm((ivp.y[:2,-1]-y0[:2])/np.array([1.,qscale]),ord=np.inf));ivp_log_multiplier=float(ivp.y[2,-1])
   if ivp_closure is None or ivp_closure>1e-5 or abs(ivp_log_multiplier)>1e-4:nondegenerate=False;reasons.append('independent IVP closure or multiplier not confirmed')
  record.update(ivp_closure=ivp_closure,ivp_log_multiplier=ivp_log_multiplier,status=('candidate_fold' if valid and nondegenerate else ('neutral_degenerate_or_unverified' if valid else 'invalid_itinerary')),period=float(sum(tv)),Cmin=cmin,Cmax=cmax,log_multiplier=float(sum(m['T'][r]*t for r,t in zip(fold.regions,tv))),matching_error=float(max(abs(rr))),crossing_speeds=speeds,singular_values=S.tolist(),curvature=curv,input_derivative=unfold,nondegenerate=nondegenerate,reasons=reasons)
 except (ValueError,OverflowError,np.linalg.LinAlgError,FloatingPointError) as e:record.update(status='verification_error',error=str(e))
 return record

def main():
 p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--starts',type=int,default=24);p.add_argument('--workers',type=int,default=16);a=p.parse_args()
 if min(a.starts,a.workers)<1:raise ValueError('Invalid options')
 a.out.mkdir(parents=True,exist_ok=True);tasks=[];refs={}
 for name in ('am','afm','fm'):
  params=json.loads((a.snapshot/f'{name}_mixed/parameters.json').read_text());ref=json.loads((a.snapshot/f'{name}_physical/parameters.json').read_text())['reference_fold'];refs[name]=ref
  for first in (0,1):
   for n in range(a.starts):tasks.append((name,params,ref,first,n,5800+first*100+n))
 print(f'Fold search: {len(tasks)} starts, workers={a.workers}',flush=True)
 with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
  results=[]
  for r in pool.map(solve,tasks):results.append(r);print(json.dumps(r),flush=True)
 summary={'scope':'Finite multi-start affine boundary matching on one- and two-threshold itineraries. Candidate fold: numerical closure, M=1, sampled itinerary and local nondegeneracy; independent IVP closure and multiplier are checked; tolerance convergence and nearby stable/unstable branches remain to be checked. Failure does not prove absence. Original LR fold values are previously calculated estimates.','reference_LR':refs,'results':results}
 (a.out/'fold_audit.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
 for name in refs:
  r=[q for q in results if q['model']==name and q['status']=='candidate_fold'];print(name,'LR I=',refs[name]['I'],'candidates=',[(round(q['I'],8),q['itinerary']) for q in r],flush=True)
 print('SAVED',a.out/'fold_audit.json',flush=True)
if __name__=='__main__':main()
