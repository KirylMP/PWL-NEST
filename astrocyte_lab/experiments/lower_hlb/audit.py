#!/usr/bin/env python3
"""Local two-region boundary return matching near the lower crossing."""
import argparse,concurrent.futures,hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'astrocyte_lab/experiments/mixed_three_fits'))
sys.path.insert(0,str(ROOT/'astrocyte_lab/experiments/physical_no_fold'))
sys.path.insert(0,str(ROOT/'experiments/fm_am_minimal_zones/fm_four_zone_refinement'))
from mixed_model import MixedModel, inverse as mixed_inverse
from physical_model import PhysicalModel, inverse as physical_inverse
from fold import BoundaryFold

def load(path):
 data=json.loads(Path(path).read_text());physical=data.get('observable_calcium')=='physical C';model=PhysicalModel(data['K']) if physical else MixedModel(data['K']);x=np.zeros(model.size);raw=np.asarray(data['parameters'],float);x[:min(len(raw),model.size)]=raw[:model.size];m=model.unpack(x);return data,model,x,m,physical_inverse if physical else mixed_inverse

def local_info(path):
 d,model,x,m,_=load(path);eigs=[]
 for region in (0,1):
  T=float(m['T'][region]);D=float(m['D'][region]);disc=T*T-4*D;omega=float(np.sqrt(-disc)/2) if disc<0 else None
  eigs.append(dict(trace=T,determinant=D,discriminant=disc,type=('stable' if T<0 else 'unstable')+(' focus' if disc<0 else ' node' if D>0 and disc>0 else ' other'),omega=omega))
 pair=all(e['omega'] is not None for e in eigs) and eigs[0]['trace']<0 and eigs[1]['trace']>0;alpha=sum(e['trace']/(2*e['omega']) for e in eigs) if pair else None
 return dict(path=str(path),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),K=d['K'],threshold_C=float(model.cuts[0]),I_cross=float(model.events[0]),region_types=eigs,hlb1_focus_pair=pair,alpha=alpha,limiting_return_multiplier=float(np.exp(np.pi*alpha)) if pair else None,hlb1_criticality=('subcritical_unstable_cycle' if alpha>0 else 'supercritical_stable_cycle' if alpha<0 else 'degenerate') if pair else 'focus_focus_assumption_not_met')

def solve(task):
 path,delta,seed=task;d,model,x,m,inv=load(path);I=float(model.events[0]+delta);fold=BoundaryFold(model,I,first=0,last=0);n=2
 rng=np.random.default_rng(seed);eig=[]
 for r in (0,1):
  disc=m['T'][r]**2-4*m['D'][r];eig.append(np.pi/(np.sqrt(-disc)/2) if disc<0 else 3.)
 times=np.clip(np.array([eig[1],eig[0]])*np.exp(rng.normal(0,0.9,2)),.02,90);qscale=max(1.,max(abs(float(v)) for v in d.get('event_q',[1.])))
 def fun(v):
  try:
   f=fold.matching(x,v)/qscale;return f if np.isfinite(f).all() else np.full(n,1e4)
  except (ValueError,OverflowError,np.linalg.LinAlgError,FloatingPointError):return np.full(n,1e4)
 sol=least_squares(fun,times,bounds=(np.full(n,1e-5),np.full(n,120.)),max_nfev=350,xtol=1e-11,ftol=1e-11,gtol=1e-11,x_scale='jac');err=fun(sol.x)
 if max(abs(err))>1e-8:return None
 try:
  residual,_,z,q,mat=fold.matching(x,sol.x,True);states=[];speeds=[]
  for j,(r,M) in enumerate(zip(fold.regions,mat)):
   y=np.r_[z[j],q[j],1.];speed=float((M@y)[0]);speeds.append(speed)
   if (1 if j==0 else -1)*speed<=1e-10:return None
   ss=np.array([(expm(M*t)@y)[:2] for t in np.linspace(0,sol.x[j],101)])
   lo=-np.inf if r==0 else m['th'][r-1];hi=np.inf if r==len(m['th']) else m['th'][r]
   if not np.isfinite(ss).all() or min(ss[:,0])<lo-1e-8 or max(ss[:,0])>hi+1e-8:return None
   states.extend(ss)
  states=np.asarray(states);c=inv(states[:,0],m['shape']);amp=float(np.max(c)-np.min(c))
  if amp<1e-8:return None
  logM=float(sum(m['T'][r]*t for r,t in zip(fold.regions,sol.x)))
  def rhs(_,y):
   region=int(np.searchsorted(m['th'],y[0],side='right'));field=m['co']@np.r_[1.,y[0],y[1],I,np.maximum(y[0]-m['padded'],0.)];return [field[0],field[1],m['T'][region]]
  y0=np.array([z[0],q[0],0.]);ivp=solve_ivp(rhs,(0.,float(sum(sol.x))),y0,method='DOP853',rtol=1e-10,atol=1e-12,max_step=.04)
  if not ivp.success:return None
  closure=float(np.max(np.abs(ivp.y[:2,-1]-y0[:2])/np.maximum(1.,abs(y0[:2]))));log_error=float(abs(ivp.y[2,-1]-logM))
  if closure>1e-6 or log_error>1e-5:return None
  return dict(ivp_closure=closure,ivp_log_multiplier=float(ivp.y[2,-1]),I=I,delta_I=delta,Cmin=float(min(c)),Cmax=float(max(c)),amplitude=amp,period=float(sum(sol.x)),log_multiplier=logM,stable=logM<0,matching_error=float(max(abs(residual))),crossing_speeds=speeds,scope='numerical threshold matching, sampled itinerary, independent DOP853 closure and multiplier check')
 except (ValueError,OverflowError,FloatingPointError,np.linalg.LinAlgError):return None

def main():
 p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path);p.add_argument('--physical-run',type=Path);p.add_argument('--candidate',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--starts',type=int,default=8);p.add_argument('--workers',type=int,default=16);p.add_argument('--deltas',type=float,nargs='+',default=[.0001,.001,.005,.01]);a=p.parse_args()
 if a.starts<1 or a.workers<1 or min(a.deltas)<=0:raise ValueError('Invalid search settings')
 if a.candidate:paths={'candidate':a.candidate.resolve()}
 else:
  if not a.snapshot or not a.physical_run:raise ValueError('Provide --snapshot and --physical-run')
  paths={}
  for mode in ('am','afm','fm'):
   paths[mode+'_physical']=(a.physical_run/mode/'comparison/new_rank_01/parameters.json').resolve();paths[mode+'_mixed']=(a.snapshot/f'{mode}_mixed/parameters.json').resolve()
 result={'scope':'Local one-cut search only; finite starting values and trajectory sampling. No general absence proof; found cycles checked by independent DOP853. HLB1 is focus-focus only; q has no physical bounds.','models':{}}
 for name,path in paths.items():
  info=local_info(path);tasks=[(path,sign*delta,4100+j) for delta in a.deltas for sign in (-1,1) for j in range(a.starts)]
  print('MODEL',name,json.dumps(info),flush=True)
  with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:raw=list(pool.map(solve,tasks))
  orbits=[]
  for r in raw:
   if r is None:continue
   if any(abs(r['I']-q['I'])<1e-10 and abs(r['amplitude']-q['amplitude'])<1e-6 and abs(r['period']-q['period'])<1e-5 for q in orbits):continue
   orbits.append(r);print('ORBIT',name,json.dumps(r),flush=True)
  result['models'][name]={'local':info,'cycles':sorted(orbits,key=lambda r:r['I'])};a.out.mkdir(parents=True,exist_ok=True);(a.out/'lower_hlb.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print('SAVED',a.out/'lower_hlb.json',flush=True)
if __name__=='__main__':main()
