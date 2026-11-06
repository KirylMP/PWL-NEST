"""Neutral-cycle search on the upper-only and both-threshold itineraries."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp

ROOT=Path(__file__).resolve().parents[1]


class Route:
 def __init__(self,d,route):
  self.d=d;self.cut=np.array(d['physical_thresholds_C']);self.N=np.array(d['alpha']);self.H=np.array(d['beta']);self.a=d['a'];self.b=d['b'];self.T=np.array(d['trace'])
  self.regions=(2,1) if route=='upper' else (1,2,1,0)
  self.boundaries=(1,1,1) if route=='upper' else (0,1,1,0,0)
 def match(self,t,I,details=False):
  z=self.cut[list(self.boundaries)];starts=[];ends=[];mats=[]
  for j,r in enumerate(self.regions):
   n=self.N[1]+sum(self.N[3:3+r]);h=self.H[1]+sum(self.H[3:3+r]);nc=self.N[0]+self.N[2]*I-self.N[3:3+r]@self.cut[:r];hc=self.H[0]+self.H[2]*I-self.H[3:3+r]@self.cut[:r]
   A=np.array([[-self.a*n,self.a,-self.a*nc],[self.b*h,-self.b,self.b*hc],[0.,0.,0.]]);Q=expm(A*t[j]);den=Q[0,1]
   if abs(den)<1e-12:raise ValueError('singular segment')
   q=(z[j+1]-Q[0,0]*z[j]-Q[0,2])/den;starts.append(q);ends.append(Q[1,0]*z[j]+Q[1,1]*q+Q[1,2]);mats.append(A)
  residual=np.array(ends)-np.roll(starts,-1)
  return (residual,z,np.array(starts),mats) if details else residual
 def inspect(self,t,I):
  residual,z,q,mats=self.match(t,I,True);values=[];speeds=[]
  for j,(r,A) in enumerate(zip(self.regions,mats)):
   y=np.array([z[j],q[j],1.]);speed=float((A@y)[0]);speeds.append(speed)
   if (1 if j<len(t)//2 else -1)*speed<=1e-8:raise ValueError('wrong direction or grazing')
   states=np.array([expm(A*s)@y for s in np.linspace(0,t[j],201)]);lo=-np.inf if r==0 else self.cut[r-1];hi=np.inf if r==2 else self.cut[r]
   if np.any(states[:,0]<lo-1e-8) or np.any(states[:,0]>hi+1e-8):raise ValueError('sampled itinerary violation')
   values.extend(states[:,0])
  if min(values)<=0 or max(values)>=2/1.185 or max(values)-min(values)<1e-4:raise ValueError('outside physical domain or trivial')
  def rhs(_,y):
   C,Q=y[:2];v=np.array([1,C,I,*np.maximum(C-self.cut,0)]);j=np.searchsorted(self.cut,C,side='right');return [self.a*(Q-self.N@v),self.b*(self.H@v-Q),self.T[j]]
  initial=np.array([z[0],q[0],0.]);logm=float(self.T[list(self.regions)]@t);ivp=solve_ivp(rhs,(0,sum(t)),initial,method='DOP853',rtol=1e-10,atol=1e-12,max_step=.025)
  closure=float(np.max(np.abs(ivp.y[:2,-1]-initial[:2])));ind_log=float(ivp.y[2,-1])
  if not ivp.success or closure>1e-5 or abs(ind_log-logm)>1e-4:raise ValueError('independent IVP unresolved')
  eps=1e-5;n=len(t);eye=np.eye(n);f=lambda v:self.match(v,I);J=np.column_stack([(f(t+eps*e)-f(t-eps*e))/(2*eps) for e in eye]);U,S,Vh=np.linalg.svd(J);left=U[:,-1];right=Vh[-1];base=f(t)
  curv=[float(left@(f(t+s*right)-2*base+f(t-s*right))/s**2) for s in (1e-3,3e-4,1e-4)];di=float(left@(self.match(t,I+eps)-self.match(t,I-eps))/(2*eps))
  return dict(I=float(I),times=t.tolist(),period=float(sum(t)),Cmin=float(min(values)),Cmax=float(max(values)),matching_error=float(max(abs(residual))),logM=logm,closure=closure,independent_logM=ind_log,crossing_speeds=speeds,singular_values=S.tolist(),curvature_samples=curv,input_derivative=di)


def main():
 p=argparse.ArgumentParser();p.add_argument('--free',type=Path,required=True);p.add_argument('--fixed',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--starts',type=int,default=16);p.add_argument('--cases',nargs='+',choices=('AM','AFM','FM'),default=('AM','AFM','FM'));a=p.parse_args();rng=np.random.default_rng(20261104)
 mixed=json.loads((ROOT/'paper/data/two_threshold_models/sources/mixed_fold_audit.json').read_text())['results']; all_results={}
 for case in a.cases:
  path=a.free/case/'parameters.json';d=json.loads(path.read_text());fixed=json.loads((a.fixed/(case.lower()+'_physical')/'parameters.json').read_text());old=next((v for v in mixed if v.get('model')==case.lower() and v.get('status')=='candidate_fold' and v.get('itinerary')=='both_thresholds'),None)
  all_results[case]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'routes':{}}
  for route in ('upper','both'):
   orb=Route(d,route);n=len(orb.regions);base=np.r_[fixed['fold_constraint']['crossing_times'],fixed['fold_constraint']['I']];seeds=[np.r_[base[1],base[2],base[-1]] if route=='upper' else base]
   if old:seeds.append(np.r_[old['times'][1],old['times'][2],old['I']] if route=='upper' else np.r_[old['times'],old['I']])
   def fun(w):
    try:
     out=np.r_[orb.match(w[:n],w[n]),orb.T[list(orb.regions)]@w[:n]/10.];return out if np.isfinite(out).all() else np.full(n+1,1e4)
    except (ValueError,OverflowError,FloatingPointError):return np.full(n+1,1e4)
   attempts=[];found=[]
   for k in range(a.starts):
    seed=seeds[k%len(seeds)].copy()
    if k>=len(seeds):seed[:n]*=np.exp(rng.normal(0,.8,n));seed[n]+=rng.normal(0,.3)
    seed=np.clip(seed,[.001]*n+[.001],[80.]*n+[1.8]);sol=least_squares(fun,seed,bounds=([1e-5]*n+[.0001],[120.]*n+[2.]),max_nfev=300,ftol=1e-10,xtol=1e-10,gtol=1e-10,x_scale='jac');res=float(max(abs(fun(sol.x))));entry={'start':k,'residual':res,'I':float(sol.x[-1]),'status':'unresolved'}
    if res<1e-7:
     try:
      result=orb.inspect(sol.x[:n],sol.x[n]);entry['status']='validated_neutral';entry['result']=result
      if not any(abs(x['I']-result['I'])<1e-6 for x in found):found.append(result)
     except ValueError as e:entry['status']='rejected';entry['reason']=str(e)
    attempts.append(entry)
   all_results[case]['routes'][route]={'attempts':attempts,'found':found};print(case,route,'validated',[(x['I'],x['period']) for x in found],'starts',len(attempts),flush=True)
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps({'scope':'finite multistart; sampled itinerary and independent IVP; no proof of global completeness','results':all_results},indent=2)+'\n')


if __name__=='__main__':main()
