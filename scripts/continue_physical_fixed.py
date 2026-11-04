"""Finite pseudo-arclength continuation of fixed-fold physical PWA orbits."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares
from scipy.integrate import solve_ivp

ROOT=Path(__file__).resolve().parents[1]
SCALE=np.array([10.,10.,10.,10.,.1])
REGIONS=(1,2,1,0);BOUND=(0,1,1,0,0)


class Orbit:
 def __init__(self,d):
  self.d=d;self.cuts=np.array(d['physical_thresholds_C']);self.N=np.array(d['alpha']);self.H=np.array(d['beta']);self.a=d['a'];self.b=d['b'];self.T=np.array(d['trace'])
 def match(self,t,I,details=False):
  heights=[];ends=[];mats=[];z=self.cuts[list(BOUND)]
  for j,r in enumerate(REGIONS):
   n=self.N[1]+sum(self.N[3:3+r]);h=self.H[1]+sum(self.H[3:3+r]);ni=self.N[0]+self.N[2]*I-self.N[3:3+r]@self.cuts[:r];hi=self.H[0]+self.H[2]*I-self.H[3:3+r]@self.cuts[:r]
   A=np.array([[-self.a*n,self.a,-self.a*ni],[self.b*h,-self.b,self.b*hi],[0.,0.,0.]])
   Q=expm(A*t[j]);den=Q[0,1]
   if abs(den)<1e-12:raise ValueError('singular segment')
   q=(z[j+1]-Q[0,0]*z[j]-Q[0,2])/den
   heights.append(q);ends.append(Q[1,0]*z[j]+Q[1,1]*q+Q[1,2]);mats.append(A)
  residual=np.array(ends)-np.roll(heights,-1)
  return (residual,z,np.array(heights),mats) if details else residual
 def rhs(self,t,y,I):
  c,q=y[:2];v=np.array([1,c,I,*np.maximum(c-self.cuts,0)]);r=np.searchsorted(self.cuts,c,side='right')
  return [self.a*(q-self.N@v),self.b*(self.H@v-q),self.T[r]]
 def point(self,t,I,rtol,require_physical=True):
  residual,z,q,mats=self.match(t,I,True);states=[];speeds=[]
  for j,(r,A) in enumerate(zip(REGIONS,mats)):
   y=np.array([z[j],q[j],1.]);speed=float((A@y)[0]);speeds.append(speed)
   if (1 if j<2 else -1)*speed<1e-9:raise ValueError('grazing or crossing direction')
   s=np.array([expm(A*x)@y for x in np.linspace(0,t[j],101)]);lo=-np.inf if r==0 else self.cuts[r-1];hi=np.inf if r==2 else self.cuts[r]
   if np.any(s[:,0]<lo-1e-8) or np.any(s[:,0]>hi+1e-8):raise ValueError('sampled itinerary violation')
   states.extend(s[:,0])
  physical=bool(min(states)>0 and max(states)<2/1.185)
  if require_physical and not physical:raise ValueError('outside physical C')
  logm=float(self.T[list(REGIONS)]@t);period=float(sum(t));direction=-1 if logm>0 else 1;initial=np.array([z[0],q[0],0.]);ivp=solve_ivp(lambda tt,y:self.rhs(tt,y,I),(0,direction*period),initial,method='DOP853',rtol=rtol,atol=rtol*.01,max_step=.05)
  closure=float(np.linalg.norm(ivp.y[:2,-1]-initial[:2],ord=np.inf))
  if not ivp.success or closure>1e-5 or abs(ivp.y[2,-1]-direction*logm)>1e-4:raise ValueError('independent integration unresolved')
  return dict(I=float(I),times=t.tolist(),period=period,Cmin=float(min(states)),Cmax=float(max(states)),physical=physical,logM=logm,stable=bool(logm<0),matching=float(max(abs(residual))),closure=closure,crossing_speeds=speeds)


def continue_branch(orb,seed,sign,steps,require_physical=True,max_I=1.35,initial_step=.01,max_step=.03):
 def fun(w):
  try:
   p=w*SCALE;return orb.match(p[:4],p[4])
  except (ValueError,OverflowError,FloatingPointError):return np.full(4,1e5)
 def jac(w):
  eps=1e-5;return np.column_stack([(fun(w+eps*v)-fun(w-eps*v))/(2*eps) for v in np.eye(5)])
 w=np.array(seed)/SCALE; tangent=np.linalg.svd(jac(w),full_matrices=True)[2][-1]*sign;out=[];step=initial_step;reason='step limit'
 lower=np.r_[np.full(4,1e-7)/SCALE[:4],0.];upper=np.r_[np.full(4,120.)/SCALE[:4],max_I/SCALE[4]]
 for _ in range(steps):
  predicted=w+step*tangent
  if np.any(predicted<=lower) or np.any(predicted>=upper):
   step*=.5
   if step<2e-5:reason='bounds';break
   continue
  def objective(x):return np.r_[fun(x),np.dot(x-predicted,tangent)]
  sol=least_squares(objective,predicted,bounds=(lower,upper),max_nfev=90,ftol=1e-10,xtol=1e-10,gtol=1e-10)
  if max(abs(objective(sol.x)))>1e-8:
   step*=.5
   if step<2e-5:reason='corrector unresolved';break
   continue
  w1=sol.x;v=w1*SCALE
  try:out.append(orb.point(v[:4],v[4],2e-9,require_physical=require_physical))
  except ValueError as exc:reason=str(exc);break
  tan=np.linalg.svd(jac(w1),full_matrices=True)[2][-1]
  if tan@tangent<0:tan=-tan
  w,tangent=w1,tan;step=min(step*1.05,max_step)
 return {'direction':sign,'reason':reason,'points':out}


def nondegeneracy(orb,seed):
 t=np.array(seed[:4]);I=seed[4];eps=1e-5;eye=np.eye(4);f=lambda v:orb.match(v,I)
 J=np.column_stack([(f(t+eps*e)-f(t-eps*e))/(2*eps) for e in eye]);U,S,Vh=np.linalg.svd(J);left=U[:,-1];right=Vh[-1];base=f(t)
 curv=[float(left@(f(t+s*right)-2*base+f(t-s*right))/s**2) for s in (1e-3,3e-4,1e-4)]
 deriv=float(left@(orb.match(t,I+eps)-orb.match(t,I-eps))/(2*eps))
 return dict(singular_values=S.tolist(),curvature_samples=curv,input_derivative=deriv,scope='finite differences; scaling-dependent numerical diagnostic')


def main():
 p=argparse.ArgumentParser();p.add_argument('--models',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=20);args=p.parse_args();results={}
 for case in ('am','afm','fm'):
  path=args.models/f'{case}_physical/parameters.json';d=json.loads(path.read_text());orb=Orbit(d);constraint=d['fold_constraint'];seed=np.array([*constraint['crossing_times'],constraint['I']]);neutral=orb.point(seed[:4],seed[4],2e-9)
  branches=[continue_branch(orb,seed,sign,args.steps) for sign in (-1,1)]
  results[case]=dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),neutral=neutral,nondegeneracy=nondegeneracy(orb,seed),branches=branches)
  print(case,'neutral',neutral['I'],'logM',neutral['logM'],'branches',[(len(b['points']),b['reason']) for b in branches],flush=True)
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps({'scope':'fixed-fold models; finite continuation, both-threshold itinerary; sampled checks not a proof of first crossing or global completeness','results':results},indent=2)+'\n')


if __name__=='__main__':main()
