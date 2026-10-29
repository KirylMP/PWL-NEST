#!/usr/bin/env python3
"""High-precision normalized boundary shooting for stable-node/unstable-focus HLB2."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import mpmath as mp
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'astrocyte_lab/experiments/lower_hlb'))
from audit import load

def check(path):
 data,model,x,m,inverse=load(path);T=np.asarray(m['T']);D=np.asarray(m['D']);disc=T*T-4*D
 report=dict(path=str(path),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),K=data['K'],I_cross=float(model.events[0]),physical_thresholds_C=model.cuts.tolist(),trace=T.tolist(),determinants=D.tolist(),discriminants=disc.tolist())
 if not (T[0]<0 and D[0]>0 and disc[0]>0 and T[1]>0 and D[1]>0 and disc[1]<0):return report|dict(status='not_node_focus')
 omega=np.sqrt(-disc[1])/2;focus_rate=T[1]/2;slow=(-T[0]-np.sqrt(disc[0]))/2;tf_est=np.pi/omega+1/focus_rate;tn_est=tf_est*focus_rate/slow
 digits=max(90,int(np.ceil(focus_rate*tf_est/np.log(10)))+75);report.update(estimated_focus_time_s=tf_est,estimated_node_time_s=tn_est,working_precision_digits=digits)
 if digits>3000 or not np.isfinite(tn_est) or tn_est>10000:return report|dict(status='precision_or_time_limit')
 mp.mp.dps=digits;A=[];b=[];Ainv=[]
 for region in (1,0):
  co=m['co'];matrix=mp.matrix([[mp.mpf(repr(float(co[i,1]+sum(co[i,4:4+region])))),mp.mpf(repr(float(co[i,2])))] for i in (0,1)]);inp=mp.matrix([mp.mpf(repr(float(co[i,3]))) for i in (0,1)]);A.append(matrix);b.append(inp);Ainv.append(matrix**-1)
 def matrixexp(i,t):
  M=A[i];tr=(M[0,0]+M[1,1])/2;w=mp.sqrt(((M[0,0]-M[1,1])/2)**2+M[0,1]*M[1,0]);E=mp.exp(tr*t)*(mp.cosh(w*t)*mp.eye(2)+(mp.sinh(w*t)/w)*(M-tr*mp.eye(2)));return E.apply(mp.re)
 def flow(i,t):
  E=matrixexp(i,t);g=Ainv[i]*(E-mp.eye(2))*b[i];u=-g[0]/E[0,1];return u,E[1,1]*u+g[1]
 def fun(t1,t2):
  # Return mismatch normalized by positive time prevents false zero-duration roots.
  if t1<=mp.mpf('1e-4') or t2<=mp.mpf('1e-4'):raise ValueError('nonpositive duration')
  q0,q1=flow(0,t1);q2,q3=flow(1,t2);return (q1-q2)/(1+abs(q1)+abs(q2)),(q3-q0)/(1+abs(q3)+abs(q0))
 guess1=mp.mpf(str(tf_est));guess2=mp.mpf(str(tn_est));solution=None;errors=[]
 for f in (1.,.99,1.02,.95,1.05):
  try:
   t1,t2=mp.findroot(fun,(guess1*mp.mpf(str(f)),guess2),solver='mdnewton',tol=mp.mpf('1e-25'),maxsteps=100,verify=True)
   if not (mp.isfinite(t1) and mp.isfinite(t2) and 0<t1<10000 and 0<t2<10000):continue
   mismatch=fun(t1,t2);q0,q1=flow(0,t1);q2,q3=flow(1,t2)
   if max(abs(mismatch[0]),abs(mismatch[1]))<mp.mpf('1e-20'):solution=(t1,t2,q0,q1,q2,q3);break
  except (ValueError,ZeroDivisionError,OverflowError,MemoryError) as e:errors.append(str(e)[:250])
 if solution is None:return report|dict(status='unresolved',errors=errors)
 t1,t2,q0,q1,q2,q3=solution
 def state(i,t,u):
  E=matrixexp(i,t);g=Ainv[i]*(E-mp.eye(2))*b[i];return E*mp.matrix([0,u])+g
 speeds=[(A[0]*mp.matrix([0,q0])+b[0])[0],(A[1]*mp.matrix([0,q1])+b[1])[0]]
 if not (speeds[0]>0 and speeds[1]<0):return report|dict(status='wrong_crossing_direction',speeds=list(map(str,speeds)))
 samples=[];valid=True
 for i,(duration,u) in enumerate(((t1,q0),(t2,q1))):
  vals=[state(i,duration*j/400,u)[0] for j in range(401)];samples.extend(vals)
  tol=mp.mpf('1e-18')*max(1,max(map(abs,vals)))
  if (i==0 and min(vals)<-tol) or (i==1 and max(vals)>tol):valid=False
 if not valid:return report|dict(status='invalid_sampled_itinerary')
 zmin=min(samples);zmax=max(samples);th0=mp.mpf(repr(float(m['th'][0])));th1=mp.mpf(repr(float(m['th'][1])));zlo=mp.mpf(repr(float(model.phi(1e-12,m['shape']))));zhi=mp.mpf(repr(float(model.phi(model.CMAX,m['shape']))))
 limits=[]
 if zmax>0:limits.extend([(th1-th0)/zmax,(zhi-th0)/zmax])
 if zmin<0:limits.append((th0-zlo)/(-zmin))
 delta_max=min(limits);logM=mp.mpf(repr(float(T[1])))*t1+mp.mpf(repr(float(T[0])))*t2
 # Choose an interior illustrative displacement that is representable in double precision.
 delta=min(mp.mpf('1e-10'),delta_max/10);low_z=th0+delta*zmin;high_z=th0+delta*zmax
 representable=float(mp.mpf(repr(float(model.events[0])))+delta)!=float(model.events[0]);cmin=cmax=None
 if representable:
  try:cmin=float(inverse(float(low_z),m['shape']));cmax=float(inverse(float(high_z),m['shape']))
  except ValueError:pass
 return report|dict(status='candidate_local_cycle',scope='Normalized affine matching at positive delta_I, high precision; 401 samples per leg for itinerary and extrema; no separate IVP or parameter-tolerance convergence check. The magnitude of the valid interval is an estimate.',focus_time_s=str(t1),node_time_s=str(t2),period_s=str(t1+t2),log_multiplier=str(logM),stable=logM<0,normalised_q_up=str(q0),normalised_q_down=str(q1),normalised_z_min=str(zmin),normalised_z_max=str(zmax),estimated_delta_I_max=str(delta_max),example_delta_I=str(delta),example_I_double_representable=representable,example_Cmin=cmin,example_Cmax=cmax,crossing_speeds_normalised=list(map(str,speeds)),matching_relative_residuals=list(map(str,fun(t1,t2))))

def main():
 p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--physical-run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 paths={'mixed':a.snapshot/'afm_mixed/parameters.json','physical':a.physical_run/'afm/comparison/new_rank_01/parameters.json'};result={'scope':'AFM lower HLB2 test, fixed candidate parameters, no refit. Validity beyond the first two regions and LR reproduction are not assumed.','models':{}}
 for name,path in paths.items():
  print('CHECK',name,str(path),flush=True)
  try:r=check(path.resolve())
  except (ValueError,OverflowError,ZeroDivisionError,MemoryError,ArithmeticError) as e:r=dict(status='error',error=str(e),path=str(path))
  result['models'][name]=r;(a.out/'hlb2_afm.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print('RESULT',name,json.dumps(r),flush=True)
 print('SAVED',a.out/'hlb2_afm.json',flush=True)
if __name__=='__main__':main()
