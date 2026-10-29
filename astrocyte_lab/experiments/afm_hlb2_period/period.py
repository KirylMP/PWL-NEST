"""Fast high-precision normalized two-region HLB2 flight times."""
import math
import mpmath as mp
import numpy as np

def limit_period(m):
 T=np.asarray(m['T']);D=np.asarray(m['D']);disc=T*T-4*D
 if not (T[0]<0 and D[0]>0 and disc[0]>0 and T[1]>0 and D[1]>0 and disc[1]<0):raise ValueError('Not stable node / unstable focus')
 omega=math.sqrt(-disc[1])/2;focus_rate=T[1]/2;slow=(-T[0]-math.sqrt(disc[0]))/2
 if slow<=0:raise ValueError('Node is not stable')
 t1=math.pi/omega+1/focus_rate;t2=t1*focus_rate/slow
 digits=max(60,int(math.ceil(focus_rate*t1/math.log(10)))+45)
 if digits>500 or t2>5000:raise ValueError('Period calculation exceeds precision or time limits')
 mp.mp.dps=digits;A=[];b=[];ai=[]
 for r in (1,0):
  co=m['co'];mat=mp.matrix([[mp.mpf(repr(float(co[i,1]+sum(co[i,4:4+r])))),mp.mpf(repr(float(co[i,2])))] for i in (0,1)]);inp=mp.matrix([mp.mpf(repr(float(co[i,3]))) for i in (0,1)]);A.append(mat);b.append(inp);ai.append(mat**-1)
 def flow(i,t):
  M=A[i];tr=(M[0,0]+M[1,1])/2;w2=((M[0,0]-M[1,1])/2)**2+M[0,1]*M[1,0]
  if w2<0:w=mp.sqrt(-w2);c=mp.cos(w*t);s=mp.sin(w*t)/w
  elif w2>0:w=mp.sqrt(w2);c=mp.cosh(w*t);s=mp.sinh(w*t)/w
  else:c=mp.mpf(1);s=t
  E=mp.exp(tr*t)*(c*mp.eye(2)+s*(M-tr*mp.eye(2)));g=ai[i]*(E-mp.eye(2))*b[i];q=-g[0]/E[0,1];return q,E[1,1]*q+g[1]
 def equations(x,y):
  if x<mp.mpf('0.001') or y<mp.mpf('0.001') or x>5000 or y>5000:raise ValueError('Invalid flight time')
  q0,q1=flow(0,x);q2,q3=flow(1,y);return ((q1-q2)/(1+abs(q1)+abs(q2)),(q3-q0)/(1+abs(q3)+abs(q0)))
 roots=None
 for f in (1.,.98,1.02):
  try:
   r=mp.findroot(equations,(mp.mpf(repr(float(t1*f))),mp.mpf(repr(float(t2)))),tol=mp.mpf('1e-16'),maxsteps=40,verify=True)
   x,y=r[0],r[1];errs=equations(x,y);q0,q1=flow(0,x);s0=(A[0]*mp.matrix([0,q0])+b[0])[0];s1=(A[1]*mp.matrix([0,q1])+b[1])[0]
   if x>0 and y>0 and s0>0 and s1<0 and max(map(abs,errs))<mp.mpf('1e-12'):roots=(float(x),float(y));break
  except (ValueError,ZeroDivisionError,OverflowError,ArithmeticError):pass
 if roots is None:raise ValueError('Local cycle flight times unresolved')
 logm=T[1]*roots[0]+T[0]*roots[1]
 if logm>=0:raise ValueError('Local orbit is not stable')
 return roots[0]+roots[1],roots,logm
