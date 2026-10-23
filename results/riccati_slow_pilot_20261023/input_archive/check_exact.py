import json,numpy as np,os,sys
from scipy.integrate import solve_ivp
from scipy.linalg import expm
sys.path.insert(0,'/mnt/data/riccati_slow_pilot')
import experiment as E
D=json.load(open('/mnt/data/riccati_slow_pilot/results.json'));k,theta=D['k'],D['theta'];l,v,A=E.l,E.v,E.A;Q=np.eye(3)-np.outer(v,l)

def exact_low(y0,I,dt):
 b=np.array([E.CO[0,0]+E.CO[0,3]*I,E.CO[1,0]+E.CO[1,3]*I,E.p[4]*(I-1.12)])
 s0=l@y0;d=l@b
 if s0>theta:
  a=d+k*theta**2;bb=E.lam-2*k*theta;c=k
 else:
  a=d;bb=E.lam;c=0
 M=np.array([[bb/2,a],[-c,-bb/2]])
 pq=expm(M*dt)@np.array([s0,1.]);s=pq[0]/pq[1]
 big=np.zeros((4,4));big[:3,:3]=A;big[:3,3]=Q@b
 z=expm(big*dt)@np.r_[Q@y0,1.]
 return z[:3]+v*s

# choose initial inside low C and active s, time tiny to avoid crossing
samples=[]
for I in [.53,.7,1.0]:
 for sl in [.25,.36,.51]:
  y0=np.array([.009,sl, .009]);ss=l@y0
  if ss<theta:continue
  for dt in [.001,.01,.05]:
   sol=solve_ivp(lambda t,y:E.f(y,I,k,theta),(0,dt),y0,rtol=5e-13,atol=5e-15,method='DOP853')
   x1=exact_low(y0,I,dt);err=float(np.max(abs(x1-sol.y[:,-1])))
   samples.append({'I':I,'s':ss,'dt':dt,'error':err,'u_min':float(min(sol.y[0])),'u_max':float(max(sol.y[0]))})

# orbit fractions active in low-C region
stats=[]
for I in E.IS:
 z=E.integrate(I,k,theta,620.,.025,10);tail=z[-1000:];s=tail@l;on=s>theta;low=tail[:,0]<E.TH[0]
 stats.append({'I':I,'active_fraction':float(on.mean()),'active_outside_low_region_fraction':float((on&~low).mean()),'slow_region_fraction':float(low.mean())})
print('exact errors',samples,'stats',stats)
D['local_exact_test']=samples;D['activation_statistics']=stats
open('/mnt/data/riccati_slow_pilot/results.json','w').write(json.dumps(D,indent=2))
