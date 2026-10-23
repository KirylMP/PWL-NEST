import os,sys,json,time
os.chdir('/mnt/data/slow_threshold_test')
sys.path.insert(0,os.getcwd())
import numpy as np
from scipy.optimize import differential_evolution
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from numba import njit
from search_slow import CO,TH,p,CREf,S,HR,HS,field
from scipy.linalg import eig
OUT='/mnt/data/riccati_slow_pilot'
A=np.zeros((3,3)); A[:2,:2]=CO[:,1:3]; A[0,0]-=p[0];A[0,2]+=p[0];A[1,0]-=p[1];A[1,2]+=p[1];A[2]=[p[2],p[3],-p[2]]
evals,V=np.linalg.eig(A);idx=np.argmin(abs(evals+.1));lam=float(evals[idx].real);v=np.real(V[:,idx]);l=np.real(np.linalg.inv(V)[idx,:]);
assert abs(l@v-1)<1e-12
@njit
def f(y,I,k,theta):
 u,h,r=y
 du=CO[0,0]+CO[0,1]*u+CO[0,2]*h+CO[0,3]*I+p[0]*(r-u)
 dh=CO[1,0]+CO[1,1]*u+CO[1,2]*h+CO[1,3]*I+p[1]*(r-u)
 dr=p[2]*(u-r)+p[3]*h+p[4]*(I-1.12)
 for j in range(3):
  sh=max(0.,u-TH[j]);du+=CO[0,4+j]*sh;dh+=CO[1,4+j]*sh
 s=l[0]*u+l[1]*h+l[2]*r
 correction=k*max(0.,s-theta)**2
 return np.array([du+v[0]*correction,dh+v[1]*correction,dr+v[2]*correction])
@njit
def integrate(I,k,theta,T=480.,dt=.025,stride=10):
 y=np.array([(1.3-CREf)/S,(.8-HR)/HS,(1.3-CREf)/S]);n=int(T/dt);res=np.empty((n//stride+1,3));res[0]=y
 for i in range(n):
  k1=f(y,I,k,theta);k2=f(y+.5*dt*k1,I,k,theta);k3=f(y+.5*dt*k2,I,k,theta);k4=f(y+dt*k3,I,k,theta)
  y=y+(dt/6)*(k1+2*k2+2*k3+k4)
  if not np.all(np.isfinite(y)) or np.max(np.abs(y))>60:return res[:1]
  if (i+1)%stride==0:res[(i+1)//stride]=y
 return res
IS=[.53,.55,.7,1.,1.12,1.13];targets=[46,35.75,22.75,17.5,17.75]
def metrics(I,k,theta,T=480.):
 z=integrate(I,k,theta,T);ts=np.arange(len(z))*.25
 if len(z)<10:return dict(I=I,period=None,amp=None,cmin=None,cmax=None,diverged=True)
 c=CREf+S*z[:,0];pk,_=find_peaks(c,prominence=.04,distance=35);late=pk[ts[pk]>.55*T]
 amp=float(np.ptp(c[ts>.7*T]));per=float(np.median(np.diff(ts[late]))) if len(late)>=4 and amp>.1 else None
 return dict(I=I,period=per,amp=amp,cmin=float(c.min()),cmax=float(c.max()),diverged=False)
def score(xx):
 k,theta=xx;vals=[metrics(i,k,theta,T=380) for i in IS];cost=0
 for r,t in zip(vals[:5],targets):
  if r['diverged']:return 1000
  if r['period'] is None:cost+=20
  else:cost+=12*np.log(r['period']/t)**2
  cost+=20*max(0.,-.003-r['cmin'])**2+20*max(0.,r['cmax']-1.36)**2
 if vals[-1]['diverged']:return 1000
 if vals[-1]['period'] is not None:cost+=20
 cost+=2*max(0.,vals[-1]['amp']-.06)**2
 return cost+.002*k*k
if __name__=='__main__':
 print('lam',lam,'left',l,'right',v,flush=True)
 print('baseline',[(a['I'],a['period']) for a in [metrics(I,0,.3) for I in IS]],flush=True)
 start=time.time();best=(score([0,.3]),[0,.3]);print('baseline score',best,flush=True)
 def objective(x):
  global best
  z=score(x)
  if z<best[0]:best=(z,list(x));print('best',round(z,5),np.round(x,5),'elapsed',round(time.time()-start,1),flush=True)
  return z
 res=differential_evolution(objective,bounds=[(-4.,4.),(.05,.55)],seed=42,maxiter=14,popsize=5,polish=False)
 x=np.asarray(best[1]);print('DONE',best,'elapsed',time.time()-start,flush=True)
 result=dict(model='global modal quadratic hinge; low-C region Riccati only',slow_eigenvalue=lam,slow_left=l.tolist(),slow_right=v.tolist(),k=float(x[0]),theta=float(x[1]),score=float(best[0]),baseline=[metrics(I,0,.3,T=620) for I in IS],new=[metrics(I,x[0],x[1],T=620) for I in IS])
 with open(OUT+'/results.json','w') as fp:json.dump(result,fp,indent=2)
 # independently check with DOP853 and event bounds
 t=np.arange(0,620.01,.2);rows=[]
 for i in IS:
  y0=[(1.3-CREf)/S,(.8-HR)/HS,(1.3-CREf)/S]
  def rhs(t,y):return np.asarray(f(np.asarray(y),i,x[0],x[1]))
  def guard(t,y):return 20.-max(abs(np.asarray(y)))
  guard.terminal=True;guard.direction=-1
  sol=solve_ivp(rhs,(0,620),y0,t_eval=t,method='DOP853',rtol=2e-9,atol=2e-11,max_step=.12,events=guard)
  c=CREf+S*sol.y[0];pt,_=find_peaks(c,prominence=.04,distance=35);late=pt[sol.t[pt]>350]
  amp=float(np.ptp(c[sol.t>400])) if len(c)>5 and np.any(sol.t>400) else float('nan')
  T=float(np.median(np.diff(sol.t[late]))) if len(late)>=4 and amp>.1 else None
  rows.append(dict(I=i,period=T,amp=amp,cmin=float(min(c)),cmax=float(max(c)),success=bool(sol.success),end=float(sol.t[-1])))
 print('DOP853',rows,flush=True);result['DOP853']=rows
 with open(OUT+'/results.json','w') as fp:json.dump(result,fp,indent=2)
