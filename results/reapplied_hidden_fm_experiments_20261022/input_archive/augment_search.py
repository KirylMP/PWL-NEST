import json,sys,time
import numpy as np
from numba import njit
from scipy.optimize import differential_evolution
from scipy.signal import find_peaks
sys.path.insert(0,'/mnt/data/fm_mechanism_compression')
from run import TH,CO,CREf,S,HR,HS,metrics
j=json.load(open('/mnt/data/fm_mechanism_compression/validated.json'))
th=np.array(j['thresholds']);co=np.array(j['fitted_coefficients']);u0=(1.3-CREf)/S;v0=(.8-HR)/HS
# r tracks u; coupling to both observation derivatives; extra modulating term on dr from v deviation as allowed
@njit
def f(u,v,r,I,co,th,p):
 du=co[0,0]+co[0,1]*u+co[0,2]*v+co[0,3]*I
 dv=co[1,0]+co[1,1]*u+co[1,2]*v+co[1,3]*I
 for k in range(len(th)):
  w=max(0.,u-th[k]); du+=co[0,4+k]*w;dv+=co[1,4+k]*w
 z=r-u
 du+=p[0]*z;dv+=p[1]*z
 dr=p[2]*(u-r)+p[3]*v+p[4]*(I-1.12)
 return du,dv,dr
@njit
def simulate(I,co,th,p,dt,n,store=5):
 u=u0;v=v0;r=u0
 out=np.empty((n//store+1,3));out[0,0]=u;out[0,1]=v;out[0,2]=r
 for k in range(n):
  k1=f(u,v,r,I,co,th,p)
  k2=f(u+.5*dt*k1[0],v+.5*dt*k1[1],r+.5*dt*k1[2],I,co,th,p)
  k3=f(u+.5*dt*k2[0],v+.5*dt*k2[1],r+.5*dt*k2[2],I,co,th,p)
  k4=f(u+dt*k3[0],v+dt*k3[1],r+dt*k3[2],I,co,th,p)
  u+=dt*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6
  v+=dt*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6
  r+=dt*(k1[2]+2*k2[2]+2*k3[2]+k4[2])/6
  if not np.isfinite(u) or not np.isfinite(v) or not np.isfinite(r) or max(abs(u),abs(v),abs(r))>50:return out[:1]
  if (k+1)%store==0:out[(k+1)//store]=u,v,r
 return out

def eval_I(I,p,T=330,dt=.05):
 z=simulate(I,co,th,np.array(p),dt,round(T/dt))
 if len(z)<10:return None,0,False,0
 c=CREf+S*z[:,0];t=np.arange(len(c))*dt*5
 peaks,_=find_peaks(c,prominence=.03,distance=int(7/(dt*5)))
 late=peaks[t[peaks]>T*.52]
 amp=float(np.ptp(c[t>T*.65]));
 period=float(np.median(np.diff(t[late]))) if len(late)>=4 else None
 stable=period is not None and np.std(np.diff(t[late]))/period<.10 and amp>.08
 return period if stable else None,amp,True,len(late)

Is=[.53,.55,.7,1.,1.12]
target=np.array([41.25,35.25,23.75,17.5,17.0])
count=0;best=(1e9,None)
def objective(x):
 global count,best
 count+=1
 out=0
 for I,Tg in zip(Is,target):
  p,amp,valid,_=eval_I(I,x)
  if not valid: return 100+sum(np.square(x))
  if p is None:out+=7+(.5 if amp<.05 else 0)
  else:out+=12*np.log(p/Tg)**2
 # high I must be rest using identical seed and length
 p,amp,valid,_=eval_I(1.13,x,T=380)
 if not valid:return 100+sum(np.square(x))
 out+=(9 if p is not None else 0)+ max(0,amp-.03)*3
 out+=.004*(x[0]**2+x[1]**2)+.05*(x[3]**2+x[4]**2)
 if out<best[0]:
  best=(out,x.copy())
  if count%20==0 or out<1.5:print('best',count,round(out,4),np.round(x,4),flush=True)
 return out
if __name__=='__main__':
 seed= np.array([0.,0.,.7,0.,0.]);print('baseline',objective(seed),flush=True)
 bounds=[(-2,2),(-2,2),(.05,2.5),(-.6,.6),(-.8,.8)]
 t=time.time()
 res=differential_evolution(objective,bounds,maxiter=32,popsize=9,seed=42,polish=False,workers=1,init='latinhypercube')
 print('complete',res.fun,res.x,'seconds',time.time()-t,flush=True)
 candidates=[('best',best[1]),('optimizer',res.x),('baseline',seed)]
 report=[]
 for name,x in candidates:
  rows=[]
  for I in [.53,.54,.55,.6,.7,.8,.9,1.,1.1,1.12,1.13,1.14]:
   p,amp,valid,npk=eval_I(I,x,T=620,dt=.025)
   rows.append({'I':I,'period':p,'amplitude':amp,'valid':valid,'n_late':npk})
  report.append({'name':name,'parameters':x.tolist(),'rows':rows})
 print(json.dumps(report,indent=2),flush=True)
 with open('/mnt/data/fm_mechanism_compression/augmented_results.json','w') as f:json.dump({'optimizer':res.fun,'evals':count,'candidates':report},f,indent=2)
