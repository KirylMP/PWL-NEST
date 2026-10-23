import json, numpy as np, time
from numba import njit
from scipy.optimize import differential_evolution
from scipy.signal import find_peaks
from scipy.integrate import solve_ivp
from run import CREf,S,HR,HS
J=json.load(open('validated.json'));CO=np.array(J['fitted_coefficients']);TH=np.array(J['thresholds']);p=np.array(json.load(open('augmented_results.json'))['candidates'][0]['parameters'])
@njit
def field(y,I,coef,th,p,typ,beta,theta,k):
 u,v,r=y
 du=coef[0,0]+coef[0,1]*u+coef[0,2]*v+coef[0,3]*I
 dv=coef[1,0]+coef[1,1]*u+coef[1,2]*v+coef[1,3]*I
 for j in range(len(th)):
  a=max(0.,u-th[j]);du+=coef[0,4+j]*a;dv+=coef[1,4+j]*a
 du+=p[0]*(r-u);dv+=p[1]*(r-u)
 dr=p[2]*(u-r)+p[3]*v+p[4]*(I-1.12)
 s=u if typ==0 else (r if typ==1 else u+beta*v)
 hinge=max(0.,theta-s)
 return np.array([du+k[0]*hinge,dv+k[1]*hinge,dr+k[2]*hinge])
@njit
def sim(I,coef,th,p,typ,beta,theta,k,T=420.,dt=.025,store=10):
 y=np.array([(1.3-CREf)/S,(.8-HR)/HS,(1.3-CREf)/S]);n=int(T/dt);out=np.empty((n//store+1,3));out[0]=y
 for it in range(n):
  q1=field(y,I,coef,th,p,typ,beta,theta,k)
  q2=field(y+q1*(dt/2),I,coef,th,p,typ,beta,theta,k)
  q3=field(y+q2*(dt/2),I,coef,th,p,typ,beta,theta,k)
  q4=field(y+q3*dt,I,coef,th,p,typ,beta,theta,k)
  y+=dt/6*(q1+2*q2+2*q3+q4)
  if not np.all(np.isfinite(y)) or np.max(np.abs(y))>30:return out[:1]
  if (it+1)%store==0:out[(it+1)//store]=y
 return out
IS=[.53,.55,.7,1.,1.12,1.13]; TARGET=[46.,35.75,22.75,17.5,17.75]
def assess(I,typ,beta,theta,k,T=420):
 z=sim(I,CO,TH,p,typ,beta,theta,np.asarray(k),T=T);dt=.25
 if len(z)<10:return None,0,-10,10,True
 c=CREf+S*z[:,0];t=np.arange(len(c))*dt
 peaks,_=find_peaks(c,prominence=.04,distance=28)
 late=peaks[t[peaks]>.50*T]
 per=float(np.median(np.diff(t[late]))) if len(late)>=4 else None
 amp=float(np.ptp(c[t>.65*T]));iscycle=per is not None and amp>.12 and np.std(np.diff(t[late]))/per<.10
 return per if iscycle else None,amp,float(np.min(c)),float(np.max(c)),False

def score(x,typ):
 theta,beta=x[:2];k=np.asarray(x[2:]);s=0.
 for I,T in zip(IS[:5],TARGET):
  per,amp,cmin,cmax,broken=assess(I,typ,beta,theta,k,T=355)
  if broken:return 200
  if per is None:s+=9
  else:s+=8*np.log(per/T)**2
  s+=6*max(0.,-.002-cmin)**2+8*max(0.,cmax-1.7)**2
  s+=2*max(0.,.6-amp)**2
 per,amp,cmin,cmax,broken=assess(1.13,typ,beta,theta,k,T=355)
 if broken:return 200
 s+=(9 if per else 0)+max(0,amp-.025)*4
 s+=.0003*np.sum(k*k)
 return float(s)

def summarize(x,typ):
 theta,beta=x[:2];k=x[2:];return {'theta':float(theta),'beta':float(beta),'k':k.tolist(),'score':score(x,typ),'rows':[{'I':I,'T':assess(I,typ,beta,theta,k,T=620)[0],'amplitude':assess(I,typ,beta,theta,k,T=620)[1],'C_min':assess(I,typ,beta,theta,k,T=620)[2],'C_max':assess(I,typ,beta,theta,k,T=620)[3]} for I in IS]}
if __name__=='__main__':
 t0=time.time();res={}
 for typ,name in [(0,'C'),(1,'r'),(2,'tilt')]:
  # hinge active in bottom region C<.043; beta tilt coefficient if tilt
  bounds=[((.031-CREf)/S,(.042-CREf)/S) if typ!=1 else (-.14,.025),(-.04,.04) if typ==2 else (0,0),(-4.,4.),(-3.,3.),(-5.,5.)]
  print('START',name,'baseline',score([(.037-CREf)/S,0,0,0,0],typ),flush=True)
  opt=differential_evolution(lambda x:score(x,typ),bounds,popsize=6,maxiter=14,polish=False,seed=2026,workers=1)
  result=summarize(opt.x,typ);res[name]=result
  print('DONE',name,'time',time.time()-t0,'score',opt.fun,'parameters',opt.x,'periods',[v['T'] for v in result['rows']],flush=True)
  open('search_results.json','w').write(json.dumps(res,indent=2))
