import numpy as np, itertools, json, time
from scipy.signal import find_peaks
from numba import njit
from scipy.optimize import least_squares
TH=np.array([.015194801159961467,.019041346380289176,.021792793150308,.03012522054076897,.14328039311772045,.25703388004174615,.3606679278346462])
CO=np.array([[.543150501625349,-65.03448277371949,.39963985456152756,.4260600169237272,65.13034286600633,.0004685526594565559,.22470178632733132,4.848499668132286,-3.5713077543201255,-1.5021552195302688,-.6112561453044335],[-.024495138874266832,-.4551445779660534,-.0959340132018189,.18515780974258322,-2.7407718801863443,.6941398702952859,2.4240291537967096,-.9848248044215164,.46114488213624283,-.22620767528155494,-.17697536932393854]])
CREf=.02812613673817445; S=.9833352639826316; HR=.8371177795140701; HS=HR*(1-HR)
@njit
def vel(u,v,I,th,co):
 x=co[0,0]+co[0,1]*u+co[0,2]*v+co[0,3]*I
 y=co[1,0]+co[1,1]*u+co[1,2]*v+co[1,3]*I
 for j in range(len(th)):
  w=max(0.,u-th[j]);x+=co[0,j+4]*w;y+=co[1,j+4]*w
 return x,y
@njit
def sim(I,th,co,dt,nsteps,u0,v0):
 u=u0;v=v0
 out=np.empty((nsteps//10+1,2));out[0]=u,v
 for k in range(nsteps):
  k1,l1=vel(u,v,I,th,co)
  k2,l2=vel(u+.5*dt*k1,v+.5*dt*l1,I,th,co)
  k3,l3=vel(u+.5*dt*k2,v+.5*dt*l2,I,th,co)
  k4,l4=vel(u+dt*k3,v+dt*l3,I,th,co)
  u+=dt*(k1+2*k2+2*k3+k4)/6
  v+=dt*(l1+2*l2+2*l3+l4)/6
  if not np.isfinite(u) or not np.isfinite(v) or abs(u)>100 or abs(v)>100:
   return out[:1]
  if (k+1)%10==0:out[(k+1)//10]=u,v
 return out
@njit
def sample(I, th,co,dt,nsteps,u0,v0,skip):
 z=sim(I,th,co,dt,nsteps,u0,v0)
 return z[skip::8]
def metrics(I,th,co,T=550):
 z=sim(I,th,co,.025,round(T/.025),(1.3-CREf)/S,(.8-HR)/HS)
 if len(z)<10:return {'period':None,'amp':None,'valid':False}
 c=CREf+S*z[:,0]; t=np.arange(len(c))*.25
 p,_=find_peaks(c,prominence=.025,distance=8)
 p=p[t[p]>T*.55]
 amp=float(np.ptp(c[t>T*.65])); period=float(np.median(np.diff(t[p]))) if len(p)>=4 else None
 stable=period is not None and np.std(np.diff(t[p]))/period<.08 and amp>.08
 return {'period':round(period,3) if stable else None,'amp':round(amp,4),'valid':True,'n_peaks_late':len(p)}
def fit(k,inds,X,V):
 th=TH[list(inds)];D=np.column_stack([np.ones(len(X)),X[:,0],X[:,1],X[:,2]]+[np.maximum(X[:,0]-v,0) for v in th])
 reg=np.diag([1e-6,1e-6,1e-6,1e-6]+[1e-4]*k)
 co=np.linalg.solve(D.T@D+reg,D.T@V).T
 return th,co
if __name__=='__main__':
 t0=time.time();Is=[.53,.55,.7,1.,1.12]
 base={str(i):metrics(i,TH,CO) for i in Is};print('base',base,flush=True)
 X=[];V=[]
 for I in Is:
  z=sample(I,TH,CO,.025,round(550/.025),(1.3-CREf)/S,(.8-HR)/HS,350)
  z=z[np.isfinite(z).all(axis=1)];z=z[::2]
  inp=np.column_stack((z,np.full(len(z),I)))
  y=np.array([vel(row[0],row[1],I,TH,CO) for row in z]);X.append(inp);V.append(y)
 X=np.vstack(X);V=np.vstack(V)
 print('samples',len(X),flush=True)
 results=[]
 for k in [3,4,5]:
  for ind in itertools.combinations(range(7),k):
   th,co=fit(k,ind,X,V)
   m=[metrics(I,th,co,T=350) for I in Is]
   hits=sum(x['period'] is not None for x in m)
   score=sum((np.log(x['period']/base[str(I)]['period'])**2 if x['period'] else 12) for I,x in zip(Is,m))
   results.append((score,hits,ind,m,co))
   if hits>=4:print('candidate',k,ind,'score',round(score,3),'periods',[x['period'] for x in m],flush=True)
 print('top:')
 results.sort(key=lambda r:r[0])
 for r in results[:6]:print(round(r[0],4),r[2],[x['period'] for x in r[3]],flush=True)
 out={'base':base,'best':[{'score':float(r[0]),'indices':list(r[2]),'thresholds':TH[list(r[2])].tolist(),'metrics':r[3],'coefficients':r[4].tolist()} for r in results[:10]],'seconds':time.time()-t0}
 open('/mnt/data/fm_mechanism_compression/results.json','w').write(json.dumps(out,indent=2))
 print('elapsed',time.time()-t0)
