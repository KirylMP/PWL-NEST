import os,json,sys
import numpy as np
from scipy.signal import find_peaks
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,'/mnt/data/fm_mechanism_compression')
from run import TH,CO,CREf,S,HR,HS,sim as base_sim
from augment_search import simulate,co,th,eval_I
j=json.load(open('/mnt/data/fm_mechanism_compression/augmented_results.json'))
p=np.array(j['candidates'][0]['parameters'])
rows=[]
for I in [.53,.54,.55,.6,.7,.8,.9,1.,1.1,1.12,1.125,1.127,1.13,1.135,1.14]:
 z=simulate(I,co,th,p,.025,int(700/.025))
 t=np.arange(len(z))*.125;C=CREf+S*z[:,0];h=HR+HS*z[:,1]
 mask=t>450
 per,amp,valid,_=eval_I(I,p,T=620,dt=.025)
 rows.append({'I':I,'period':per,'amplitude':amp,'C_min':float(C[mask].min()),'C_max':float(C[mask].max()),'h_min':float(h[mask].min()),'h_max':float(h[mask].max()),'r_min':float(z[mask,2].min()),'r_max':float(z[mask,2].max())})
print('bounds',[(r['I'],round(r['C_min'],3),round(r['C_max'],3),round(r['h_min'],3),round(r['h_max'],3)) for r in rows])
fig,ax=plt.subplots(2,1,figsize=(9,8),layout='constrained')
xs=np.array([r['I'] for r in rows]);ys=np.array([r['period'] if r['period'] else np.nan for r in rows])
ax[0].plot(xs,ys,'o-',label='4 regions + hidden state')
source=[41.25,37.25,35.25,29.5,23.75,20.75,18.75,17.5,16.75,17.0]
ax[0].plot(xs[:10],source,'s--',label='8-region source')
ax[0].axvline(1.13,linestyle=':',color='gray');ax[0].set(ylabel='Period (s)',xlabel='Constant IP3 (uM)');ax[0].legend();ax[0].grid(alpha=.2)
for I in [.55,.7,1.12,1.13]:
 z=simulate(I,co,th,p,.025,int(620/.025));t=np.arange(len(z))*.125
 ax[1].plot(t[t>=520]-520,(CREf+S*z[:,0])[t>=520],label=f'I={I}')
ax[1].set(xlabel='Time within late 100 s (s)',ylabel='Calcium C (uM)');ax[1].legend();ax[1].grid(alpha=.2)
fig.savefig('/mnt/data/fm_mechanism_compression/augmented_periods.png',dpi=160)
# Test different C h initial conditions, source and augmented
from numba import njit
from augment_search import f
@njit
def alt_sim(I,p,u,v,r,dt,n):
 out=np.empty((n//5+1,2));out[0]=u,v
 for k in range(n):
  k1=f(u,v,r,I,co,th,p)
  k2=f(u+.5*dt*k1[0],v+.5*dt*k1[1],r+.5*dt*k1[2],I,co,th,p)
  k3=f(u+.5*dt*k2[0],v+.5*dt*k2[1],r+.5*dt*k2[2],I,co,th,p)
  k4=f(u+dt*k3[0],v+dt*k3[1],r+dt*k3[2],I,co,th,p)
  u+=dt*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6
  v+=dt*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6
  r+=dt*(k1[2]+2*k2[2]+2*k3[2]+k4[2])/6
  if (k+1)%5==0:out[(k+1)//5]=u,v
 return out
from run import sim as source_sim
other=[]
for I in [1.12,1.13]:
 for C0,h0 in [(.08,.8),(.6,.8),(1.3,.8),(.04,.91)]:
  u=(C0-CREf)/S;v=(h0-HR)/HS
  za=alt_sim(I,p,u,v,u,.025,round(600/.025))
  zs=source_sim(I,TH,CO,.025,round(600/.025),u,v)
  def assess(z,step):
   c=CREf+S*z[:,0];tt=np.arange(len(c))*step;pk,_=find_peaks(c,prominence=.025,distance=50); pk=pk[tt[pk]>350]
   return {'amplitude':round(float(np.ptp(c[tt>430])),4),'late_peaks':len(pk),'final_C':round(float(c[-1]),4)}
  other.append({'I':I,'C0':C0,'h0':h0,'source':assess(zs,.25),'augmented':assess(za,.125)})
print('IC',other)
with open('/mnt/data/fm_mechanism_compression/augmented_validation.json','w') as f:json.dump({'parameters':p.tolist(),'rows':rows,'initial_conditions':other},f,indent=2)
