import numpy as np,json
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from search_slow import CO,TH,p,CREf,S,HR,HS,field,IS
results=json.load(open('search_results.json'))
def check(name,x,I):
 theta=x['theta'];beta=x['beta'];k=np.array(x['k']);typ={'C':0,'r':1,'tilt':2}[name]
 y0=np.array([(1.3-CREf)/S,(.8-HR)/HS,(1.3-CREf)/S]);t=np.arange(0,620.001,.2)
 sol=solve_ivp(lambda t,y:field(y,I,CO,TH,p,typ,beta,theta,k),(0,620),y0,t_eval=t,method='DOP853',rtol=2e-9,atol=2e-11,max_step=.12)
 c=CREf+S*sol.y[0];h=HR+HS*sol.y[1];pk,_=find_peaks(c,prominence=.04,distance=35);late=pk[t[pk]>320]
 period=round(float(np.median(np.diff(t[late]))),3) if len(late)>4 and np.ptp(c[t>400])>.12 else None
 return {'I':I,'period':period,'late_amp':round(float(np.ptp(c[t>400])),4),'C_min':round(float(c.min()),4),'C_max':round(float(c.max()),4),'h_min':round(float(h.min()),4),'h_max':round(float(h.max()),4)}
base={'theta':(.037-CREf)/S,'beta':0,'k':[0,0,0]};results['baseline']=base
for name in ['baseline','C','r','tilt']:
 typ='C' if name=='baseline' else name
 results[name]['DOP853']=[check(typ,results[name],I) for I in IS]
 print(name,results[name]['DOP853'],flush=True)
open('validated_slow.json','w').write(json.dumps(results,indent=2))
