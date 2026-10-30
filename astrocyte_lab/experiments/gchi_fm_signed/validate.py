import os,json
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.integrate import solve_ivp
P=Path(__file__).resolve().parent
m=json.load(open(P/os.environ.get('FIT_OUTPUT','asinh_free_q_refined.json')));refs=json.load(open(P/'reference_cycles.json'))['results'];train=[r for r in refs if r['mode']=='FM' and r['valid'] and r['gamma'] in [.022,.04,.1,.4,1.3,5.]]
def calc(r):
 co=np.array(m['coefficients']);cuts=m['thresholds_C'];gamma=r['gamma'];u=np.arcsinh(gamma/m['input_scale']);x=np.array(r['last_peak_state']);c,i=x[0],x[2]
 if abs(co[0,2])<1e-10:return {'gamma':gamma,'valid':False,'reason':'q nullcline unavailable'}
 x[1]=-(co[0,0]+co[0,1]*c+co[0,3]*i+sum(co[0,4+j]*max(c-cuts[j],0) for j in range(3)))/co[0,2]
 if gamma==5.:x=np.array(json.load(open(P/'initial_scan.json'))[-1]['hits'][0]['initial'])
 def f(t,x):return co@np.array([1,*x,*[max(x[0]-z,0) for z in cuts],u])
 def peak(t,x):return f(t,x)[0]
 peak.direction=-1
 sol=solve_ivp(f,(0,1600),x,method='DOP853',rtol=2e-8,atol=1e-10,max_step=.15,events=peak,dense_output=True);ts=sol.t_events[0];ts=ts[ts>800];out={'gamma':gamma,'valid':False,'peaks':int(len(ts)),'solver_success':bool(sol.success),'trajectory_min_C':float(min(sol.y[0])),'trajectory_min_I':float(min(sol.y[2]))}
 if len(ts)<3 or not sol.success:return out
 T=ts[-1]-ts[-2];Y=sol.sol(np.linspace(ts[-2],ts[-1],1201));closure=float(np.max(abs(Y[:,-1]-Y[:,0])/np.maximum(1,abs(Y[:,0]))));out.update({'period':float(T),'Cmin':float(min(Y[0])),'Cmax':float(max(Y[0])),'Imin':float(min(Y[2])),'Imax':float(max(Y[2])),'closure':closure,'reference':{'period':r['period'],'Cmin':r['Cmin'],'Cmax':r['Cmax'],'Imin':r['Imin'],'Imax':r['Imax']}});out['valid']=bool(closure<1e-4 and abs((ts[-2]-ts[-3])-T)<max(.1,T*.01) and min(sol.y[0])>0 and min(sol.y[2])>=0 and np.ptp(Y[0])>.001)
 return out
with ProcessPoolExecutor(max_workers=int(os.environ.get('VALIDATION_WORKERS','1'))) as pool:
 out=[]
 for x in pool.map(calc,train):out.append(x);print(x['gamma'],'valid',x['valid'],'T',x.get('period'),'Cmin',x['trajectory_min_C'],flush=True)
(P/'validation.json').write_text(json.dumps({'scope':'1600s DOP853, one initial condition per gamma, full-state closure; no proof of no cycles','results':out},indent=2)+'\n')
