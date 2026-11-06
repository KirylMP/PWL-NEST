"""Compare selected physical PWA candidates with LR for one shared IP3 protocol."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.signal import find_peaks
from fit_lower_fixed_fold import rhs as pwa_rhs

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'paper/data/random_selected'
KER={'AM':.1,'AFM':.075,'FM':.05}
SEGMENTS=[(0.,80.,.2),(80.,160.,.58),(160.,240.,.3),(240.,320.,.75),(320.,400.,.2)]
DT=.1

def lr_rhs(I,y,K):
    C,h=y;Q=1.049*(I+.13)/(I+.9434)
    dc=(6*(I/(I+.13))**3*(C/(C+.08234))**3*h**3+.11)*(2-1.185*C)-.9*C*C/(C*C+K*K)
    return [dc,.2*(Q-(Q+C)*h)]

def initial_lr(K):
    Q=1.049*.13/.9434
    C=brentq(lambda c:lr_rhs(0.,[c,Q/(Q+c)],K)[0],1e-9,1.)
    return [C,Q/(Q+C)]

def integrate(field,init):
    state=np.asarray(init,float);times=[];values=[]
    for start,end,I in SEGMENTS:
        grid=np.arange(round(start/DT),round(end/DT)+1)*DT
        sol=solve_ivp(lambda t,y:field(I,y),(start,end),state,method='DOP853',t_eval=grid,rtol=1e-9,atol=1e-11,max_step=.1)
        if not sol.success or not np.isfinite(sol.y).all():raise RuntimeError(sol.message)
        times.append(sol.t if not times else sol.t[1:]);values.append(sol.y.T if not values else sol.y.T[1:]);state=sol.y[:,-1]
    return np.concatenate(times),np.concatenate(values)

def main():
    models=json.loads((OUT/'manifest.json').read_text())['models'];fig,axes=plt.subplots(4,1,figsize=(10,8.5),sharex=True,constrained_layout=True)
    t=np.arange(0,400.00001,DT);I=np.zeros_like(t)
    for start,end,v in SEGMENTS:I[(t>=start)&(t<end)]=v
    I[-1]=SEGMENTS[-1][2];axes[0].step(t,I,where='post',color='#3f566e');axes[0].set_ylabel(r'$I$ ($\mu$M)')
    result={};samples={'time_s':t,'I':I}
    for j,case in enumerate(('AM','AFM','FM'),1):
        raw=(ROOT/models[case]['candidate_path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=models[case]['candidate_sha256']:raise ValueError('hash mismatch')
        model=json.loads(raw);K=KER[case]
        tl,yl=integrate(lambda I,y:lr_rhs(I,y,K),initial_lr(K));tp,yp=integrate(lambda I,y:pwa_rhs(model,I,y),model['rest_target'])
        if not np.allclose(t,tl) or not np.allclose(t,tp):raise ValueError('sample grid mismatch')
        samples[case+'_LR_C']=yl[:,0];samples[case+'_PWA_C']=yp[:,0]
        segments=[]
        for start,end,v in SEGMENTS:
            mask=(t>=start)&(t<end);err=yp[mask,0]-yl[mask,0]
            segments.append({'start_s':start,'end_s':end,'I':v,'rmse_C':float(np.sqrt(np.mean(err**2))),
                             'LR_peaks':int(len(find_peaks(yl[mask,0],prominence=.01)[0])),
                             'PWA_peaks':int(len(find_peaks(yp[mask,0],prominence=.01)[0])),
                             'last_C_LR':float(yl[mask,0][-1]),'last_C_PWA':float(yp[mask,0][-1])})
        result[case]={'candidate_sha256':models[case]['candidate_sha256'],'lr_initial':initial_lr(K),'pwa_initial':model['rest_target'],
                      'min_C_LR':float(min(yl[:,0])),'min_C_PWA':float(min(yp[:,0])),
                      'rmse_C':float(np.sqrt(np.mean((yp[:,0]-yl[:,0])**2))),'segments':segments}
        axes[j].plot(t,yl[:,0],color='#252525',lw=1.1,label='LR');axes[j].plot(t,yp[:,0],color='#c5353c',lw=1.,label='PWA')
        axes[j].set_ylabel(case+' C (µM)');axes[j].grid(alpha=.2)
        if j==1:axes[j].legend(frameon=False,ncol=2,fontsize=9)
        print(case,'RMSE',result[case]['rmse_C'],'minimum',result[case]['min_C_PWA'],flush=True)
    axes[0].grid(alpha=.2);axes[-1].set_xlabel('Time (s)')
    out=ROOT/'paper/figures/random_selected_variable_input'
    fig.savefig(str(out)+'.pdf',bbox_inches='tight');fig.savefig(str(out)+'.png',dpi=200,bbox_inches='tight')
    np.savez_compressed(OUT/'variable_input_timeseries.npz',**samples)
    (OUT/'variable_input_summary.json').write_text(json.dumps({'protocol':SEGMENTS,'sample_interval_s':DT,'method':'DOP853, piecewise constant input; rtol=1e-9 atol=1e-11 max_step=0.1s','initialization':'separate I=0 equilibria; hidden q is not LR h', 'results':result},indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
