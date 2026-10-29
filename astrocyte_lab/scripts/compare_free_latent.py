#!/usr/bin/env python3
"""DOP853 calcium comparison with a hidden coordinate independent of LR h."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from fit_physical_calcium import PhysicalModel
from fit_free_latent import FreeModel

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('free',type=Path);ap.add_argument('baseline',type=Path);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--inputs',help='Comma-separated I; omit for training inputs');ap.add_argument('--duration',type=float,default=300);ap.add_argument('--workers',type=int,default=3)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    free=json.loads((a.free/'parameters.json').read_text());base=json.loads((a.baseline/'parameters.json').read_text());case=free['case'].lower()
    if base.get('case','').lower()!=case or base.get('gate_coordinate')!='identity':raise ValueError('Expected same-case identity baseline')
    fm=FreeModel(case);pm=PhysicalModel(case,'identity');models=[pm.unpack(base['parameters']),fm.unpack(free['parameters'])];orbits=np.load(fm.root/'orbit_samples.npz')
    inputs=np.unique(orbits['I']) if not a.inputs else np.array([float(x) for x in a.inputs.split(',')]);grid=np.arange(0,a.duration+.05,.1)
    def run(I):
        I=float(I);closest=float(np.unique(orbits['I'])[np.argmin(abs(np.unique(orbits['I'])-I))]);idx=np.flatnonzero(orbits['I']==closest);j=idx[np.argmax(orbits['C'][idx])]
        y0=np.array([float(orbits['C'][j]),float(orbits['h'][j])]);cdot=float(fm.lr.rhs(y0,I)[0]);result=[]
        for k in range(3):
            if k==0:
                initial=y0
                def rhs(t,y):return fm.lr.rhs(y,I)
            else:
                model=pm if k==1 else fm;m=models[k-1];hidden=float(model.basis(y0[0],I,m)@m['N']+cdot/m['a']);initial=np.array([y0[0],hidden])
                def rhs(t,y):return m['co']@np.r_[1.,y[0],y[1],I,np.maximum(y[0]-m['padded'],0.)]
            sol=solve_ivp(rhs,(0,a.duration),initial,t_eval=grid,method='DOP853',rtol=2e-9,atol=2e-11,max_step=.05)
            if not sol.success or len(sol.t)!=len(grid):raise ValueError(f'Integration failed: I={I} model={k} {sol.message}')
            C=sol.y[0];tail=C[len(C)//2:];peaks,_=find_peaks(tail,prominence=.005)
            info=dict(Cmin=float(tail.min()),Cmax=float(tail.max()),peaks=int(len(peaks)),period=float(np.median(np.diff(grid[len(C)//2:][peaks]))) if len(peaks)>=2 else None,hidden_min=float(sol.y[1].min()),hidden_max=float(sol.y[1].max()),physical_C_domain=bool(np.all((C>0)&(C<fm.CMAX))))
            result.append((C,info))
        print(f'I={I:.6f} '+ ' '.join(f'{k}: max={s[1]["Cmax"]:.4f} T={s[1]["period"]} peaks={s[1]["peaks"]}' for k,s in zip(('LR','baseline','free'),result)),flush=True)
        return I,y0[0],cdot,result
    with ThreadPoolExecutor(max_workers=a.workers) as pool:out=list(pool.map(run,inputs))
    result=dict(case=case,method='DOP853, fixed I and same initial observed C and Cdot; C and h for LR are taken near the peak of the nearest archived LR orbit, but h is never an objective for the free model. Peaks in latter half. Not a complete cycle or bifurcation scan.',settings=dict(duration=a.duration,inputs_source='explicit' if a.inputs else 'training'),rows=[dict(I=I,C0=C,initial_Cdot=cdot,LR=s[0][1],baseline=s[1][1],free=s[2][1]) for I,C,cdot,s in out])
    (a.out/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,which in zip(axes.flat,np.linspace(0,len(out)-1,4,dtype=int)):
        I,C,cdot,sols=out[which]
        for (curve,_),color,label in zip(sols,('tab:purple','black','tab:orange'),('LR','physical h','free q')):ax.plot(grid,curve,color=color,label=label,lw=1)
        ax.set(title=f'I={I:.4f} µM',xlabel='Time (s)',ylabel='C (µM)');ax.grid(alpha=.2)
    axes.flat[0].legend();fig.suptitle(f'{case.upper()}: same initial C and Cdot')
    fig.savefig(a.out/'comparison.png',dpi=150);plt.close(fig)
    print(f'Saved {a.out}',flush=True)

if __name__=='__main__':main()
