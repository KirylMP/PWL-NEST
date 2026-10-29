#!/usr/bin/env python3
"""Compare fixed-input trajectories against LR without a Poincare map."""
import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks
from fit_physical_calcium import PhysicalModel

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('identity',type=Path);ap.add_argument('power',type=Path);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--duration',type=float,default=300);ap.add_argument('--sample',type=float,default=.1);ap.add_argument('--workers',type=int,default=3)
    ap.add_argument('--inputs',help='Comma-separated fixed I values; by default uses the LR orbit-field training inputs')
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    configs=[]
    for path in (a.identity,a.power):
        data=json.loads((path/'parameters.json').read_text());case=data['case'].lower();model=PhysicalModel(case,data['gate_coordinate']);configs.append((data,model,model.unpack(data['parameters'])))
    if configs[0][0]['case']!=configs[1][0]['case']:raise ValueError('Models must use same AM/FM source')
    model=configs[0][1];orb=np.load(model.root/'orbit_samples.npz');inputs=np.unique(orb['I']) if not a.inputs else np.array([float(x) for x in a.inputs.split(',')]);grid=np.arange(0,a.duration+a.sample/2,a.sample)
    if a.duration<60 or a.sample<=0 or a.workers<1:raise ValueError('Invalid integration settings')
    tasks=[]
    for I in inputs:
        j=int(np.argmin(np.abs(orb['I']-I)));y0=np.array([orb['C'][j],orb['h'][j]])
        tasks.append((float(I),y0))
    def integrate(task):
        I,y0=task;sols=[]
        for k in range(3):
            if k==0:
                def rhs(t,y):return model.lr.rhs(y,I)
                initial=y0
            else:
                m=configs[k-1][2];mod=configs[k-1][1];initial=np.array([y0[0],y0[1]**m['gamma']])
                def rhs(t,y):return m['co']@np.r_[1.,y[0],y[1],I,np.maximum(y[0]-m['padded'],0.)]
            sol=solve_ivp(rhs,(0,a.duration),initial,t_eval=grid,method='DOP853',rtol=2e-9,atol=2e-11,max_step=.05)
            if not sol.success or len(sol.t)!=len(grid):raise ValueError(f'Integration failed I={I} model={k}: {sol.message}')
            C=sol.y[0];tail=C[len(C)//2:];peaks,_=find_peaks(tail,prominence=.005);mins,_=find_peaks(-tail,prominence=.005)
            gate=sol.y[1];summary=dict(Cmin=float(tail.min()),Cmax=float(tail.max()),peaks=int(len(peaks)),period=float(np.median(np.diff(grid[len(C)//2:][peaks]))) if len(peaks)>=2 else None,nfev=int(sol.nfev),gate_min=float(gate.min()),gate_max=float(gate.max()),gate_physical=bool(np.all((gate>=0)&(gate<=1))))
            sols.append((C,summary))
        print(f'I={I:.6f} '+ ' '.join(f'{name}: peaks={q[1]["peaks"]} max={q[1]["Cmax"]:.4f} T={q[1]["period"]}' for name,q in zip(('LR','identity','power'),sols)),flush=True)
        return I,sols
    with ThreadPoolExecutor(max_workers=a.workers) as pool:out=list(pool.map(integrate,tasks))
    result=dict(case=model.case,method='DOP853 independent fixed-input forward trajectories from the same physical initial state; peaks in second half. Initial state taken from nearest LR training orbit. Not a bifurcation diagram or a cycle completeness test.',settings=dict(duration=a.duration,sample=a.sample,inputs_source='explicit' if a.inputs else 'training'),models=[str(a.identity),str(a.power)],rows=[dict(I=I,LR=s[0][1],identity=s[1][1],power=s[2][1]) for I,s in out])
    (a.out/'trajectory_comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,which in zip(axes.flat,np.linspace(0,len(out)-1,4,dtype=int)):
        I,sols=out[which]
        for (C,_),color,label in zip(sols,('tab:purple','black','tab:orange'),('LR','identity','power h')):ax.plot(grid,C,color=color,label=label,lw=1)
        ax.set(title=f'I={I:.4f} µM',xlabel='Time (s)',ylabel='C (µM)');ax.grid(alpha=.2)
    axes.flat[0].legend();fig.suptitle(f'{model.case.upper()}: physical calcium, fixed input; same initial state')
    fig.savefig(a.out/'trajectory_comparison.png',dpi=150);plt.close(fig)
    print(f'Saved {a.out}',flush=True)

if __name__=='__main__':main()
