#!/usr/bin/env python3
"""Compare LR and PWA phase portraits near FM onset using saved cycle scans."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/fm_am_minimal_zones/rest_refinement'))
from model import Model


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('model_dir',type=Path)
    ap.add_argument('--inputs',type=float,nargs='+',default=[.525,.528,.5288,.5292])
    ap.add_argument('--manifold-time',type=float,default=60.)
    args=ap.parse_args()
    if args.manifold_time<=0:ap.error('manifold-time must be positive')
    p=args.model_dir/'parameters.json'
    if not p.is_file():p=args.model_dir/'candidate.json'
    data=json.loads(p.read_text());scan_path=args.model_dir/'lower_bifurcation.json'
    if not scan_path.is_file():raise ValueError('lower_bifurcation.json is missing; run analyze_lower_fm.py first')
    rows=json.loads(scan_path.read_text())['rows']
    if not rows:raise ValueError('Lower scan has no rows')
    model=Model('fm',data.get('coordinate_family','mixed'),data['construction'],data.get('fm_lower_type','node'))
    x=model.constrain_events(data['parameters']);m=model.unpack(x);f=model.f
    from lr_reference import rhs as lr_rhs, jac as lr_jac
    from geometry_reference import truth
    chosen=[min(rows,key=lambda q:abs(q['I']-I)) for I in args.inputs]
    if len({q['I'] for q in chosen})<len(chosen):raise ValueError('Requested inputs map to duplicate scan rows')
    fig,axes=plt.subplots(len(chosen),2,figsize=(14,4.1*len(chosen)),squeeze=False,layout='constrained')
    report=[]
    for row,axs in zip(chosen,axes):
        I=float(row['I']);line=dict(I=I,models={});cs=np.linspace(.005,1.16,600)
        for key,ax in zip(('lr','pwa'),axs):
            if key=='lr':
                def rhs(t,y):return lr_rhs(y,I)
                def physical(y):return np.asarray(y[0]),np.asarray(y[1])
                nulls=truth(cs,I);eqs=[]
                for e in f.eqs(I):
                    matrix=lr_jac(e,I);eqs.append((np.asarray(e),np.asarray(e),matrix))
            else:
                def rhs(t,y):return m['co']@np.r_[1.,y[0],y[1],I,np.maximum(y[0]-m['padded'],0.)]
                def physical(y):return f.inverse(np.asarray(y[0]),m),np.asarray(y[1])
                basis=model.basis(cs,I,m);nulls=np.stack([basis@m['N'],basis@m['H']],axis=-1);eqs=[]
                for e in model.equilibrium(I,m):
                    region=e['region'];co=m['co'];matrix=np.column_stack((co[:,1]+co[:,4:4+region].sum(axis=1),co[:,2]))
                    eqs.append((np.asarray(e['z']),np.array([e['C'],e['z'][1]]),matrix))
            ax.plot(cs,nulls[:,0],color='tab:orange',lw=1,label='dC/dt = 0')
            ax.plot(cs,nulls[:,1],color='tab:green',lw=1,label='dh/dt = 0')
            eq_report=[];saddles=[]
            for state,physical_eq,A in eqs:
                eig,vec=np.linalg.eig(A);det=float(np.linalg.det(A));trace=float(np.trace(A))
                typ='saddle' if det<0 else ('stable' if trace<0 else 'unstable')
                ax.scatter(*physical_eq,marker='x' if typ=='saddle' else 'o',s=42,color='tab:red' if typ=='saddle' else ('tab:blue' if typ=='stable' else 'tab:orange'),zorder=5)
                eq_report.append(dict(C=float(physical_eq[0]),h=float(physical_eq[1]),type=typ,trace=trace,det=det,eigenvalues=[[float(z.real),float(z.imag)] for z in eig]))
                if typ=='saddle':saddles.append((state,physical_eq,eig,vec))
            lower=float(f.phi(np.array([1e-8]),m['shape'])[0]) if key=='pwa' else 1e-8
            upper=float(f.phi(np.array([f.CMAX-1e-8]),m['shape'])[0]) if key=='pwa' else f.CMAX-1e-8
            def domain(t,y):return min(y[0]-lower,upper-y[0],y[1],1-y[1])
            domain.terminal=True;domain.direction=0
            for state,peq,eig,vec in saddles:
                for k,lam in enumerate(eig):
                    if abs(lam.imag)>1e-8 or abs(lam.real)<1e-10:continue
                    direction=1 if lam.real>0 else -1
                    v=np.real(vec[:,k]);v=v/np.linalg.norm(v)
                    for sign in (-1,1):
                        sol=solve_ivp(rhs,(0,direction*args.manifold_time),state+sign*1e-6*v,method='DOP853',rtol=1e-8,atol=1e-10,max_step=.1,events=domain)
                        c,h=physical(sol.y);ax.plot(c,h,color='tab:red',alpha=.42,lw=.75,ls='-' if direction>0 else ':')
            cycle_report=[]
            for cyc in row.get(key,[]):
                y0=np.array([cyc['section'],cyc['s']],float);period=float(cyc['T'])
                sol=solve_ivp(rhs,(0,period),y0,method='DOP853',rtol=2e-10,atol=2e-12,max_step=.03,t_eval=np.linspace(0,period,1800),events=domain)
                c,h=physical(sol.y);c0,h0=physical(y0);closure=float(np.linalg.norm(np.array([c[-1]-c0,h[-1]-h0])))
                verified=bool(sol.success and len(sol.t)>1 and abs(sol.t[-1]-period)<1e-7 and closure<1e-4)
                ax.plot(c,h,color='tab:purple' if key=='lr' else 'black',lw=1.5,ls='-' if cyc['stable'] else '--',alpha=1 if verified else .3)
                near=[]
                for _,peq,_,_ in saddles:
                    dist=np.hypot(c-peq[0],h-peq[1]);j=int(np.argmin(dist))
                    near.append(dict(sampled_distance=float(dist[j]),dC=float(c[j]-peq[0]),dh=float(h[j]-peq[1])))
                cycle_report.append(dict(stable=bool(cyc['stable']),period=period,closure_physical_norm=closure,independent_closure_passed=verified,closest_saddle=near))
            ax.set(xlim=(0,1.16),ylim=(.55,1.01),xlabel='C (µM)',ylabel='h',title=f'{key.upper()}  I={I:.7f}')
            ax.grid(alpha=.15)
            line['models'][key]=dict(equilibria=eq_report,cycles=cycle_report,scan_diagnostics=row.get(key+'_diagnostics'))
        report.append(line)
        print(f'PHASE I={I:.7f}: LR={len(row.get("lr",[]))} PWA={len(row.get("pwa",[]))}',flush=True)
    handles=[Line2D([0],[0],color='tab:orange',label='C nullcline'),Line2D([0],[0],color='tab:green',label='h nullcline'),
             Line2D([0],[0],color='black',ls='-',label='Stable cycle'),Line2D([0],[0],color='black',ls='--',label='Unstable cycle'),
             Line2D([0],[0],color='tab:red',ls='-',label='Saddle unstable manifold'),Line2D([0],[0],color='tab:red',ls=':',label='Saddle stable manifold'),
             Line2D([0],[0],color='tab:red',marker='x',ls='none',label='Saddle')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,-.02),ncol=4,fontsize=9)
    fig.suptitle(f'LR and PWA phase portraits, {data["construction"]} ({data.get("stage")})')
    out=args.model_dir/'phase_near_onset.png';fig.savefig(out,dpi=150,bbox_inches='tight');plt.close(fig)
    dest=args.model_dir/'phase_near_onset.json'
    dest.write_text(json.dumps(dict(candidate=str(p.resolve()),scan=str(scan_path.resolve()),
        method='Stored finite-scan cycles integrated independently; saddle manifolds initialized by eigenvectors. Faint cycles failed the independent closure check. Distance uses sampled Euclidean coordinates C,h and is descriptive, not a separatrix proof.',rows=report),indent=2)+'\n')
    print(f'Saved {out}\nSaved {dest}')


if __name__=='__main__':main()
