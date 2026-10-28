#!/usr/bin/env python3
"""Compare the detected lower boundary of large LR and FM PWA cycles."""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/fm_am_minimal_zones/rest_refinement'))
from model import Model
from run import json_numpy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('model_dir',type=Path)
    p.add_argument('--start',type=float,default=.48);p.add_argument('--stop',type=float,default=.532)
    p.add_argument('--points',type=int,default=105);p.add_argument('--seeds',type=int,default=64)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--bisections',type=int,default=12)
    p.add_argument('--large-min',type=float,default=.5,help='Minimum physical Cmax for the large-cycle branch')
    a=p.parse_args()
    if a.start>=a.stop or a.points<3 or a.seeds<4 or a.workers<1 or a.bisections<0:p.error('Invalid scan settings')
    candidate=a.model_dir/'parameters.json'
    if not candidate.exists():candidate=a.model_dir/'candidate.json'
    data=json.loads(candidate.read_text())
    if data.get('case')!='fm':raise ValueError('Expected FM candidate')
    model=Model('fm',data.get('coordinate_family','mixed'),data['construction'],data.get('fm_lower_type','node'))
    x=model.constrain_events(data['parameters']);m=model.unpack(x);f=model.f
    from lr_reference import rhs as lr_rhs,jac as lr_jac
    bounds=f.phi(np.array([1e-12,f.CMAX]),m['shape'])
    def scan(I):
        I=float(I);le=f.eqs(I);pe=model.equilibrium(I,m)
        lc,ld=f.cycles(I,le,n=a.seeds,tol=1e-9)
        pc,pd=f.cycles(I,pe,m['co'],m['padded'],bounds,n=a.seeds,tol=1e-9)
        for q in lc:q['Cmin'],q['Cmax']=float(q['min']),float(q['max'])
        for q in pc:q['Cmin'],q['Cmax']=map(float,f.inverse(np.array([q['min'],q['max']]),m))
        return dict(I=I,lr=lc,pwa=pc,lr_diagnostics=ld,pwa_diagnostics=pd)
    def large(row,key):return [q for q in row[key] if q['stable'] and q['Cmax']>=a.large_min]
    rows=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for k,row in enumerate(pool.map(scan,np.linspace(a.start,a.stop,a.points)),1):
            rows.append(row)
            if k==1 or k%5==0 or k==a.points:print(f'ONSET {k}/{a.points} I={row["I"]:.8f} LR_large={len(large(row,"lr"))} PWA_large={len(large(row,"pwa"))}',flush=True)
    brackets={};refinement={};nearby={}
    def orbit_check(I,q,key):
        if key=='lr':
            def rhs(t,y):return lr_rhs(y,I)
            def physical(y):return np.asarray(y[0]),np.asarray(y[1])
            saddles=[]
            for e in f.eqs(I):
                J=lr_jac(e,I)
                if np.linalg.det(J)<0:saddles.append(np.asarray(e))
        else:
            def rhs(t,y):return m['co']@np.r_[1.,y[0],y[1],I,np.maximum(y[0]-m['padded'],0.)]
            def physical(y):return f.inverse(np.asarray(y[0]),m),np.asarray(y[1])
            saddles=[np.array([e['C'],e['z'][1]]) for e in model.equilibrium(I,m) if m['D'][e['region']]<0]
        y0=np.array([q['section'],q['s']]);T=float(q['T'])
        sol=solve_ivp(rhs,(0,T),y0,method='DOP853',rtol=2e-10,atol=2e-12,max_step=.03,t_eval=np.linspace(0,T,1500))
        if not sol.success or len(sol.t)<2:return dict(success=False,message=sol.message)
        c,h=physical(sol.y);c0,h0=physical(y0);close=float(np.hypot(c[-1]-c0,h[-1]-h0))
        distances=[]
        for eq in saddles:
            dd=np.hypot(c-eq[0],h-eq[1]);j=int(np.argmin(dd))
            distances.append(dict(saddle_C=float(eq[0]),saddle_h=float(eq[1]),sampled_distance=float(dd[j]),dC=float(c[j]-eq[0]),dh=float(h[j]-eq[1])))
        return dict(success=True,closure_physical_norm=close,closure_passed=bool(close<1e-4),sampled_saddle_distance=distances)
    for key in ('lr','pwa'):
        transitions=[(l,r) for l,r in zip(rows[:-1],rows[1:]) if not large(l,key) and large(r,key)]
        if not transitions:
            brackets[key]=None;refinement[key]=[]
            print(f'ONSET {key}: no absent-to-present bracket in scan; boundary may be outside range or missed',flush=True)
            continue
        l,r=transitions[0];history=[];present=[r]
        for j in range(a.bisections):
            mid=scan((l['I']+r['I'])/2);history.append(mid)
            if large(mid,key):r=mid;present.append(mid)
            else:l=mid
            print(f'ONSET {key} refine {j+1}/{a.bisections}: interval=[{l["I"]:.10f},{r["I"]:.10f}]',flush=True)
        brackets[key]=dict(no_cycle_detected_at=l['I'],cycle_detected_at=r['I'],scope='finite seed search; not a rigorous mathematical bracket')
        refinement[key]=history
        candidates=sorted(present,key=lambda row:row['I'])[:3];nearby[key]=[]
        for row in candidates:
            q=max(large(row,key),key=lambda v:v['Cmax']-v['Cmin'])
            check=orbit_check(row['I'],q,key)
            nearby[key].append(dict(I=row['I'],period=q['T'],Cmin=q['Cmin'],Cmax=q['Cmax'],log_multiplier=q['log_multiplier'],independent_check=check))
    result=dict(candidate=str(candidate.resolve()),candidate_stage=data.get('stage'),settings=vars(a)|{'model_dir':str(a.model_dir)},
        method='Finite input grid and forward/backward Poincare seed search, then conditional bisection. Non-detection is not proof of absence; independent integration is performed only for sampled detected cycles.',
        brackets=brackets,refinement=refinement,nearby=nearby,rows=rows)
    out=a.model_dir/'onset_boundary.json';out.write_text(json.dumps(result,indent=2,default=json_numpy)+'\n')
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for key,color in [('lr','tab:purple'),('pwa','black')]:
        for row in rows:
            for q in row[key]:
                kw=dict(s=12,edgecolors=color,facecolors=color if q['stable'] else 'none',linewidths=.6)
                axes[0].scatter([row['I']]*2,[q['Cmin'],q['Cmax']],**kw);axes[1].scatter(row['I'],q['T'],**kw)
        axes[2].step([r['I'] for r in rows],[len(large(r,key)) for r in rows],where='mid',color=color,label=key.upper())
        if brackets[key] is not None:
            for ax in axes:ax.axvspan(brackets[key]['no_cycle_detected_at'],brackets[key]['cycle_detected_at'],color=color,alpha=.12)
    axes[0].set_ylabel('Cycle extrema C (µM)');axes[1].set_ylabel('Period (s)');axes[2].set_ylabel('Large stable cycles found');axes[2].legend()
    for ax in axes:ax.set_xlabel('I (µM)');ax.grid(alpha=.2)
    fig.suptitle('Lower onset: finite detection, filled stable / open unstable')
    image=a.model_dir/'onset_boundary.png';fig.savefig(image,dpi=160);plt.close(fig)
    print(f'Saved {out}\nSaved {image}\nBrackets: {json.dumps(brackets)}',flush=True)


if __name__=='__main__':main()
