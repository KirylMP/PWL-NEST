#!/usr/bin/env python3
"""Search for a saddle-loop candidate by matching saddle manifolds on a section."""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/fm_am_minimal_zones/rest_refinement'))
from model import Model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('model_dir',type=Path);p.add_argument('--start',type=float,default=.515);p.add_argument('--stop',type=float,default=.531)
    p.add_argument('--points',type=int,default=33);p.add_argument('--workers',type=int,default=8);p.add_argument('--section',type=float,default=.2)
    p.add_argument('--epsilon',type=float,default=1e-6);p.add_argument('--max-time',type=float,default=600.)
    p.add_argument('--unstable-branch',choices=['left','right'],default='left',help='Initial calcium direction of the outgoing saddle branch')
    p.add_argument('--output-prefix',default='separatrix_onset',help='Output filename stem within the model directory')
    a=p.parse_args()
    if a.start>=a.stop or a.points<3 or a.workers<1 or a.epsilon<=0 or a.max_time<=0 or not .14<a.section<1.:p.error('Invalid arguments')
    if not a.output_prefix or Path(a.output_prefix).name!=a.output_prefix:p.error('output-prefix must be a simple filename')
    candidate=a.model_dir/'parameters.json'
    if not candidate.exists():candidate=a.model_dir/'candidate.json'
    data=json.loads(candidate.read_text())
    if data.get('case')!='fm':raise ValueError('Expected FM candidate')
    model=Model('fm',data.get('coordinate_family','mixed'),data['construction'],data.get('fm_lower_type','node'))
    x=model.constrain_events(data['parameters']);m=model.unpack(x);f=model.f
    from lr_reference import rhs as lr_rhs,jac as lr_jac
    def calculate(I,key,epsilon):
        I=float(I)
        if key=='lr':
            eqs=[(np.array(e),np.array(e),lr_jac(e,I)) for e in f.eqs(I) if np.linalg.det(lr_jac(e,I))<0]
            def rhs(t,y):return lr_rhs(y,I)
            section=a.section;lo=1e-10;hi=f.CMAX-1e-9
        else:
            eqs=[]
            for e in model.equilibrium(I,m):
                r=e['region'];co=m['co'];A=np.column_stack((co[:,1]+co[:,4:4+r].sum(axis=1),co[:,2]))
                if np.linalg.det(A)<0:eqs.append((np.asarray(e['z']),np.array([e['C'],e['z'][1]]),A))
            def rhs(t,y):return m['co']@np.r_[1.,y[0],y[1],I,np.maximum(y[0]-m['padded'],0.)]
            section=float(f.phi(a.section,m['shape']));lo=float(f.phi(1e-10,m['shape']));hi=float(f.phi(f.CMAX-1e-9,m['shape']))
        if len(eqs)!=1:return dict(I=I,key=key,ok=False,reason=f'Expected one saddle; found {len(eqs)}')
        state,physical,A=eqs[0];eigs,vecs=np.linalg.eig(A)
        if np.any(np.abs(eigs.imag)>1e-9) or np.sum(eigs.real>0)!=1 or np.sum(eigs.real<0)!=1:return dict(I=I,key=key,ok=False,reason='Saddle eigenvalues invalid')
        def crossing(t,y):return y[0]-section
        crossing.terminal=True;crossing.direction=0
        def domain(t,y):return min(y[0]-lo,hi-y[0],y[1],1-y[1])
        domain.terminal=True;domain.direction=0
        hits={};detail=[]
        for direction,label in [(1,'unstable'),(-1,'stable')]:
            index=int(np.argmax(eigs.real) if direction==1 else np.argmin(eigs.real));v=np.real(vecs[:,index]);v/=np.linalg.norm(v)
            hits[label]=[]
            for sign in (-1,1):
                sol=solve_ivp(rhs,(0,direction*a.max_time),state+sign*epsilon*v,method='DOP853',rtol=1e-9,atol=1e-11,max_step=.1,events=(crossing,domain))
                if len(sol.t_events[0]):
                    y=sol.y_events[0][0];hits[label].append(dict(h=float(y[1]),sign=sign,time=float(sol.t_events[0][0]),initial_C_direction='left' if sign*v[0]<0 else 'right'))
                final=sol.y[:,-1];final_c=float(f.inverse(np.asarray(final[0]),m)) if key=='pwa' else float(final[0])
                detail.append(dict(manifold=label,sign=sign,crossed=bool(len(sol.t_events[0])),domain_exit=bool(len(sol.t_events[1])),solver_success=bool(sol.success),
                    reached_time_limit=bool(sol.success and not len(sol.t_events[0]) and not len(sol.t_events[1]) and abs(sol.t[-1])>=a.max_time-1e-7),
                    final_time=float(sol.t[-1]),final_C=final_c,final_h=float(final[1]),message=sol.message))
        out=dict(I=I,key=key,epsilon=epsilon,saddle_C=float(physical[0]),saddle_h=float(physical[1]),eigenvalues=[float(z.real) for z in eigs],hits=hits,details=detail)
        outgoing=[q for q in hits['unstable'] if q['initial_C_direction']==a.unstable_branch]
        out['ok']=len(outgoing)==1 and len(hits['stable'])==1
        if out['ok']:
            out['selected_unstable']=outgoing[0];out['selected_stable']=hits['stable'][0]
            out['difference_h']=outgoing[0]['h']-hits['stable'][0]['h']
        else:out['reason']='Section crossings missing or ambiguous'
        return out
    grid=np.linspace(a.start,a.stop,a.points);rows=[]
    def pair(I):return dict(I=float(I),lr=calculate(I,'lr',a.epsilon),pwa=calculate(I,'pwa',a.epsilon))
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for j,row in enumerate(pool.map(pair,grid),1):
            rows.append(row)
            if j==1 or j%4==0 or j==len(grid):print(f'SEPARATRIX {j}/{len(grid)} I={row["I"]:.8f} LR={row["lr"].get("difference_h")} PWA={row["pwa"].get("difference_h")}',flush=True)
    roots={}
    for key in ('lr','pwa'):
        roots[key]=[]
        for left,right in zip(rows[:-1],rows[1:]):
            l,r=left[key],right[key]
            if not(l['ok'] and r['ok']) or l['difference_h']*r['difference_h']>0:continue
            if (l['selected_unstable']['sign'],l['selected_stable']['sign'])!=(r['selected_unstable']['sign'],r['selected_stable']['sign']):continue
            signs=(l['selected_unstable']['sign'],l['selected_stable']['sign'])
            def fun(I):
                v=calculate(I,key,a.epsilon)
                if not v['ok'] or (v['selected_unstable']['sign'],v['selected_stable']['sign'])!=signs:raise ValueError('Manifold section itinerary changed')
                return v['difference_h']
            try:
                root=float(brentq(fun,l['I'],r['I'],xtol=1e-9));check=calculate(root,key,a.epsilon)
                smaller=calculate(root,key,a.epsilon/4)
                roots[key].append(dict(I=root,main=check,smaller_epsilon=smaller,scope='Finite-offset section matching candidate; not an independent proof of a homoclinic bifurcation'))
                print(f'SEPARATRIX {key} candidate I={root:.10f} mismatch={check.get("difference_h")} smaller_eps={smaller.get("difference_h")}',flush=True)
            except (ValueError,RuntimeError) as exc:print(f'SEPARATRIX {key} root unresolved: {exc}',flush=True)
    output=dict(candidate=str(candidate.resolve()),settings=vars(a)|{'model_dir':str(a.model_dir)},
        method='Forward unstable and backward stable saddle eigenvector branches, finite offset, first crossing of chosen vertical C section. Match heights; scan and finite offset can miss or misidentify connections.',rows=rows,roots=roots)
    dest=a.model_dir/(a.output_prefix+'.json');dest.write_text(json.dumps(output,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    for key,color in [('lr','tab:purple'),('pwa','black')]:
        ax.plot([r['I'] for r in rows],[r[key].get('difference_h',np.nan) for r in rows],'.-',color=color,label=key.upper())
        for root in roots[key]:ax.axvline(root['I'],color=color,ls=':',alpha=.5)
    if not any(row[key]['ok'] for row in rows for key in ('lr','pwa')):
        ax.text(.5,.5,'No unambiguous section crossings found.\nSee JSON branch diagnostics.',transform=ax.transAxes,ha='center',va='center')
        ax.set_xlim(a.start,a.stop)
    ax.axhline(0,color='gray',lw=.8);ax.set(xlabel='I (µM)',ylabel='h unstable minus h stable at section',title=f'Saddle manifold matching, C={a.section:g} µM')
    ax.grid(alpha=.25);ax.legend();image=a.model_dir/(a.output_prefix+'.png');fig.savefig(image,dpi=160);plt.close(fig)
    print(f'Saved {dest}\nSaved {image}')


if __name__=='__main__':main()
