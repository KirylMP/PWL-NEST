#!/usr/bin/env python3
"""LR spectral transitions and equilibrium types as K_ER varies.

Physical calcium C parametrizes the equilibrium locus. At fixed C,K, the
calcium balance is monotone in nonnegative I for these LR parameters, so a
bracketed scalar solve includes folded I(C) branches. Finite-grid searches do
not certify the absence of tangential or closely spaced transitions.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

from sweep_K_zeros import ROOT, CMAX, gate, calcium_field, jacobian, classify


def equilibrium_at_C(C,K,p,I_min,I_max):
    def balance(I):return calcium_field(C,gate(C,I,p),I,K,p)
    lo,hi=balance(I_min),balance(I_max)
    if lo==0:I=I_min
    elif hi==0:I=I_max
    elif lo*hi>0:return None
    else:I=brentq(balance,I_min,I_max,xtol=1e-12)
    h=gate(C,I,p);tr,det,disc,kind=classify(jacobian(C,h,I,K,p))
    return dict(C=float(C),I=float(I),h=float(h),trace=tr,determinant=det,
                discriminant=disc,type=kind)


def transitions(K,p,I_min,I_max,c_grid):
    values=[equilibrium_at_C(float(C),K,p,I_min,I_max) for C in c_grid]
    found=[]
    for field,label in [('trace','trace_zero'),('determinant','det_zero'),('discriminant','disc_zero')]:
        def value(C):
            point=equilibrium_at_C(C,K,p,I_min,I_max)
            if point is None:raise ValueError('outside physical equilibrium branch')
            return point[field]
        for j,(a,b) in enumerate(zip(values[:-1],values[1:])):
            if a is None or b is None:continue
            if a[field]*b[field]>0:continue
            try:C=brentq(value,float(c_grid[j]),float(c_grid[j+1]),xtol=1e-13)
            except ValueError:continue
            point=equilibrium_at_C(C,K,p,I_min,I_max)
            # Trace sign changes inside a saddle do not change equilibrium type.
            if field=='trace' and point['determinant']<=1e-9:continue
            if field=='discriminant' and point['determinant']<=1e-9:continue
            eps=min(1e-6,max(1e-9,(c_grid[j+1]-c_grid[j])*0.05))
            left=equilibrium_at_C(max(C-eps,float(c_grid[j])),K,p,I_min,I_max)
            right=equilibrium_at_C(min(C+eps,float(c_grid[j+1])),K,p,I_min,I_max)
            if left is None or right is None:continue
            if found and any(t['event']==label and abs(t['C']-C)<1e-8 for t in found):continue
            found.append(dict(K_ER=float(K),event=label,C=point['C'],I=point['I'],h=point['h'],
                left_type=left['type'],right_type=right['type'],trace=point['trace'],
                determinant=point['determinant'],discriminant=point['discriminant']))
    found.sort(key=lambda v:v['C'])
    # Regions are derived only over contiguous sampled C intervals with a
    # physical I in range; no extrapolation over missing equilibrium branches.
    segments=[];start=None
    for j,v in enumerate(values+[None]):
        if v is not None and start is None:start=j
        if v is None and start is not None:
            end=j-1
            if end>start:
                lo,hi=float(c_grid[start]),float(c_grid[end])
                cuts=[lo]+[q['C'] for q in found if lo<q['C']<hi]+[hi]
                for a,b in zip(cuts[:-1],cuts[1:]):
                    mid=equilibrium_at_C((a+b)/2,K,p,I_min,I_max)
                    if mid is not None:segments.append(dict(K_ER=float(K),C_min=a,C_max=b,type=mid['type'],I_at_mid=mid['I']))
            start=None
    return found,segments


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',type=Path,default=ROOT/'results'/'equilibrium_types_K_sweep')
    parser.add_argument('--k-min',type=float,default=.025);parser.add_argument('--k-max',type=float,default=.2)
    parser.add_argument('--n-k',type=int,default=36);parser.add_argument('--n-c',type=int,default=1201)
    parser.add_argument('--i-min',type=float,default=0.);parser.add_argument('--i-max',type=float,default=2.)
    a=parser.parse_args()
    if a.n_k<2 or a.n_c<20 or not (0<a.k_min<a.k_max and 0<=a.i_min<a.i_max):parser.error('invalid sweep range or grid size')
    p=json.loads((ROOT.parent/'fm_am_reduction/config/cases.json').read_text())['shared_parameters']
    kval=np.unique(np.r_[np.linspace(a.k_min,a.k_max,a.n_k),[k for k in [.05,.1] if a.k_min<=k<=a.k_max]])
    cgrid=np.unique(np.r_[np.geomspace(1e-6,.1,max(120,a.n_c//3)),np.linspace(.1,CMAX-1e-8,a.n_c)])
    all_events=[];all_segments=[];counts=[]
    for K in kval:
        ev,seg=transitions(float(K),p,a.i_min,a.i_max,cgrid)
        counts.append(len(ev));all_events.extend(ev);all_segments.extend(seg)
        print(f'K_ER={K:.6f}: {len(ev)} spectral transitions; types={[s["type"] for s in seg]}',flush=True)
    a.out_dir.mkdir(parents=True,exist_ok=True)
    for name,rows,fields in [('thresholds.csv',all_events,['K_ER','event','C','I','h','left_type','right_type','trace','determinant','discriminant']),
                             ('region_types.csv',all_segments,['K_ER','C_min','C_max','type','I_at_mid'])]:
        with (a.out_dir/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(rows)
    summary=dict(K_range=[float(kval[0]),float(kval[-1])],I_range=[a.i_min,a.i_max],
                 n_K=len(kval),n_C=len(cgrid),spectral_transition_counts=[dict(K_ER=float(k),count=int(n)) for k,n in zip(kval,counts)],
                 definition='type-changing trace/determinant/discriminant roots on physical LR equilibria',
                 caveat='Finite-grid bracketing may miss tangencies or close transitions. These are LR spectral transitions, not a proof of minimal PWA threshold count.')
    (a.out_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    types=['stable_node','stable_focus','saddle','unstable_node','unstable_focus','nonhyperbolic']
    colors=['#276bba','#30a5b2','#e9a33a','#d95d54','#953f96','#888888']
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    for event,color,label in [('trace_zero','#111111','trace = 0'),('det_zero','#d97706','det = 0'),('disc_zero','#7e22ce','discriminant = 0')]:
        q=[r for r in all_events if r['event']==event]
        axes[0].scatter([r['K_ER'] for r in q],[r['C'] for r in q],s=12,c=color,label=label)
    axes[0].set(xlabel='K_ER',ylabel='C at equilibrium transition (µM)',title='LR spectral thresholds');axes[0].legend(fontsize=8)
    axes[1].step(kval,counts,where='mid');axes[1].set(xlabel='K_ER',ylabel='Number of spectral transitions',title='Transition count on sampled C branch')
    cmax=max((s['C_max'] for s in all_segments),default=.8);grid=np.full((300,len(kval)),np.nan);cc=np.linspace(0,cmax,300)
    for col,K in enumerate(kval):
        for seg in (s for s in all_segments if s['K_ER']==K):
            grid[(cc>=seg['C_min'])&(cc<=seg['C_max']),col]=types.index(seg['type'])
    cmap=ListedColormap(colors);cmap.set_bad('white')
    axes[2].pcolormesh(kval,cc,grid,shading='nearest',cmap=cmap,norm=BoundaryNorm(np.arange(-.5,6.5),cmap.N))
    axes[2].set(xlabel='K_ER',ylabel='C on LR equilibrium branch (µM)',title='Equilibrium type by C and K_ER')
    axes[2].legend(handles=[Patch(color=c,label=t.replace('_',' ')) for c,t in zip(colors,types)],fontsize=7,loc='upper left')
    for ax in axes:
        ax.axvline(.05,color='grey',ls=':',lw=.8);ax.axvline(.1,color='grey',ls='--',lw=.8);ax.set_xlim(kval[0],kval[-1]);ax.grid(alpha=.15)
    fig.savefig(a.out_dir/'thresholds_and_types_vs_K.png',dpi=180)
    fig.savefig(a.out_dir/'thresholds_and_types_vs_K.pdf');plt.close(fig)
    print(f'Results: {a.out_dir}',flush=True)


if __name__=='__main__':main()
