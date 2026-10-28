#!/usr/bin/env python3
"""Overlay LR and PWA equilibrium branches, with optional archived cycles."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/fm_am_minimal_zones/rest_refinement'))
from model import Model


def classify(trace,det):
    if det < -1e-10:return 'saddle'
    if det > 1e-10 and trace < -1e-10:return 'stable'
    if det > 1e-10 and trace > 1e-10:return 'unstable'
    return 'nonhyperbolic'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('model_dir',type=Path,help='Catalog model directory, or fit/fm run directory')
    p.add_argument('--samples',type=int,default=4000)
    p.add_argument('--cmax',type=float,default=.8)
    args=p.parse_args()
    if args.samples<100 or not .2<args.cmax<2/1.185:p.error('Use samples >= 100 and 0.2 < cmax < 2/1.185')
    source=args.model_dir/'parameters.json'
    if not source.exists():source=args.model_dir/'candidate.json'
    data=json.loads(source.read_text())
    if data.get('case')!='fm':raise ValueError('Expected FM candidate')
    construction=data.get('construction')
    if construction is None:
        count=len(data['physical_thresholds_C'])
        construction={3:'base',7:'full_fixed'}.get(count)
    if construction is None:raise ValueError('Candidate lacks a recognized construction')
    model=Model('fm',data.get('coordinate_family','mixed'),construction,data.get('fm_lower_type','node'))
    x=model.constrain_events(data['parameters']);m=model.unpack(x);f=model.f
    Cs=np.unique(np.r_[np.linspace(max(1e-5,float(model.rest[0])),args.cmax,args.samples),m['tc']])
    rows={'lr':[],'pwa':[]}
    for c in Cs:
        c=float(c)
        try:
            I=float(f.input_at_C(c));h=float(f.gate(c,I));J=f.jac([c,h],I)
            trace=float(np.trace(J));det=float(np.linalg.det(J))
            if 0<=I<=2 and 0<h<1:rows['lr'].append(dict(C=c,I=I,h=h,trace=trace,det=det,type=classify(trace,det)))
        except ValueError:pass
        basis0=model.basis(c,0.,m);diff=float(basis0@(m['N']-m['H']));slope=float(m['N'][2]-m['H'][2])
        if abs(slope)<1e-14:continue
        I=-diff/slope;h=float(model.basis(c,I,m)@m['N']);r=int(np.searchsorted(m['tc'],c,side='right'))
        trace=float(m['T'][r]);det=float(m['D'][r])
        if 0<=I<=2 and 0<h<1:rows['pwa'].append(dict(C=c,I=float(I),h=h,trace=trace,det=det,type=classify(trace,det),region=r+1))
    cycles=[]
    for filename in ('branches.json','candidate_branches.json','lower_bifurcation.json'):
        q=args.model_dir/filename
        if not q.exists():continue
        for row in json.loads(q.read_text()).get('rows',[]):
            for v in row.get('cycles',row.get('pwa',[])):
                if 'Cmin' in v and 'Cmax' in v:cycles.append((float(row['I']),float(v['Cmin']),float(v['Cmax']),bool(v['stable'])))
    ref_path=f.DATA_DIR/'LR_cycles_reference.json'
    lr_cycles=[]
    for row in json.loads(ref_path.read_text())['rows']:
        for v in row['cycles']:lr_cycles.append((float(row['I']),float(v['min']),float(v['max']),bool(v['stable'])))
    fig,axes=plt.subplots(1,3,figsize=(17,5),layout='constrained')
    styles={'stable':'-','unstable':'--','saddle':':','nonhyperbolic':'none'}
    for key,color in [('lr','tab:purple'),('pwa','black')]:
        rr=rows[key]
        for typ,style in styles.items():
            if style=='none':continue
            xx=np.array([r['I'] for r in rr]);cc=np.array([r['C'] if r['type']==typ else np.nan for r in rr]);hh=np.array([r['h'] if r['type']==typ else np.nan for r in rr])
            axes[0].plot(xx,cc,color=color,ls=style,lw=1.8)
            axes[1].plot(xx,cc,color=color,ls=style,lw=2)
            axes[2].plot(xx,hh,color=color,ls=style,lw=2)
    for values,color in ((lr_cycles,'tab:purple'),(cycles,'black')):
        for I,cmin,cmax,stable in values:
            kw=dict(s=10,edgecolors=color,facecolors=color if stable else 'none',linewidths=.6,alpha=.6,zorder=2)
            axes[0].scatter([I,I],[cmin,cmax],**kw)
            axes[1].scatter([I,I],[cmin,cmax],**kw)
    for c in m['tc']:
        for ax in axes[:2]:ax.axhline(c,color='tab:cyan',ls='--',lw=.55,alpha=.45)
    for I in (0.5107261652060441,0.5290700389163682,0.5439919432478553,0.48230398233782223):
        axes[1].axvline(I,color='gray',ls=':',lw=.7,alpha=.55)
        axes[2].axvline(I,color='gray',ls=':',lw=.7,alpha=.55)
    axes[0].set(xlim=(.0,1.3),ylim=(0,1.25),title='Equilibria and cycle extrema',ylabel='C (µM)')
    axes[1].set(xlim=(.465,.56),ylim=(.02,.17),title='Lower transition: calcium',ylabel='C (µM)')
    axes[2].set(xlim=(.465,.56),ylim=(.65,1),title='Lower transition: h',ylabel='h')
    for ax in axes:ax.set_xlabel('I (µM)');ax.grid(alpha=.2)
    handles=[Line2D([0],[0],color='tab:purple',label='Li–Rinzel'),Line2D([0],[0],color='black',label='PWA'),
             Line2D([0],[0],color='gray',ls='-',label='Stable equilibrium'),Line2D([0],[0],color='gray',ls='--',label='Unstable equilibrium'),
             Line2D([0],[0],color='gray',ls=':',label='Saddle'),Line2D([0],[0],color='gray',marker='o',linestyle='none',label='Stable cycle extrema'),
             Line2D([0],[0],color='gray',marker='o',markerfacecolor='none',linestyle='none',label='Unstable cycle extrema')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,-.04),ncol=4,fontsize=9)
    fig.suptitle(f'FM equilibrium branches: LR and PWA ({construction}, stage={data.get("stage")})')
    out=args.model_dir/'equilibria_comparison.png';fig.savefig(out,dpi=160,bbox_inches='tight');plt.close(fig)
    result=dict(candidate=str(source.resolve()),construction=construction,stage=data.get('stage'),thresholds=m['tc'].tolist(),samples=args.samples,
                method='Equilibrium branches parameterized by physical C; PWA I solved from affine nullclines, LR I solved from LR equations. Cycles taken from existing finite scans; not a completeness proof.',
                pwa_cycle_points=len(cycles),lr_cycle_points=len(lr_cycles),rows=rows)
    path=args.model_dir/'equilibria_comparison.json';path.write_text(json.dumps(result,indent=2)+'\n')
    print(f'Candidate: {source}\nPlot: {out}\nData: {path}\nExisting cycle rows: PWA={len(cycles)} LR={len(lr_cycles)}')


if __name__=='__main__':main()
