"""Draft article figures from selected PWA branches and cached LR data."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/selected_lr_branches'
REF=ROOT/'paper/figures/figure1_data'
MODELS=ROOT/'paper/data/selected_lr_candidates'
CASES=('AM','AFM','FM')
XLIM={'AM':(.29,.70),'AFM':(.27,.98),'FM':(.46,1.20)}
CLIM={'AM':(0,.72),'AFM':(0,1.08),'FM':(0,1.25)}
TLIM={'AM':(0,30),'AFM':(0,60),'FM':(0,80)}


def equilibrium(d,I):
    c=np.array(d['physical_thresholds_C']);N=np.array(d['alpha']);H=np.array(d['beta']);a,b=d['a'],d['b'];out=[]
    for j in range(3):
        ns=N[1]+sum(N[3:3+j]);hs=H[1]+sum(H[3:3+j]);ni=N[0]+N[2]*I-N[3:3+j]@c[:j];hi=H[0]+H[2]*I-H[3:3+j]@c[:j]
        if abs(ns-hs)<1e-12:continue
        C=(hi-ni)/(ns-hs);lo=-np.inf if j==0 else c[j-1];up=np.inf if j==2 else c[j]
        if C<lo-1e-10 or C>up+1e-10 or C<0:continue
        tr=-a*ns-b;det=a*b*(ns-hs)
        out.append((float(C),'stable' if det>0 and tr<0 else 'saddle' if det<0 else 'unstable'))
    return out


def load():
    manifest=json.loads((MODELS/'manifest.json').read_text());models={}
    for c in CASES:
        e=manifest['models'][c];p=ROOT/e['candidate_path']
        if hashlib.sha256(p.read_bytes()).hexdigest()!=e['candidate_sha256']:raise ValueError('candidate mismatch')
        models[c]=json.loads(p.read_text())
    branches=json.loads((DATA/'branches.json').read_text())['results']
    extended=json.loads((DATA/'extended.json').read_text())['results']
    upper=json.loads((DATA/'upper_only.json').read_text())['results']
    fmextra=json.loads((DATA/'upper_fm_extended.json').read_text())
    if 'results' in fmextra:fmextra=next(v for v in fmextra['results'] if v['case']=='FM')
    forward=json.loads((DATA/'forward.json').read_text())['results']
    for item in branches+extended+upper+[fmextra]:
        if item['candidate_sha256']!=manifest['models'][item['case']]['candidate_sha256']:raise ValueError('branch mismatch')
    refs=list(csv.DictReader((REF/'cycles.csv').open()));meta=json.loads((REF/'metadata.json').read_text())
    return manifest,models,branches,extended,upper,fmextra,forward,refs,meta


def plot(kind,data):
    manifest,models,branches,extended,upper,fmextra,forward,refs,meta=data
    fig,axes=plt.subplots(2,3,figsize=(10,6.8),sharex='col',sharey='col')
    fig.subplots_adjust(left=.08,right=.985,top=.91,bottom=.16,wspace=.27,hspace=.19)
    for col,case in enumerate(CASES):
        eq=np.load(REF/f'{case}_equilibria.npz');ax=axes[0,col]
        if kind=='C':
            for stable,colr,style in ((True,'#245f9b','-'),(False,'#bf434a','--')):
                ax.plot(eq['I'],np.where(eq['stable']==stable,eq['C'],np.nan),c=colr,ls=style,lw=1.15)
        for stable in (True,False):
            rr=[v for v in refs if v['case']==case and v['displayed']=='True' and (v['stable']=='True')==stable]
            for key in (('C_min','C_max') if kind=='C' else ('period_s',)):
                ax.scatter([float(v['I']) for v in rr],[float(v[key]) for v in rr],s=6 if stable else 9,
                           facecolors='#202020' if stable else 'white',edgecolors='#202020' if stable else '#a32839',lw=.6,zorder=3)
        lr_fold=next(e for e in meta['events'] if e['case']==case and e['kind']=='cycle_fold')
        for y in ([lr_fold['C_min'],lr_fold['C_max']] if kind=='C' else [lr_fold['period_s']]):
            ax.scatter(lr_fold['I'],y,marker='D',s=19,c='#7354a4',zorder=5)
        ax.set_title(f'{case}: Li–Rinzel',fontsize=9)
        ax=axes[1,col];model=models[case]
        if kind=='C':
            grid=np.linspace(*XLIM[case],700);eqs=[equilibrium(model,float(i)) for i in grid]
            for state,colr,style in (('stable','#245f9b','-'),('unstable','#bf434a','--'),('saddle','#a15f16',':')):
                ax.plot(grid,[next((r[0] for r in entries if r[1]==state),np.nan) for entries in eqs],c=colr,ls=style,lw=1.15)
        # Use the longer stable continuation and the short second fold side.
        data_branches=[v for v in extended if v['case']==case]+[v for v in branches if v['case']==case and v['branch']['direction']==-1]
        pts=[p for entry in data_branches for p in entry['branch']['points'] if p['sampled_positive_C']]
        upper_entry=fmextra if case=='FM' else next(v for v in upper if v['case']==case)
        pts += [p for b in upper_entry['branches'] for p in b['points'] if p['sampled_positive_C']]
        for stable in (True,False):
            pp=[v for v in pts if v['stable']==stable]
            for key in (('Cmin','Cmax') if kind=='C' else ('period',)):
                ax.scatter([v['I'] for v in pp],[v[key] for v in pp],s=5 if stable else 8,
                           facecolors='#202020' if stable else 'white',edgecolors='#202020' if stable else '#a32839',lw=.55,zorder=4)
        # DOP853 observations give extra finite stable samples; not evidence of absence elsewhere.
        ff=[v for v in forward[case] if v['valid'] and v['stable']]
        for key in (('Cmin','Cmax') if kind=='C' else ('period',)):
            ax.scatter([v['I'] for v in ff],[v[key] for v in ff],marker='x',s=13,c='#777777',linewidths=.55,zorder=3)
        fold=next(v['neutral'] for v in branches if v['case']==case and 'neutral' in v)
        for y in ([fold['Cmin'],fold['Cmax']] if kind=='C' else [fold['period']]):ax.scatter(fold['I'],y,marker='D',s=19,c='#7354a4',zorder=6)
        ax.set_title(f'{case}: selected PWA',fontsize=9)
        for row in range(2):
            axes[row,col].set_xlim(*XLIM[case]);axes[row,col].set_ylim(*(CLIM if kind=='C' else TLIM)[case]);axes[row,col].grid(alpha=.12)
        axes[1,col].set_xlabel(r'$I$ ($\mu$M)',fontsize=8)
    for row in range(2):axes[row,0].set_ylabel(r'$C$ ($\mu$M)' if kind=='C' else 'Period (s)',fontsize=8)
    legend=[Line2D([],[],color='#245f9b',label='stable equilibrium'),Line2D([],[],color='#bf434a',ls='--',label='unstable equilibrium'),
            Line2D([],[],marker='o',ls='',color='#202020',ms=4,label='stable cycle'),Line2D([],[],marker='o',ls='',color='#a32839',mfc='white',ms=4,label='unstable cycle'),
            Line2D([],[],marker='D',ls='',color='#7354a4',ms=4,label='cycle fold'),
            Line2D([],[],marker='x',ls='',color='#777777',ms=4,label='direct simulation')]
    if kind=='T':legend=legend[2:]
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.045),ncol=len(legend),frameon=False,fontsize=7)
    fig.suptitle('Равновесия и экстремумы циклов' if kind=='C' else 'Периоды циклов',fontsize=11)
    name='selected_bifurcations' if kind=='C' else 'selected_periods'
    for ext in ('png','pdf','svg'):fig.savefig(DATA/f'{name}.{ext}',dpi=250,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    data=load();plot('C',data);plot('T',data)
