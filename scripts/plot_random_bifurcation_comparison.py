"""Compare archival LR branches with finite new selected PWA continuations."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from plot_selected_lr_branches import equilibrium

ROOT=Path(__file__).resolve().parents[1];base=ROOT/'paper/data/random_selected';ref=ROOT/'paper/figures/figure1_data'
rr=list(csv.DictReader((ref/'cycles.csv').open()));both=json.loads((base/'continued_both.json').read_text())['results'];upper=json.loads((base/'continued_upper.json').read_text())['results'];fm=json.loads((base/'continued_upper_fm_extended.json').read_text())['branch']
limits={'AM':(.29,.70),'AFM':(.27,.98),'FM':(.46,1.2)}
fig,axes=plt.subplots(2,3,figsize=(11,6.8),sharex='col',sharey='col',constrained_layout=True)
for col,case in enumerate(('AM','AFM','FM')):
    eq=np.load(ref/f'{case}_equilibria.npz');a=axes[0,col]
    for stable,style in [(True,'-'),(False,'--')]:a.plot(eq['I'],np.where(eq['stable']==stable,eq['C'],np.nan),color='#225f98' if stable else '#b84145',ls=style,lw=1)
    for stable in (True,False):
        rows=[x for x in rr if x['case']==case and x['displayed']=='True' and (x['stable']=='True')==stable]
        for key in ('C_min','C_max'):a.scatter([float(x['I']) for x in rows],[float(x[key]) for x in rows],s=6,facecolors='#222222' if stable else 'none',edgecolors='#222222' if stable else '#b84145',linewidths=.5)
    a.set_title(case+' — LR')
    m=json.loads((base/case/'candidate.json').read_text());rep=json.loads((base/case/'report.json').read_text());a=axes[1,col]
    xs=np.linspace(*limits[case],700);ev=[equilibrium(m,x) for x in xs]
    for state,style,colr in [('stable','-','#225f98'),('unstable','--','#b84145')]:a.plot(xs,[next((v[0] for v in vals if v[1]==state),np.nan) for vals in ev],ls=style,c=colr,lw=1)
    pts=[p for r in both if r['case']==case for p in r['branch']['points'] if p['sampled_positive_C']]
    u=next(x for x in upper if x['case']==case);branches=[fm] if case=='FM' else u['branches'];pts += [p for b in branches for p in b['points'] if p['sampled_positive_C']]
    for stable in (True,False):
        rows=[x for x in pts if x['stable']==stable]
        for key in ('Cmin','Cmax'):a.scatter([x['I'] for x in rows],[x[key] for x in rows],s=6,facecolors='#222222' if stable else 'none',edgecolors='#222222' if stable else '#b84145',linewidths=.5)
    a.set_title(case+' — PWA');a.set_xlabel(r'IP$_3$ $I$ ($\mu$M)');a.set_xlim(*limits[case]);axes[0,col].grid(alpha=.2);a.grid(alpha=.2)
axes[0,0].set_ylabel('C (µM)');axes[1,0].set_ylabel('C (µM)')
axes[0,0].set_ylim(0,.8);axes[0,1].set_ylim(0,1.1);axes[0,2].set_ylim(0,2.1)
out=ROOT/'paper/figures/random_selected_bifurcations';fig.savefig(str(out)+'.pdf',bbox_inches='tight');fig.savefig(str(out)+'.png',dpi=200,bbox_inches='tight');print(out)
