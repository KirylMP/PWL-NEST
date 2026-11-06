"""Visualize only computed positive-calcium segments, not complete branches."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]
data=json.loads((root/'paper/data/random_selected/continued_both.json').read_text())['results']
upper=json.loads((root/'paper/data/random_selected/continued_upper.json').read_text())['results']
extra=root/'paper/data/random_selected/continued_upper_fm_extended.json'
if extra.exists():
    ext=json.loads(extra.read_text())['branch']
    for item in upper:
        if item['case']=='FM':item['branches']=[ext]
fig, axes=plt.subplots(1,3,figsize=(11,3.2),constrained_layout=True)
for ax, case in zip(axes,('AM','AFM','FM')):
    report=json.loads((root/'paper/data/random_selected'/case/'report.json').read_text())
    fold=report['best']['fold']
    for row in data:
        if row['case']!=case or 'branch' not in row:continue
        pts=[p for p in row['branch']['points'] if p['sampled_positive_C']]
        for stable,color,label in [(True,'#235a9b','stable'),(False,'#c83a38','unstable')]:
            group=[p for p in pts if p['stable']==stable]
            for key in ('Cmin','Cmax'):
                ax.scatter([p['I'] for p in group],[p[key] for p in group],c=color,s=8 if stable else 28, zorder=2 if stable else 5, label=label if key=='Cmax' and group else None)
    for item in upper:
        if item['case']!=case:continue
        for branch in item['branches']:
            group=[p for p in branch['points'] if p['sampled_positive_C'] and not p['stable']]
            for key in ('Cmin','Cmax'):
                ax.scatter([p['I'] for p in group],[p[key] for p in group],c='#c83a38',s=6,marker='x',label='unstable, upper only' if key=='Cmax' and group else None)
    # Fold calcium is separately calculated from the exact neutral orbit in the audit if available.
    ax.axvline(fold['I'],color='0.5',lw=.7,ls=':')
    ax.set_title(case);ax.set_xlabel(r'IP$_3$ $I$ ($\mu$M)');ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=8)
axes[0].set_ylabel(r'Calcium extrema ($\mu$M)')
out=root/'paper/figures/random_selected_fold_continuation'
fig.savefig(str(out)+'.pdf',bbox_inches='tight');fig.savefig(str(out)+'.png',dpi=200,bbox_inches='tight')
print(out)
