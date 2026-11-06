"""Plot observed cycle extrema and periods; these points are not continuation."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / 'paper/data/random_selected'
TARGETS = ROOT / 'paper/data/physical_models/fixed_fold'
fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex='col', constrained_layout=True)
for col, case in enumerate(('AM', 'AFM', 'FM')):
    d = json.loads((TARGETS/case/'parameters.json').read_text())
    report = json.loads((MODELS/case/'report.json').read_text())
    rows = sorted(d['training_cycles'] + d['holdout_cycles'], key=lambda r:r['I'])
    for key, marker in [('target_Cmin', 'o'), ('target_Cmax', 's')]:
        axes[0,col].scatter([x['I'] for x in rows], [x[key] for x in rows], c='black', marker=marker, s=18, zorder=4, label='LR' if key=='target_Cmax' else None)
    axes[1,col].scatter([x['I'] for x in rows], [x['target_period'] for x in rows], c='black', marker='s', s=18, zorder=4)
    samples = [(v,'fit') for v in report['best']['cycles']]
    samples += [(v['cycle']['candidates'][0],'test') for v in report['holdout'] if v['cycle']['candidates']]
    if case == 'AM':
        # Recovered with IVP-seeded numerical matching; original report is unchanged.
        samples.append(({'I':0.40101959300428724,'Cmin':0.1073082756056295,'Cmax':0.3075071163209864,'period':12.445502993683665},'test'))
    for group, marker in [('fit','o'), ('test','^')]:
        vals = [v for v,g in samples if g==group]
        for key in ('Cmin','Cmax'):
            axes[0,col].scatter([v['I'] for v in vals], [v[key] for v in vals], s=24, marker=marker, facecolors='#c5353c' if group=='fit' else 'none', edgecolors='#c5353c', label='PWA' if key=='Cmax' and group=='fit' else None)
        axes[1,col].scatter([v['I'] for v in vals], [v['period'] for v in vals], s=24, marker=marker, facecolors='#c5353c' if group=='fit' else 'none', edgecolors='#c5353c')
    axes[0,col].set_title(case);axes[1,col].set_xlabel(r'IP$_3$ $I$ ($\mu$M)');axes[0,col].legend(frameon=False,fontsize=8)
    for ax in axes[:,col]:ax.grid(alpha=.2)
axes[0,0].set_ylabel(r'Calcium extrema ($\mu$M)');axes[1,0].set_ylabel('Period (s)')
out=ROOT/'paper/figures/random_selected_cycles'
fig.savefig(str(out)+'.pdf',bbox_inches='tight');fig.savefig(str(out)+'.png',dpi=200,bbox_inches='tight')
print(out)
