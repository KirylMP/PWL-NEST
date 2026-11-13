"""Plot preliminary transformed-coordinate fits and the AM input reset."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'results/fold_anchors'
fig,axes=plt.subplots(3,2,figsize=(10,10),constrained_layout=True)
for n,case in enumerate(('AM','AFM','FM')):
 seeds={('AM','log'):6,('AM','square'):3,('AFM','log'):2,('AFM','square'):3,('FM','log'):7,('FM','square'):0};data={k:{'results':json.loads((out/case/k/f"seed_{seeds[(case,k)]}_verification.json").read_text())['cycles']} for k in ('log','square')}; ref=data['log']['results'];I=np.array([r['I'] for r in ref]);order=np.argsort(I)
 for label,color in [('reference','#252525'),('log','#cf3a47'),('square','#2475a6')]:
  rows=ref if label=='reference' else data[label]['results'];key='reference' if label=='reference' else 'prediction'
  for col,metrics in [(0,('Cmin','Cmax')),(1,('period',))]:
   for metric in metrics:
    values=[r[key][metric] if r[key] else np.nan for r in rows];axes[n,col].plot(I[order],np.asarray(values)[order],marker='o',ms=2,lw=1,color=color,ls='--' if metric=='Cmin' else '-',label=label if metric in ('Cmax','period') else None)
 report=json.loads((ROOT/'paper/data/random_selected'/case/'report.json').read_text());old={x['I']:x for x in report['best']['cycles']}
 for row in report['holdout']:
  if row['cycle']['candidates']:old[row['I']]=row['cycle']['candidates'][0]
 for col,metrics in [(0,('Cmin','Cmax')),(1,('period',))]:
  for metric in metrics:
   values=[old[r['I']][metric] if r['I'] in old else np.nan for r in ref];axes[n,col].plot(I[order],np.asarray(values)[order],marker='s',ms=2,lw=.8,color='#32904b',ls='--' if metric=='Cmin' else '-',label='physical PWA' if metric in ('Cmax','period') else None)
 for col in (0,1):axes[n,col].set_xlabel('IP₃ I (µM)');axes[n,col].grid(alpha=.2)
 axes[n,0].set_ylabel(f'{case}: calcium (µM)');axes[n,1].set_ylabel('Period (s)')
axes[0,0].set_title('Extrema: min dashed, max solid');axes[0,1].set_title('Period');axes[0,0].legend(frameon=False)
fig.savefig(out/'comparison.png',dpi=180);fig.savefig(out/'comparison.pdf');plt.close(fig)
z=np.load(out/'dynamic.npz');t=z['AM_time'];fig,ax=plt.subplots(figsize=(8,3.6),constrained_layout=True)
for k,c in [('physical','#252525'),('log','#cf3a47'),('square','#2475a6')]:ax.plot(t,z['AM_'+k+'_C'],c=c,lw=1,label=k)
for x in (80,160,240,320):ax.axvline(x,c='0.7',ls=':',lw=.8)
ax.axhline(0,color='0.5',lw=.7);ax.set_xlabel('Time (s)');ax.set_ylabel('AM calcium (µM)');ax.legend(frameon=False);ax.grid(alpha=.2)
fig.savefig(out/'AM_dynamic.png',dpi=180);fig.savefig(out/'AM_dynamic.pdf');plt.close(fig)
