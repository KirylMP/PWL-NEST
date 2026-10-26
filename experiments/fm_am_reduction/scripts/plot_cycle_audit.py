#!/usr/bin/env python3
"""Plot an existing count audit without refitting or integrating trajectories."""
import json
import os
import tempfile
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',os.path.join(tempfile.gettempdir(),'pwl-fm-matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME',os.path.join(tempfile.gettempdir(),'pwl-fm-cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_audit(directory):
    rows=json.loads((directory/'summary.json').read_text())
    manifest=json.loads((directory/'manifest.json').read_text())
    labels=sorted(set(q['label'] for q in rows))
    fig,axes=plt.subplots(len(labels),2,figsize=(12,3.5*len(labels)),squeeze=False,sharex=True)
    for ri,label in enumerate(labels):
        group=sorted((q for q in rows if q['label']==label),key=lambda q:q['I'])
        for col,kind in enumerate(['stable','unstable']):
            ax=axes[ri,col]
            for grid,style in [('sparse','o--'),('dense','.-')]:
                vals=[q[grid]['by_epsilon'][-1]['counts'] for q in group]
                ax.plot([q['I'] for q in group],[v[col] if v is not None else np.nan for v in vals],style,label='PWA '+grid)
            ax.plot([q['I'] for q in group],[(q['dense']['by_epsilon'][-1]['reference_counts'] or [np.nan]*2)[col] for q in group],
                    ':',color='purple',label='LR dense')
            observed=[q for q in group if 'coverage' in q]
            if observed:
                ax.scatter([q['I'] for q in observed],[q['coverage']['PWA'][kind] for q in observed],
                           marker='x',s=65,color='black',label='PWA closed orbits observed')
            ax.set_title(label+' / '+kind);ax.set_ylabel('count');ax.set_xlabel('I (uM)')
            ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Smoothed counts on the LR training section; final configured epsilon')
    fig.tight_layout();fig.savefig(directory/'counter_comparison.png',dpi=180)
    fig.savefig(directory/'counter_comparison.pdf');plt.close(fig)
    levels=manifest['structural_I']
    if not levels:return
    fig,axes=plt.subplots(len(labels),len(levels),figsize=(4.5*len(levels),3.5*len(labels)),squeeze=False)
    for ri,label in enumerate(labels):
        for ci,ip3 in enumerate(levels):
            ax=axes[ri,ci]
            path=directory/f'{label}_I_{ip3:.10f}.json'
            if not path.exists():continue
            row=json.loads(path.read_text())
            for key,color in [('LR','purple'),('PWA','black')]:
                records=row['records'][key]
                ax.plot([q['u'] for q in records],[q.get('G',np.nan) for q in records],'.-',color=color,label=key)
            eps=manifest['config']['cycle_smoothing']['epsilon_schedule'][-1]
            ax.axhline(0,color='gray',lw=.8);ax.axhspan(-eps,eps,color='gray',alpha=.15)
            ax.set_yscale('symlog',linthresh=eps);ax.set_xlabel('u');ax.set_ylabel('G(u)')
            count=row.get('coverage',{}).get('PWA',{})
            ax.set_title(f"{label}\nI={ip3:.6f}\nobserved={count.get('observed_cycles','?')}; on section={count.get('cycles_crossing_training_section','?')}",fontsize=9)
            ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Forward return maps; gray band: smoothing width; gaps are not connected')
    fig.tight_layout();fig.savefig(directory/'return_maps.png',dpi=180)
    fig.savefig(directory/'return_maps.pdf');plt.close(fig)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    plot_audit(parser.parse_args().directory)
