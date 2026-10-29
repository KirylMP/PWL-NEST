#!/usr/bin/env python3
"""Plot recorded NEST traces from a completed benchmark."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('traces',type=Path)
a=p.parse_args();data=np.load(a.traces)
fig,axes=plt.subplots(2,4,figsize=(16,7),constrained_layout=True)
for row,case in enumerate(('am','fm')):
    for col,(key,title) in enumerate([('Ca_astro',r'$C$ ($\mu$M)'),('h_IP3R',r'LR $h$ / PWA hidden $q$'),('IP3',r'$I$ ($\mu$M)'),('I_SIC',r'$I_{SIC}$ (pA)')]):
        ax=axes[row,col]
        for model,label,color in [('lr','Li–Rinzel','#a855c7'),('pwa','PWA','#111111')]:
            timekey='sic_times' if key=='I_SIC' else 'times'
            ax.plot(data[f'{case}_{model}_{timekey}']/1000.,data[f'{case}_{model}_{key}'],color=color,lw=1,label=label)
        ax.set(title=f'{case.upper()}: {title}',xlabel='Time (s)');ax.grid(alpha=.18)
        ax.spines[['top','right']].set_visible(False)
axes[0,0].legend();fig.suptitle('NEST: driven PWA and Li–Rinzel trajectories',fontsize=13)
out=a.traces.with_name(a.traces.stem+'_comparison');fig.savefig(out.with_suffix('.png'),dpi=180);fig.savefig(out.with_suffix('.pdf'));print(f'Wrote {out}.png and {out}.pdf')
