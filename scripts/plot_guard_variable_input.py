"""Plot the shared figure-8 input reset, using saved DOP853 trajectories."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]
z=np.load(root/'paper/data/random_selected/guard_variable_input.npz')
t=z['time_s']; mask=(t>=318)&(t<=350)
fig,axes=plt.subplots(1,3,figsize=(10,3),sharex=True,constrained_layout=True)
for ax,case in zip(axes,('AM','AFM','FM')):
    for mode,col in [('three','#222222'),('four','#c3313e')]:
        ax.plot(t[mask]-320,z[f'{case}_{mode}_C'][mask],color=col,lw=1.2,label={'three':'3 zones','four':'4 zones'}[mode])
    ax.axvline(0,c='0.5',ls=':',lw=1);ax.axhline(0,c='0.3',lw=.7);ax.axhline(.025,c='#698777',ls='--',lw=.8)
    ax.set_title(case);ax.set_xlabel('Time after input decrease (s)');ax.set_ylabel('C (µM)');ax.grid(alpha=.2)
axes[0].legend(frameon=False,fontsize=8)
fig.savefig(root/'paper/figures/guard_variable_input.pdf',bbox_inches='tight')
fig.savefig(root/'paper/figures/guard_variable_input.png',dpi=200,bbox_inches='tight')
