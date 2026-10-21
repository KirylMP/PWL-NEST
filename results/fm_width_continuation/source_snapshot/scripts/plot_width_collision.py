"""Show the input jump interacting with the known fifth-peak width failure."""
import argparse,json,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/pwl-nest-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/pwl-nest-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pwl_nest.calcium_coordinates import CalciumTransformPWA,simulate_coordinates
from pwl_nest.peak_shape import peak_shape_errors
from overnight_coordinate_grid import atomic

parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
args=parser.parse_args();out=Path(args.output);config=json.loads((out/'config.json').read_text())
spec=json.loads((out/'cache/FM/data.json').read_text());p=next(p for p in spec['train'] if p['name']=='phase_heldout_irregular')
with np.load(out/'cache/FM'/f'{p["name"]}.npz') as d:t=d['t'];truth=d['x']
with np.load(out/f'fine_train_{p["name"]}.npz') as d:tn=d['t'];candidate=d['candidate']
source=CalciumTransformPWA.load(config['source_model'])
to,old,_=simulate_coordinates(source,spec['rest'],**{k:v for k,v in p.items() if k!='name'},dt=.05,rtol=2e-11,atol=2e-13,max_step=.05)
a,b,_,width,_=peak_shape_errors(to,old,t,truth);_,c,_,new_width,_=peak_shape_errors(tn,candidate,t,truth)
j=int(np.argmax(abs(width)));center=a['times'][j]+20.;jump=next(t for t,delta in p['jumps'] if t>center)
fig,axes=plt.subplots(2,1,figsize=(9,5),sharex=True,constrained_layout=True,gridspec_kw={'height_ratios':[3,1]})
axes[0].plot(t,truth[:,0],color='black',label='Li–Rinzel')
axes[0].plot(to,old[:,0],color='tab:orange',label='Retained model')
axes[0].plot(tn,candidate[:,0],color='tab:blue',ls='--',label='Rejected trial')
axes[1].plot(t,truth[:,2],color='tab:green')
for ax in axes:
 ax.axvline(jump,color='gray',ls=':',label='Input jump');ax.grid(alpha=.2);ax.set_xlim(center-5.,center+12.)
axes[0].set(ylabel='C (μM)',title=f'Known training failure: peak {j+1}, width error {100*width[j]:.1f}% → {100*new_width[j]:.1f}%')
axes[1].set(ylabel='I (μM)',xlabel='Time (s)');axes[0].legend(fontsize=8)
for suffix in ['png','pdf']:fig.savefig(out/f'width_collision.{suffix}',dpi=180)
atomic(out/'width_collision.json',dict(protocol=p['name'],peak_index=j+1,input_jump_s=jump,reference_peak_s=center,
 reference_width_s=float(a['widths'][j]),old_width_s=float(b['widths'][j]),trial_width_s=float(c['widths'][j]),
 old_width_relative_error=float(width[j]),trial_width_relative_error=float(new_width[j])))
