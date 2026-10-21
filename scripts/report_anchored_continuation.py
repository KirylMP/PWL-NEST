"""Audit fine training shapes and export a completed continuation experiment."""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR','/tmp/pwl-nest-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/pwl-nest-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pwl_nest.calcium_coordinates import CalciumTransformPWA,simulate_coordinates
from pwl_nest.peak_shape import peak_shape_check
from overnight_coordinate_grid import atomic
from continue_anchored_fit import SEED

parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
args=parser.parse_args();out=Path(args.output)
summary=json.loads((out/'summary.json').read_text());old,new=summary['compared']
model=CalciumTransformPWA.load(out/new['model'])
spec=json.loads((out/'cache/FM/data.json').read_text());checks=[]
for p in spec['train']:
    t,x,_=simulate_coordinates(model,spec['rest'],**{k:v for k,v in p.items() if k!='name'},
        dt=.05,rtol=2e-11,atol=2e-13,max_step=.05)
    with np.load(out/'cache/FM'/f'{p["name"]}.npz') as data:
        check=peak_shape_check(t,x,data['t'],data['x'])
    checks.append(dict(name=p['name'],**check))
    np.savez_compressed(out/f'fine_train_{p["name"]}.npz',t=t,candidate=x)
atomic(out/'fine_training_audit.json',dict(dt=.05,protocols=checks,passes=all(p['passes'] for p in checks)))

columns=['protocol','old_C_RMSE','new_C_RMSE','LR_peaks','old_peaks','new_peaks','old_shape_pass','new_shape_pass']
with (out/'comparison.csv').open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=columns,lineterminator='\n');writer.writeheader()
    for a,b in zip(old['heldout']['protocols'],new['heldout']['protocols']):
        writer.writerow(dict(protocol=a['name'],old_C_RMSE=a['raw_full_RMSE'][0],new_C_RMSE=b['raw_full_RMSE'][0],
            LR_peaks=a['peaks']['counts'][0],old_peaks=a['peaks']['counts'][1],new_peaks=b['peaks']['counts'][1],
            old_shape_pass=a['shape_check']['passes'],new_shape_pass=b['shape_check']['passes']))

fig,axes=plt.subplots(3,2,figsize=(13,9),constrained_layout=True)
for ax,p,a,b in zip(axes.flat,spec['test'],old['heldout']['protocols'],new['heldout']['protocols']):
    with np.load(out/'diagnostics'/old['name']/f'{p["name"]}.npz') as d:
        ax.plot(d['t'],d['lr'][:,0],color='black',lw=1.2,label='Li–Rinzel')
        ax.plot(d['t'],d['candidate'][:,0],color='tab:orange',lw=.9,alpha=.8,label='Previous')
    with np.load(out/'diagnostics'/new['name']/f'{p["name"]}.npz') as d:
        ax.plot(d['t'],d['candidate'][:,0],color='tab:blue',lw=.9,label='Continuation')
    label=f'tau={p["tau"]:g} s' if 'irregular' not in p['name'] else f'Irregular, tau={p["tau"]:g} s'
    ax.set(title=f'{label} | peaks LR / old / new: {a["peaks"]["counts"][0]} / {a["peaks"]["counts"][1]} / {b["peaks"]["counts"][1]}',
        xlabel='Time (s)',ylabel='C (μM)',xlim=(0,p['duration']))
    ax.grid(alpha=.15)
axes.flat[0].legend(ncol=3,fontsize=8)
fig.suptitle('Fresh inputs: exact rest, unchanged shape constraints')
for suffix in ['png','pdf']:fig.savefig(out/f'heldout_comparison.{suffix}',dpi=170)
plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(13,3.6),constrained_layout=True)
for ax,tau in zip(axes,[48.,160.]):
    p=next(p for p in spec['train'] if p['tau']==tau)
    with np.load(out/'cache/FM'/f'{p["name"]}.npz') as d:ax.plot(d['t'],d['x'][:,0],color='black',lw=1.2,label='Li–Rinzel')
    with np.load(out/f'fine_train_{p["name"]}.npz') as d:ax.plot(d['t'],d['candidate'][:,0],color='tab:blue',lw=1,label='Continuation')
    ax.set(title=f'Promoted training input: tau={tau:g} s',xlabel='Time (s)',ylabel='C (μM)',xlim=(0,p['duration']));ax.grid(alpha=.15)
axes[0].legend()
for suffix in ['png','pdf']:fig.savefig(out/f'promoted_training.{suffix}',dpi=170)
plt.close(fig)

if new['checks']['eligible'] and all(p['passes'] for p in checks):
    exported=copy.deepcopy(model)
    for key in list(exported.metadata):
        if 'loss' in key:exported.metadata.pop(key)
    exported.metadata.update(accepted=False,current_validation=str(out),
        recommended_by_training_only=True,verified_train_loss=new['verified_train_loss'],
        training_checks=new['checks'],fine_training_shape_passes=len(checks),
        heldout_shape_passes=new['heldout']['shape_passes'],heldout_protocol_count=len(spec['test']),
        source_model=str(SEED),selection_scope='expanded training only; frozen before fresh-input readout')
    directory=Path('models/continued_shape_candidates');directory.mkdir(exist_ok=True)
    path=directory/'FM_control.json';exported.save(path)
    atomic(out/'export_manifest.json',dict(model=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        raw_source=new['model'],physical_coefficients_unchanged=bool(np.array_equal(exported.latent.matrix,model.latent.matrix)),
        accepted=False))
print(json.dumps(dict(fine_train_passes=sum(p['passes'] for p in checks),train_count=len(checks),
    max_height_error=max(p['max_height_relative_error'] for p in checks),max_width_error=max(p['max_width_relative_error'] for p in checks)),indent=2))
