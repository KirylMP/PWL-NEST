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
from pwl_nest.peak_timing import interpolated_peak_times
from overnight_coordinate_grid import atomic

parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
parser.add_argument('--export-model',default='models/continued_shape_candidates/FM_control.json')
parser.add_argument('--training-plots',nargs=2,default=['anchored_heldout_tau_48.0','anchored_heldout_tau_160.0'])
args=parser.parse_args();out=Path(args.output)
summary=json.loads((out/'summary.json').read_text());old,new=summary['compared']
candidate_label='Continuation' if summary['recommendation']['name']==new['name'] else 'Diagnostic trial'
config=json.loads((out/'config.json').read_text())
model=CalciumTransformPWA.load(out/new['model'])
baseline_model=CalciumTransformPWA.load(old['model'])
spec=json.loads((out/'cache/FM/data.json').read_text());checks=[];phase_training=[]
for p in spec['train']:
    t,x,_=simulate_coordinates(model,spec['rest'],**{k:v for k,v in p.items() if k!='name'},
        dt=.05,rtol=2e-11,atol=2e-13,max_step=.05)
    with np.load(out/'cache/FM'/f'{p["name"]}.npz') as data:
        check=peak_shape_check(t,x,data['t'],data['x'])
        truth_times=interpolated_peak_times(data['t'],data['x'])
    bt,bx,_=simulate_coordinates(baseline_model,spec['rest'],**{k:v for k,v in p.items() if k!='name'},
        dt=.05,rtol=2e-11,atol=2e-13,max_step=.05)
    new_times=interpolated_peak_times(t,x);old_times=interpolated_peak_times(bt,bx)
    phase_training.append(dict(name=p['name'],reference_peaks=truth_times.tolist(),
        old_errors=(old_times-truth_times).tolist() if len(old_times)==len(truth_times) else None,
        new_errors=(new_times-truth_times).tolist() if len(new_times)==len(truth_times) else None))
    checks.append(dict(name=p['name'],**check))
    np.savez_compressed(out/f'fine_train_{p["name"]}.npz',t=t,candidate=x)
atomic(out/'fine_training_audit.json',dict(dt=.05,protocols=checks,passes=all(p['passes'] for p in checks)))

def phase_aggregate(rows,key):
    if any(row[key] is None for row in rows):return None
    values=np.concatenate([row[key] for row in rows])
    return dict(mean_absolute_error_s=float(np.mean(abs(values))),max_absolute_error_s=float(np.max(abs(values))),
        peak_count=len(values))

phase_heldout=[]
for a,b in zip(old['heldout']['protocols'],new['heldout']['protocols']):
    times=np.array(a.get('interpolated_peak_times_LR',[]));old_times=np.array(a.get('interpolated_peak_times_candidate',[]))
    new_times=np.array(b.get('interpolated_peak_times_candidate',[]))
    phase_heldout.append(dict(name=a['name'],reference_peaks=times.tolist(),
        old_errors=(old_times-times).tolist() if a['completed'] and len(times)==len(old_times) else None,
        new_errors=(new_times-times).tolist() if b['completed'] and len(times)==len(new_times) else None))
fine_training_phase=phase_aggregate(phase_training,'new_errors')
phase_ceiling=config.get('phase_ceiling')
fine_phase_passes=bool(fine_training_phase is not None and (phase_ceiling is None or fine_training_phase['mean_absolute_error_s']<=phase_ceiling))
atomic(out/'phase_audit.json',dict(training=phase_training,heldout=phase_heldout,
    phase_ceiling_s=phase_ceiling,fine_training_phase_passes=fine_phase_passes,
    training_aggregate={key:phase_aggregate(phase_training,key) for key in ['old_errors','new_errors']},
    heldout_aggregate={key:phase_aggregate(phase_heldout,key) for key in ['old_errors','new_errors']},
    definition='Chronological absolute peak-time differences, without shifting trajectories. Aggregate unavailable on any count mismatch.'))

fig,axes=plt.subplots(3,2,figsize=(11,8),constrained_layout=True)
for ax,row in zip(axes.flat,phase_heldout):
    for key,color,label in [('old_errors','tab:orange','Previous'),('new_errors','tab:blue',candidate_label)]:
        if row[key] is not None:ax.plot(row['reference_peaks'],row[key],'o-',ms=3,color=color,label=label)
        else:ax.text(.05,.85 if key=='old_errors' else .7,label+': peak count mismatch',transform=ax.transAxes,color=color)
    ax.axhline(0,color='black',lw=.8);ax.grid(alpha=.2)
    ax.set(title=row['name'].replace('_heldout_',' / '),xlabel='Reference peak time from input onset (s)',ylabel='Candidate − LR time (s)')
axes.flat[0].legend();fig.suptitle('Absolute phase errors on fresh inputs')
for suffix in ['png','pdf']:fig.savefig(out/f'phase_errors.{suffix}',dpi=170)
plt.close(fig)

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
        ax.plot(d['t'],d['candidate'][:,0],color='tab:blue',lw=.9,label=candidate_label)
    label=f'tau={p["tau"]:g} s' if 'irregular' not in p['name'] else f'Irregular, tau={p["tau"]:g} s'
    ax.set(title=f'{label} | peaks LR / old / new: {a["peaks"]["counts"][0]} / {a["peaks"]["counts"][1]} / {b["peaks"]["counts"][1]}',
        xlabel='Time (s)',ylabel='C (μM)',xlim=(0,p['duration']))
    ax.grid(alpha=.15)
axes.flat[0].legend(ncol=3,fontsize=8)
fig.suptitle('Fresh inputs: exact rest, unchanged shape constraints')
for suffix in ['png','pdf']:fig.savefig(out/f'heldout_comparison.{suffix}',dpi=170)
plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(13,3.6),constrained_layout=True)
for ax,name in zip(axes,args.training_plots):
    p=next(p for p in spec['train'] if p['name']==name)
    with np.load(out/'cache/FM'/f'{p["name"]}.npz') as d:ax.plot(d['t'],d['x'][:,0],color='black',lw=1.2,label='Li–Rinzel')
    with np.load(out/f'fine_train_{p["name"]}.npz') as d:ax.plot(d['t'],d['candidate'][:,0],color='tab:blue',lw=1,label=candidate_label)
    label='irregular' if 'irregular' in name else 'pulse'
    ax.set(title=f'Promoted training {label}: tau={p["tau"]:g} s',xlabel='Time (s)',ylabel='C (μM)',xlim=(0,p['duration']));ax.grid(alpha=.15)
axes[0].legend()
for suffix in ['png','pdf']:fig.savefig(out/f'promoted_training.{suffix}',dpi=170)
plt.close(fig)

if new['checks']['eligible'] and all(p['passes'] for p in checks) and fine_phase_passes and summary['recommendation']['name']==new['name']:
    exported=copy.deepcopy(model)
    for key in list(exported.metadata):
        if 'loss' in key:exported.metadata.pop(key)
    exported.metadata.update(accepted=False,current_validation=str(out),
        recommended_by_training_only=True,verified_train_loss=new['verified_train_loss'],
        training_checks=new['checks'],fine_training_shape_passes=len(checks),
        fine_training_phase_check=dict(passes=fine_phase_passes,ceiling_s=phase_ceiling,**fine_training_phase),
        heldout_shape_passes=new['heldout']['shape_passes'],heldout_protocol_count=len(spec['test']),
        source_model=old['model'],selection_scope='expanded training only; frozen before fresh-input readout')
    path=Path(args.export_model);path.parent.mkdir(parents=True,exist_ok=True);exported.save(path)
    atomic(out/'export_manifest.json',dict(model=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        raw_source=new['model'],physical_coefficients_unchanged=bool(np.array_equal(exported.latent.matrix,model.latent.matrix)),
        accepted=False))
print(json.dumps(dict(fine_train_passes=sum(p['passes'] for p in checks),train_count=len(checks),
    max_height_error=max(p['max_height_relative_error'] for p in checks),max_width_error=max(p['max_width_relative_error'] for p in checks)),indent=2))
