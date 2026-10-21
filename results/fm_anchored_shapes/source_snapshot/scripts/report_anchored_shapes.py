"""Physical trajectory, shape and exact-rest audit for constrained FM fitting."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from pwl_nest.calcium_coordinates import CalciumTransformPWA,simulate_coordinates
from pwl_nest.rest_protocols import stable_rest
from audit_input_threshold import audit

COLORS=dict(original='#d8962b',control='#777777',linear='#2874b8',quadratic='#ab3c79')
LABELS=dict(original='original FM',control='exact-rest control',linear='exact-rest linear',quadratic='exact-rest quadratic')


def report(out):
    summary=json.loads((out/'summary.json').read_text());selected=summary['selected']
    rest=np.asarray(summary['config']['rest']);checks={};table=[]
    for row in selected:
        family=row['family'];path=Path(row['model']) if family=='original' else out/row['model']
        model=CalciumTransformPWA.load(path);actual,count=stable_rest(model,.45,rest)
        if family!='original':
            analytic=audit(model)
            t,x,_=simulate_coordinates(model,rest,20.,tau=20.,baseline=.45,dt=.25,rtol=2e-11,atol=2e-13,max_step=.1)
            analytic.update(rest=actual.tolist(),LR_rest=rest.tolist(),stable_rest_count=count,
                equilibrium_field_error=float(abs(model.field(model.to_latent(rest))).max()),
                pre_input_drift_max_error=float(abs(x-rest).max()),
                scope='Exact-rest and interior regional propagation audit; no full event solver or speed claim')
            assert analytic['equilibrium_field_error']<2e-10 and analytic['pre_input_drift_max_error']<2e-10
            checks[family]=analytic
        protocols=row['heldout']['protocols'];complete=all(p['completed'] for p in protocols)
        shape=[p.get('shape_check',{}) for p in protocols]
        heights=[p.get('peaks',{}).get('peak_amplitude_MAE') for p in protocols]
        width_errors=[]
        for p in protocols:
            peaks=p.get('peaks',{})
            if peaks.get('counts',[0,-1])[0]==peaks.get('counts',[0,-1])[1] and peaks.get('LR_widths_half_prominence_s'):
                width_errors.append(float(np.mean(abs(np.array(peaks['PWA_widths_half_prominence_s'])/peaks['LR_widths_half_prominence_s']-1))))
        table.append(dict(family=family,parameters=29 if family=='original' else row['parameters'],regions=8,
            train_loss=row['verified_train_loss'],train_eligible=row['checks']['eligible'],
            train_shape_passes=row['checks'].get('shape_passes',0),C_rest=float(actual[0]),h_rest=float(actual[1]),
            static_shape_passes=row['checks'].get('static_shape_passes',0),
            train_max_height_relative_error=row['checks'].get('max_height_relative_error'),
            train_max_width_relative_error=row['checks'].get('max_width_relative_error'),
            test_C_RMSE=row['heldout']['aggregate']['C_RMSE'],test_peak_count_matches=row['heldout']['aggregate']['peak_count_matches'],
            test_shape_passes=row['heldout']['shape_passes'],
            test_mean_peak_height_MAE=float(np.mean(heights)) if complete and all(h is not None for h in heights) else None,
            test_mean_peak_width_relative_MAE=float(np.mean(width_errors)) if len(width_errors)==len(protocols) else None,
            a=model.mix_linear,b=model.mix_quadratic))
    (out/'analytic_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    with (out/'comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]),lineterminator='\n');writer.writeheader();writer.writerows(table)

    protocol=next(p for p in json.loads((out/'cache/FM/data.json').read_text())['fixed'] if p['name']=='fixed_0.53')
    with np.load(out/'cache/FM/fixed_0.53.npz') as d:tr,truth=d['t'],d['x']
    fig,axes=plt.subplots(1,2,figsize=(13,4.6),constrained_layout=True)
    mask=tr>=tr[-1]-400
    axes[0].plot(tr,truth[:,0],color='#222222',lw=1.3,label='Li-Rinzel')
    axes[1].plot(truth[mask,0],truth[mask,1],color='#222222',lw=1.3,label='Li-Rinzel')
    for row in selected:
        family=row['family'];path=Path(row['model']) if family=='original' else out/row['model']
        model=CalciumTransformPWA.load(path)
        t,x,_=simulate_coordinates(model,**{k:v for k,v in protocol.items() if k!='name'},dt=.2,
                                    rtol=2e-11,atol=2e-13,max_step=.25)
        np.savez_compressed(out/f'autonomous_053_{family}.npz',t=t,lr=truth,candidate=x)
        axes[0].plot(t,x[:,0],lw=.9,color=COLORS[family],label=LABELS[family])
        mask=t>=t[-1]-400
        axes[1].plot(x[mask,0],x[mask,1],lw=.9,color=COLORS[family],label=LABELS[family])
    axes[0].set(xlim=(1450,1600),xlabel='Time (s)',ylabel='C (uM)',title='Full spikes at constant I = 0.53')
    axes[1].set(xlabel='C (uM)',ylabel='h',title='Last 400 s: physical phase portrait')
    for ax in axes:ax.legend(fontsize=7)
    fig.suptitle('Autonomous validation: tiny oscillations do not substitute for full FM cycles')
    for ext in ['png','pdf']:fig.savefig(out/f'autonomous_053.{ext}',dpi=180)
    plt.close(fig)

    # Maximum error per input, so a good average cannot hide one malformed peak.
    names=[p['name'] for p in selected[0]['checks']['shape']]
    fig,axes=plt.subplots(1,2,figsize=(13,4.6),constrained_layout=True)
    positions=np.arange(len(names));bar_width=.18
    for j,row in enumerate(selected):
        family=row['family']
        for ax,key in zip(axes,['max_height_relative_error','max_width_relative_error']):
            values=np.array([p[key] for p in row['checks']['shape']])*100
            bars=ax.bar(positions+(j-1.5)*bar_width,np.minimum(values,60.),width=bar_width,
                        label=LABELS[family],color=COLORS[family])
            for bar,p in zip(bars,row['checks']['shape']):
                if not p['counts_match']:bar.set_hatch('///')
        for ax in axes:ax.axhline(10,color='#b22222',lw=.6)
    for ax,title in zip(axes,['Maximum peak-height error','Maximum half-prominence width error']):
        ax.set_xticks(positions,[n.replace('pulse_tau_','tau ').replace('promoted_tau_','tau ') for n in names],rotation=25)
        ax.set(title=title,ylabel='Relative error (%)',ylim=(0, max(12,min(62,ax.get_ylim()[1]))))
        ax.legend(fontsize=7)
    fig.suptitle('Independent DOP853 training checks: 10% limit; hatching means peak counts differ')
    for ext in ['png','pdf']:fig.savefig(out/f'training_shapes.{ext}',dpi=180)
    plt.close(fig)

    names=['anchored_heldout_tau_24.0','anchored_heldout_tau_48.0','anchored_heldout_tau_90.0',
           'anchored_heldout_tau_160.0','anchored_heldout_irregular']
    fig,axes=plt.subplots(len(names),2,figsize=(14,13),constrained_layout=True)
    for (left,right),name in zip(axes,names):
        lag_curves=0
        for row in selected:
            family=row['family'];path=out/'diagnostics'/family/f'{name}.npz'
            if not path.exists():continue
            with np.load(path) as d:t,truth,x=d['t'],d['lr'],d['candidate']
            if family=='original':left.plot(t,truth[:,0],color='#222222',lw=1.3,label='Li-Rinzel')
            label=LABELS[family]+('' if row['checks']['eligible'] else ' [train checks fail]')
            left.plot(t,x[:,0],color=COLORS[family],lw=.9,label=label)
            record=next(p for p in row['heldout']['protocols'] if p['name']==name)
            a=np.array(record['interpolated_peak_times_LR'])+20.;b=np.array(record['interpolated_peak_times_candidate'])+20.
            if len(a)==len(b):
                right.plot(a,b-a,'o-',ms=3,color=COLORS[family],label=LABELS[family]);lag_curves+=1
            else:right.plot([],[],color=COLORS[family],label=f'{LABELS[family]}: {len(b)} vs {len(a)} peaks')
        left.set(title=name,xlabel='Time (s)',ylabel='C (uM)',xlim=(18.,t[-1]))
        right.axhline(0,color='gray',lw=.7);right.set(title='Peak lag: candidate - LR',xlabel='Reference peak time (s)',ylabel='Lag (s)')
        left.legend(fontsize=7);right.legend(fontsize=7)
        if not lag_curves:
            right.text(.5,.45,'All candidate peak counts differ from LR.\nNo peak-lag curve computed.',
                       ha='center',va='center',transform=right.transAxes)
            right.set_axis_off()
    fig.suptitle('FM: exact LR rest and shape-constrained training; genuinely new inputs\nNo clipping or phase shifts; test results do not select models')
    for ext in ['png','pdf']:fig.savefig(out/f'heldout_comparison.{ext}',dpi=180)
    plt.close(fig)
    print(json.dumps(table,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output')
    report(Path(parser.parse_args().output).resolve())
