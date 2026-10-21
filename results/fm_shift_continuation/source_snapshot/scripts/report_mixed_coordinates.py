"""Audit and plot physical switching curves and fresh FM validation."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from pwl_nest.calcium_coordinates import CalciumTransformPWA
from audit_input_threshold import audit

COLORS=dict(original='#d8962b',control='#777777',linear='#2874b8',quadratic='#ab3c79')
LABELS=dict(original='original FM',control='refitted control',linear='linear shear',quadratic='quadratic shear')


def report(out):
    summary=json.loads((out/'summary.json').read_text());rows=summary['candidates']
    models={r['family']:CalciumTransformPWA.load(out/'trials'/r['family']/'candidate.json')
            for r in rows if r['family']!='original'}
    checks={}
    for family,m in models.items():
        result=audit(m)
        v=m.gate.forward(np.linspace(.25,.98,19));maximum=0.;valid=0
        local=m.latent.local_coefficients();continuity=0.
        for j,theta in enumerate(m.latent.thresholds):
            for vv in v:
                y=np.array([theta,vv,.7])
                try:physical=m.to_physical(y)
                except ValueError:continue
                valid+=1
                maximum=max(maximum,abs(float(m.to_latent(physical)[0])-theta))
                continuity=max(continuity,float(abs((local[j]-local[j+1])@np.r_[1.,y]).max()))
        result.update(state_mixing=dict(a=m.mix_linear,b=m.mix_quadratic,center=m.mix_center),
            physical_threshold_samples=valid,straightening_max_error=maximum,
            latent_threshold_continuity_max_error=continuity,
            coordinate_Jacobian_determinant='phi_prime(C)*psi_prime(h), independent of a,b',
            regional_matrices_constant=True,
            physical_field_PWA=bool(m.coordinate.kind=='identity' and m.h_kind=='identity' and m.mix_quadratic==0),
            scope='Regional exact flow and inverse-coordinate/threshold checks; no complete event solver or speed benchmark')
        assert valid>0 and maximum<1e-12 and continuity<1e-10
        checks[family]=result
    (out/'analytic_checks.json').write_text(json.dumps(checks,indent=2)+'\n')

    with np.load(out/'cache/FM/pulse_tau_100.0.npz') as data:source=data['x']
    fig,axes=plt.subplots(3,2,figsize=(11,10),constrained_layout=True)
    for (left,right),(family,m) in zip(axes,models.items()):
        left.plot(source[:,0],source[:,1],color='#bbbbbb',lw=.7)
        latent=m.to_latent(source)
        right.plot(latent[:,0],latent[:,1],color='#bbbbbb',lw=.7)
        h=np.linspace(.25,.98,500);v=m.gate.forward(h)
        for theta in m.latent.thresholds:
            u=theta+m.mix(v);c=np.full(len(u),np.nan)
            for i,value in enumerate(u):
                try:c[i]=m.coordinate.inverse(value)
                except ValueError:pass
            left.plot(c,h,color=COLORS[family],lw=.8)
            right.axvline(theta,color=COLORS[family],lw=.8)
        left.set(xlabel='Physical C (uM)',ylabel='Physical h',xlim=(0.,1.4),ylim=(.25,.98),
                 title=f'{LABELS[family]}: physical boundaries')
        right.set(xlabel='U',ylabel='V',title='Same boundaries: U = constant')
    fig.suptitle('Seven switching surfaces, eight regions in every model\nGray: the same Li-Rinzel training trajectory')
    for ext in ['png','pdf']:fig.savefig(out/f'switching_geometry.{ext}',dpi=180)
    plt.close(fig)

    protocols=['mixed_heldout_tau_42.0','mixed_heldout_tau_72.0','mixed_heldout_tau_125.0',
               'mixed_heldout_tau_190.0','mixed_heldout_irregular']
    fig,axes=plt.subplots(len(protocols),2,figsize=(14,13),constrained_layout=True)
    for (left,right),name in zip(axes,protocols):
        for row in rows:
            family=row['family'];path=out/'diagnostics'/family/f'{name}.npz'
            if not path.exists():continue
            with np.load(path) as data:t,truth,x=data['t'],data['lr'],data['candidate']
            if family=='original':left.plot(t,truth[:,0],color='#222222',lw=1.3,label='Li-Rinzel')
            label=LABELS[family]+('' if row['checks']['eligible'] else ' [train checks fail]')
            left.plot(t,x[:,0],color=COLORS[family],lw=.9,label=label)
            record=next(r for r in row['heldout']['protocols'] if r['name']==name)
            a=np.array(record['interpolated_peak_times_LR'])+20.
            b=np.array(record['interpolated_peak_times_candidate'])+20.
            if len(a)==len(b):right.plot(a,b-a,'o-',ms=3,color=COLORS[family],label=LABELS[family])
            else:right.plot([],[],color=COLORS[family],label=f'{LABELS[family]}: {len(b)} vs {len(a)} peaks')
        left.set(title=name,xlabel='Time (s)',ylabel='C (uM)',xlim=(18.,t[-1]))
        right.axhline(0,color='gray',lw=.7)
        right.set(title='Peak lag: candidate - LR',xlabel='Reference peak time (s)',ylabel='Lag (s)')
        left.legend(fontsize=7);right.legend(fontsize=7)
    fig.suptitle('FM mixed coordinates: fresh inputs, no clipping or phase shifts\nMatched optimization caps; train eligibility shown explicitly')
    for ext in ['png','pdf']:fig.savefig(out/f'heldout_comparison.{ext}',dpi=180)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output')
    report(Path(parser.parse_args().output).resolve())
