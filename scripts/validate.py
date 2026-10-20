"""Generate fresh baseline evidence; legacy reference PWS columns are not targets."""
import csv
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/workspace/PWL-NEST/.cache/matplotlib')
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pwl_nest.models import ROOT, trajectory
from pwl_nest.native import observables


def metrics(t, c):
    keep = t >= t[-1]-400
    x, z = t[keep], c[keep]
    peaks, _ = find_peaks(z, prominence=1e-4)
    return dict(Cmin=float(z.min()), Cmax=float(z.max()),
                period=float(np.median(np.diff(x[peaks]))) if len(peaks)>=4 else None)


def main():
    destination = ROOT/'results'
    destination.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6))
    report = {'protocol': 'fixed IP3; large-excursion starts; final 400 s; independent numerical and native solvers',
              'scope': 'starting baselines, not full blueprint acceptance or speed benchmark', 'cases': {}}
    for row, (case, grid, representative) in enumerate([
            ('AM', [.35, .37, .4, .5, .6, .65, .8], .5),
            ('FM', [.5, .54, .65, .8, 1.05, 1.13, 1.2], .8)]):
        refs = {round(float(r['I']), 8):r for r in csv.DictReader((ROOT/'models'/f'{case}_reference.csv').open())}
        scan = []
        for ip3 in grid:
            result = observables(case, ip3)
            result = {k: v if np.isfinite(v) else None for k,v in result.items()}
            reference = refs.get(ip3)
            record = {'I':ip3, **result}
            if reference:
                record['LR_reference'] = {k:float(reference['LR_'+k]) if np.isfinite(float(reference['LR_'+k])) else None for k in ('Cmin','Cmax','period')}
            scan.append(record)
        numerical = {}
        for kind in ['lr', 'pwa']:
            t, y = trajectory(case, representative, kind=kind)
            numerical[kind] = metrics(t, y[:,0])
            mask = t >= t[-1]-150
            axes[row,0].plot(t[mask]-t[mask][0], y[mask,0], label=kind)
            np.savez_compressed(destination/f'{case}_{kind}_trace.npz', t=t, state=y, I=representative)
        exact = observables(case, representative)
        errors = {k:abs(numerical['pwa'][k]-exact[k]) for k in ('Cmin','Cmax','period')}
        if errors['Cmin']>2e-4 or errors['Cmax']>2e-4 or errors['period']>.12:
            raise AssertionError(f'Independent solver disagreement: {case} {errors}')
        report['cases'][case] = {'scan':scan, 'representative_I':representative,
                                 'numerical':numerical,'native':exact,'independent_solver_errors':errors}
        axes[row,0].set(title=f'{case}, IP3={representative}', xlabel='time (s)', ylabel='C (uM)')
        axes[row,0].legend()
        axes[row,1].plot(grid,[r['Cmax']-r['Cmin'] for r in scan],'o-',label='PWA amplitude')
        axes[row,1].set(title=f'{case} native scan',xlabel='IP3 (uM)',ylabel='amplitude (uM)')
    fig.tight_layout()
    fig.savefig(destination/'starting_models.png',dpi=160)
    (destination/'validation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({c:r['independent_solver_errors'] for c,r in report['cases'].items()},indent=2))

if __name__ == '__main__':
    main()
