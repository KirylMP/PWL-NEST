"""Render and audit the supplied summary; no simulations or model selection."""
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

OUT = Path(__file__).resolve().parent
SOURCE = OUT / 'source_summary.json'
CS = ['identity', 'sqrt', 'cbrt', 'sixth_root', 'log', 'square', 'cube',
      'sixth', 'exp', 'inverse', 'inverse_sqrt', 'hill', 'bounded_logit', 'sqrt_tails']
HS = ['identity', 'logit', 'log', 'exp', 'cube', 'sixth']
rows = json.loads(SOURCE.read_text())
assert len({(r['case'], r['C'], r['h']) for r in rows}) == len(rows)
assert all(r['status'] == 'verified' for r in rows)
if SOURCE.resolve() != (OUT / 'source_summary.json').resolve():
    shutil.copyfile(SOURCE, OUT / 'source_summary.json')

columns = ['case', 'C', 'h', 'verified_train_loss', 'test_C_RMSE',
           'test_phase_aligned_C_RMSE', 'test_peak_count_matches',
           'test_all_physical', 'stable_rest_count', 'verified_restarts', 'restart', 'model']
with (OUT / 'all_cells.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=columns, lineterminator='\n')
    writer.writeheader()
    writer.writerows({k: r.get(k) for k in columns} for r in rows)

protocol_columns = ['case', 'C', 'h', 'protocol', 'completed', 'C_RMSE',
                    'aligned_C_RMSE', 'LR_peaks', 'candidate_peaks',
                    'first_peak_lag_s', 'last_peak_lag_s', 'physical', 'error']
with (OUT / 'protocols.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=protocol_columns, lineterminator='\n')
    writer.writeheader()
    for r in rows:
        for t in r['test']:
            peaks = t.get('peaks', {})
            counts = peaks.get('counts', [None, None])
            writer.writerow(dict(case=r['case'], C=r['C'], h=r['h'], protocol=t['name'],
                completed=t['completed'], C_RMSE=t.get('raw_full_RMSE', [None])[0],
                aligned_C_RMSE=t.get('global_shift', {}).get('aligned_eval_rmse'),
                LR_peaks=counts[0], candidate_peaks=counts[1],
                first_peak_lag_s=peaks.get('first_lag_s'),
                last_peak_lag_s=peaks.get('last_lag_s'), physical=t.get('physical'),
                error=t.get('error')))

for case in ['AM', 'FM']:
    records = {(r['C'], r['h']): r for r in rows if r['case'] == case}
    fig, axes = plt.subplots(1, 3, figsize=(16, 9), constrained_layout=True)
    keys = ['test_C_RMSE', 'test_phase_aligned_C_RMSE', 'verified_train_loss']
    titles = ['Test C RMSE (uM), original timing',
              'Test C RMSE (uM), phase aligned', 'Verified training loss']
    for ax, key, title in zip(axes, keys, titles):
        matrix = np.full((len(CS), len(HS)), np.nan)
        for i, c in enumerate(CS):
            for j, h in enumerate(HS):
                r = records.get((c, h))
                if r is not None and r.get(key) is not None:
                    matrix[i, j] = r[key]
        finite = matrix[np.isfinite(matrix) & (matrix > 0)]
        cmap = plt.get_cmap('viridis_r').copy()
        cmap.set_bad('#eeeeee')
        im = ax.imshow(np.ma.masked_invalid(matrix), aspect='auto', cmap=cmap,
                       norm=LogNorm(vmin=finite.min(), vmax=finite.max()))
        for i, c in enumerate(CS):
            for j, h in enumerate(HS):
                r = records.get((c, h)); v = matrix[i, j]
                label = '—' if r is None else 'FAIL' if not np.isfinite(v) else f'{v:.3g}'
                if np.isfinite(v) and v >= 1000:
                    label = f'{v / 1000:.0f}k' if v < 1e6 else f'{v / 1e6:.2g}M'
                if r is not None and (not r.get('test_all_physical') or r.get('test_peak_count_matches') != 4):
                    label += '!'
                ax.text(j, i, label, ha='center', va='center', fontsize=8,
                        bbox=dict(facecolor='white', alpha=.8, edgecolor='none', pad=.8))
        ax.set_xticks(range(len(HS)), HS, rotation=35, ha='right')
        ax.set_yticks(range(len(CS)), CS)
        ax.set(xlabel='h coordinate', ylabel='C coordinate', title=title)
        fig.colorbar(im, ax=ax, shrink=.6)
    fig.suptitle(f'{case}: supplied train-selected candidates ({len(records)}/84 cells)\n'
                 '! = physical check or peak-count mismatch; — = absent from summary; colors logarithmic', fontsize=12)
    for suffix in ['png', 'pdf']:
        fig.savefig(OUT / f'{case}_coordinate_grid.{suffix}', dpi=180)
    plt.close(fig)

print(json.dumps(dict(cells=len(rows), verified_restarts=sum(r['verified_restarts'] for r in rows),
    output=str(OUT)), ensure_ascii=False))
