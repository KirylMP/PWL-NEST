"""Input-locked held-out trajectories for the matched I-threshold experiment."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / '.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(out):
    summary = json.loads((out / 'summary.json').read_text())
    selected = summary['selected']
    colors = dict(original='#df9130', control='#7d7d7d', one_I_knot='#2475b5')
    labels = dict(original='original FM', control='refitted, no I knot', one_I_knot='one fitted I knot')
    protocols = ['heldout_tau_28.0', 'heldout_tau_57.0', 'heldout_tau_95.0', 'heldout_tau_170.0']
    fig, axes = plt.subplots(4, 2, figsize=(14, 11), constrained_layout=True)
    for (left, right), protocol in zip(axes, protocols):
        duration = next(p['duration'] for p in summary['config']['heldout_protocols'] if p['name'] == protocol)
        has_curve = False
        for row in selected:
            family = row['family']
            path = out / 'diagnostics' / family / f'{protocol}.npz'
            record = next(r for r in row['heldout']['protocols'] if r['name'] == protocol)
            label = labels[family] + ('' if row['checks']['eligible'] else ' [train checks fail]')
            if not path.exists():
                left.plot([], [], color=colors[family], label=label + ': failed')
                continue
            with np.load(path) as d:
                t, lr, candidate = d['t'], d['lr'], d['candidate']
            if family == 'original':
                left.plot(t, lr[:, 0], color='#222222', linewidth=1.4, label='Li-Rinzel')
            left.plot(t, candidate[:, 0], color=colors[family], linewidth=1., alpha=.9, label=label)
            a = np.array(record['interpolated_peak_times_LR']) + 20.
            b = np.array(record['interpolated_peak_times_candidate']) + 20.
            if len(a) == len(b):
                right.plot(a, b - a, 'o-', color=colors[family], markersize=3, label=labels[family])
                has_curve = True
            else:
                right.plot([], [], color=colors[family], label=f'{labels[family]}: {len(b)} vs {len(a)} peaks')
        left.set(title=protocol, xlabel='Time (s)', ylabel='C (uM)', xlim=(18., duration))
        right.axhline(0, color='grey', linewidth=.7)
        right.set(title='Peak lag: candidate - LR', xlabel='Reference peak time (s)', ylabel='Lag (s)')
        if not has_curve:
            right.text(.5, .5, 'Peak counts differ; no lag curve computed.', ha='center', va='center', transform=right.transAxes)
            right.set_axis_off()
        left.legend(fontsize=7, loc='upper right')
        right.legend(fontsize=7)
    knot = next((r['input_thresholds'][0] for r in selected if r['family'] == 'one_I_knot'), None)
    knot_text = f'{knot:.6f}' if knot is not None else 'unavailable'
    fig.suptitle(f'FM: matched training budgets, genuinely new inputs; I knot = {knot_text}\n'
                 'No clipping or phase shifts; see report for training and test failures')
    for suffix in ['png', 'pdf']:
        fig.savefig(out / f'heldout_comparison.{suffix}', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output')
    plot(Path(parser.parse_args().output).resolve())
