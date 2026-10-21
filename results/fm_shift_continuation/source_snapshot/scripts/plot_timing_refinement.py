"""Plot frozen and train-selected models on old and new timed input tests."""
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
    for row in json.loads((out / 'summary.json').read_text()):
        name = row['name']
        protocols = [('previous_tests', 'pulse_tau_65.0'), ('new_tests', 'new_tau_35.0'),
                     ('new_tests', 'new_tau_80.0'), ('new_tests', 'new_tau_140.0')]
        fig, axes = plt.subplots(len(protocols), 2, figsize=(14, 11), constrained_layout=True)
        for (left, right), (split, protocol) in zip(axes, protocols):
            duration = next(p['duration'] for p in row['old_protocols'] + row['new_protocols'] if p['name'] == protocol)
            has_lag_curve = False
            for label, color in [('baseline', '#db8b24'), ('selected', '#2778b3')]:
                path = out / 'diagnostics' / name / label / split / f'{protocol}.npz'
                record = next(r for r in row[f'{label}_{split}']['protocols'] if r['name'] == protocol)
                if not path.exists():
                    left.text(.5, .5, f'{label}: trajectory failed', transform=left.transAxes)
                    continue
                with np.load(path) as data:
                    t, lr, candidate = data['t'], data['lr'], data['candidate']
                if label == 'baseline':
                    left.plot(t, lr[:, 0], color='#444444', linewidth=1.5, label='Li-Rinzel')
                left.plot(t, candidate[:, 0], color=color, linewidth=1., alpha=.9, label=label)
                a = np.array(record['interpolated_peak_times_LR']) + 20.
                b = np.array(record['interpolated_peak_times_candidate']) + 20.
                if len(a) == len(b):
                    right.plot(a, b - a, 'o-', color=color, label=label, markersize=4)
                    has_lag_curve = True
                else:
                    right.plot([], [], color=color, label=f'{label}: peaks {len(b)} vs {len(a)}')
            left.set(title=protocol, ylabel='C (uM)', xlabel='Time (s)', xlim=(18, duration))
            right.axhline(0, color='grey', linewidth=.7)
            right.set(title='Peak time error: candidate - LR', xlabel='Reference peak time (s)', ylabel='Lag (s)')
            if not has_lag_curve:
                right.text(.5, .5, 'Peak counts differ; no lag curve computed.',
                           transform=right.transAxes, ha='center', va='center')
                right.set_axis_off()
            left.legend(fontsize=8, loc='upper right')
            right.legend(fontsize=8)
        fig.suptitle(f'{name}: original timing; no trajectory shifts\n'
                     f'Independent training loss: {row["baseline_train_loss"]:.3f} → {row["selected_train_loss"]:.3f}')
        for suffix in ['png', 'pdf']:
            fig.savefig(out / f'{name}_timing_comparison.{suffix}', dpi=180)
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output')
    plot(Path(parser.parse_args().output).resolve())
