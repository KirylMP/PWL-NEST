"""Exploratory lower-zone fit which preserves the incumbent fold locally.

The correction is proportional to (cut-C)_+. The fold trajectory stays above
the cut, so the original vector field and its derivative are unchanged along
that trajectory. Other bifurcations, equilibrium, and accuracy require checks.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]


def make_model(base, cut, da, db):
    d = dict(base)
    d['physical_thresholds_C'] = [float(cut)] + base['physical_thresholds_C']
    for key, delta in (('alpha', da), ('beta', db)):
        v = np.asarray(base[key], float).copy()
        v[0] += delta * cut
        v[1] -= delta
        d[key] = np.r_[v[:3], delta, v[3:]].tolist()
    a, b = float(d['a']), float(d['b'])
    n, h = np.asarray(d['alpha']), np.asarray(d['beta'])
    mats = [np.array([[-a * (n[1] + sum(n[3:3+j])), a],
                      [b * (h[1] + sum(h[3:3+j])), -b]]) for j in range(4)]
    d['trace'] = [float(np.trace(m)) for m in mats]
    d['determinant'] = [float(np.linalg.det(m)) for m in mats]
    return d


def rhs(model, I, state):
    c, q = state
    v = np.r_[1., c, I, np.maximum(c - np.asarray(model['physical_thresholds_C']), 0.)]
    return [model['a'] * (q - np.dot(model['alpha'], v)),
            model['b'] * (np.dot(model['beta'], v) - q)]


def metrics(model, row, duration):
    I = row['I']
    def turning(t, y):
        return rhs(model, I, y)[0]
    turning.direction = -1
    y0 = np.asarray(model['rest_target'])
    sol = solve_ivp(lambda t, y: rhs(model, I, y), (0, duration), y0,
                    events=turning, rtol=2e-6, atol=2e-8, max_step=.35,
                    dense_output=True)
    times = sol.t_events[0]
    if not sol.success or len(times) < 4 or not np.all(np.isfinite(sol.y)):
        return None
    period = times[-1] - times[-2]
    previous = times[-2] - times[-3]
    if period <= 0 or abs(period - previous) > max(.1, .02 * period):
        return None
    trace = sol.sol(np.linspace(times[-2], times[-1], 250))[0]
    return {'Cmin': float(trace.min()), 'Cmax': float(trace.max()),
            'period': float(period), 'min_all': float(sol.y[0].min())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases', nargs='+', default=['AM', 'AFM', 'FM'], choices=['AM', 'AFM', 'FM'])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-nfev', type=int, default=20)
    p.add_argument('--duration', type=float, default=300.)
    p.add_argument('--training', choices=['three', 'all'], default='all')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    audit = json.loads((ROOT / 'paper/data/physical_models/fixed_fold/fold_audit.json').read_text())['results']
    for case in args.cases:
        path = ROOT / 'paper/data/physical_models/fixed_fold' / case / 'parameters.json'
        base = json.loads(path.read_text())
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if audit[case.lower()]['sha256'] != sha:
            raise ValueError('fold evidence does not match candidate')
        fold_min = audit[case.lower()]['neutral']['Cmin']
        upper = min(base['physical_thresholds_C'][0] * .95, fold_min * .9)
        lower = max(.0001, upper * .25)
        cut = .9 * upper
        rows = (base['training_cycles'] if args.training == 'all'
                else [base['training_cycles'][i] for i in (0, 3, 6)])
        evaluations = 0
        def residual(x):
            nonlocal evaluations
            evaluations += 1
            model = make_model(base, x[0], x[1], x[2])
            res = []
            for row in rows:
                m = metrics(model, row, args.duration)
                if m is None:
                    res.extend([20., 20., 20.])
                else:
                    res.extend([(m['Cmin']-row['target_Cmin'])/.03,
                                (m['Cmax']-row['target_Cmax'])/.08,
                                (m['period']-row['target_period'])/5.])
            # Keep rest field close to the incumbent; this is not an equality constraint.
            res.extend((np.asarray(rhs(model, 0., base['rest_target'])) / [.03, .01]).tolist())
            # Stable lower linear system: strict margin in trace and determinant.
            res.extend([max(0., model['trace'][0]+.001)*5.,
                        max(0., .001-model['determinant'][0])*5.])
            return np.array(res)
        start = np.array([cut, .03, -.03])
        before_model = make_model(base, cut, 0., 0.)
        before = [metrics(before_model, r, args.duration) for r in rows]
        print(case, 'baseline', before, flush=True)
        baseline_loss = float(np.dot(r0 := residual([cut, 0., 0.]), r0))
        result = least_squares(residual, start, bounds=([lower, -4., -4.], [upper, 4., 4.]),
                               max_nfev=args.max_nfev, diff_step=1e-3, x_scale='jac',
                               ftol=1e-4, xtol=1e-4, gtol=1e-4)
        model = make_model(base, *result.x)
        after = [metrics(model, r, args.duration) for r in rows]
        heldout = []
        for row in base['holdout_cycles']:
            heldout.append({'I': row['I'], 'baseline': metrics(before_model, row, args.duration),
                            'candidate': metrics(model, row, args.duration),
                            'target': {k: row[k] for k in ('target_Cmin', 'target_Cmax', 'target_period')}})
        model['lower_zone_fit'] = {'status': 'exploratory, not validated', 'source_sha256': sha,
            'cut': float(result.x[0]), 'alpha_local': float(result.x[1]), 'beta_local': float(result.x[2]),
            'fold_Cmin_sampled': fold_min, 'fold_preservation': 'field exactly unchanged for C >= cut',
            'warning': 'original optimizer and recorded cycle metadata refer to the source model'}
        out = args.output / case
        out.mkdir()
        (out/'candidate.json').write_text(json.dumps(model, indent=2, allow_nan=False)+'\n')
        report = {'training_I': [r['I'] for r in rows], 'baseline': before, 'candidate': after,
                  'target': [{k: r[k] for k in ('target_Cmin','target_Cmax','target_period')} for r in rows],
                  'optimization': {'success': bool(result.success), 'message': str(result.message),
                    'nfev': int(result.nfev), 'residual_evaluations': evaluations,
                    'baseline_squared_residual': baseline_loss,
                    'squared_residual': float(result.fun@result.fun)},
                  'cut_bounds': [lower, upper], 'lower_trace': model['trace'][0],
                  'lower_determinant': model['determinant'][0],
                  'rest_rhs': rhs(model, 0., base['rest_target']), 'holdout': heldout,
                  'duration_s': args.duration, 'training_mode': args.training}
        (out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print(case, 'fit', result.x.tolist(), report['optimization'], 'metrics', after, flush=True)


if __name__ == '__main__':
    main()
