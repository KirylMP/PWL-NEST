"""Compare selected PWA NEST trajectories with an independent SciPy IVP."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from benchmark_constant_ip3 import ROOT, KER, load_case, parameters


def rhs(data, I, y):
    C, q = y
    terms = np.r_[1., C, I, np.maximum(C - np.asarray(data['physical_thresholds_C']), 0.)]
    return [data['a'] * (q - np.dot(data['alpha'], terms)),
            data['b'] * (np.dot(data['beta'], terms) - q)]


def run_case(case, I, initial, args):
    import nest
    from pwl_nest.nest_adapter import install
    data, sha = load_case(case, 'selected')
    nest.ResetKernel()
    nest.SetKernelStatus({'resolution': args.dt_ms, 'local_num_threads': 1, 'print_time': False})
    install()
    node = nest.Create('astrocyte_lr_pwa', params=parameters(data, I, *initial))
    meter = nest.Create('multimeter', params={'interval': args.interval_ms,
                                               'record_from': ['Ca_astro', 'h_IP3R', 'IP3']})
    nest.Connect(meter, node)
    nest.Simulate(args.duration_s * 1000.)
    events = meter.get('events')
    times = np.asarray(events['times'], dtype=float) / 1000.
    if not len(times) or not np.all(np.diff(times) > 0):
        raise RuntimeError('No samples or non-increasing NEST timestamps')
    sol = solve_ivp(lambda t, y: rhs(data, I, y), (0., args.duration_s), initial,
                    method='DOP853', rtol=1e-10, atol=1e-12, max_step=.05, dense_output=True)
    if not sol.success:
        raise RuntimeError(sol.message)
    ref = sol.sol(times)
    observed = np.array([events['Ca_astro'], events['h_IP3R']], dtype=float)
    if not np.all(np.isfinite(observed)):
        raise ValueError('Non-finite NEST samples')
    errors = np.max(np.abs(observed - ref), axis=1)
    iperr = float(np.max(np.abs(np.asarray(events['IP3']) - I)))
    return {'case': case, 'I': I, 'initial_C': float(initial[0]), 'initial_q': float(initial[1]),
            'candidate_sha256': sha, 'samples': len(times), 'first_time_s': float(times[0]),
            'max_abs_C': float(errors[0]), 'max_abs_q': float(errors[1]), 'max_abs_IP3': iperr,
            'region_crossings': int(np.sum(node.get('region_crossings'))),
            'passed': bool(max(errors) <= args.tolerance and iperr <= 1e-10)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases', nargs='+', choices=sorted(KER), default=['AM', 'AFM', 'FM'])
    p.add_argument('--duration-s', type=float, default=25.)
    p.add_argument('--dt-ms', type=float, default=.1)
    p.add_argument('--interval-ms', type=float, default=10.)
    p.add_argument('--tolerance', type=float, default=2e-3)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if min(a.duration_s, a.dt_ms, a.interval_ms, a.tolerance) <= 0 or any(
            abs(x / a.dt_ms - round(x / a.dt_ms)) > 1e-7 for x in (a.duration_s * 1000, a.interval_ms)):
        p.error('Duration and recording interval must be positive multiples of resolution')
    if a.output.exists():
        raise FileExistsError(a.output)
    import nest
    module = ROOT / '.build/pwl_astro_module.so'
    if not module.is_file():
        raise FileNotFoundError(module)
    records = []
    for case in a.cases:
        data, _ = load_case(case)
        lower, upper = data['physical_thresholds_C']
        fold = data['reference_fold']['I']
        for I in (max(0., fold - .08), fold - .01, fold + .01):
            for C in (float(data['rest_target'][0]), (lower + upper) / 2, upper + .08):
                terms = np.r_[1., C, I, np.maximum(C - np.asarray([lower, upper]), 0.)]
                q = float(np.dot(data['alpha'], terms))
                try:
                    item = run_case(case, I, (C, q), a)
                except Exception as exc:
                    item = {'case': case, 'I': I, 'initial_C': C, 'initial_q': q,
                            'passed': False, 'error': f'{type(exc).__name__}: {exc}'}
                records.append(item)
                print(case, f'I={I:.6f}', f'C0={C:.6f}', 'OK' if item['passed'] else 'FAIL',
                      item.get('max_abs_C'), item.get('max_abs_q'), item.get('region_crossings'), item.get('error', ''), flush=True)
    out = {'nest_version': nest.__version__, 'module_sha256': hashlib.sha256(module.read_bytes()).hexdigest(),
           'settings': vars(a) | {'output': str(a.output)}, 'records': records,
           'scope': 'finite trajectories, fixed IP3, physical C and hidden PWA q; not a bifurcation or speed test'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    print('Saved', a.output)
    if not all(x['passed'] for x in records):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
