#!/usr/bin/env python3
"""Recompute LR target cycles, retaining deviations from archived fit targets."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
SEEDS = ROOT / 'astrocyte_lab/experiments/no_fold_three_fits/seeds'
OUT = ROOT / 'paper/local_results/lr_target_recalculation.json'

def compute(task):
    K, ref = task
    I = ref['I']
    def rhs(t, y):
        C, h = y
        Q = 1.049 * (I + .13) / (I + .9434)
        dc = (6*(I/(I+.13))**3*(C/(C+.08234))**3*h**3+.11)*(2-1.185*C)-.9*C*C/(C*C+K*K)
        return [dc, .2*(Q-(Q+C)*h)]
    def peak(t, y): return rhs(t, y)[0]
    peak.direction = -1
    sol = solve_ivp(rhs, (0, 450), ref['initial'], method='DOP853', rtol=2e-9,
                    atol=2e-11, max_step=.08, dense_output=True, events=peak)
    peaks = sol.t_events[0]; peaks = peaks[peaks > 225]
    result = dict(K=K, I=I, split=ref['split'], valid=False)
    if not sol.success or len(peaks) < 3:
        result['reason'] = 'integration failed or fewer than three late peaks'
        return result
    T = float(peaks[-1]-peaks[-2])
    if abs(T-(peaks[-2]-peaks[-3])) > max(.3,.03*T):
        result['reason'] = 'late peak intervals did not converge'
        return result
    C = sol.sol(peaks[-2]+T*np.arange(len(ref['shape']))/len(ref['shape']))[0]
    if not np.isfinite(C).all() or max(C)-min(C) <= .001:
        result['reason'] = 'nonfinite or small cycle'
        return result
    old = np.asarray(ref['shape'])
    result.update(valid=True, period=T, Cmin=float(min(C)), Cmax=float(max(C)),
                  shape=C.tolist(), initial=ref['initial'],
                  differences=dict(period=T-ref['period'], Cmin=float(min(C))-ref['Cmin'],
                                   Cmax=float(max(C))-ref['Cmax'],
                                   shape_rmse=float(np.sqrt(np.mean((C-old)**2)))))
    return result

def main():
    p = argparse.ArgumentParser(); p.add_argument('--workers', type=int, default=2); a=p.parse_args()
    tasks=[]
    for K in (.05,.075,.1):
        data=json.loads((SEEDS/f'K_{K:.3f}_validation.json').read_text())
        tasks.extend((K, row['LR']) for row in data['cycles'])
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        rows=list(pool.map(compute,tasks))
    OUT.write_text(json.dumps(dict(method='DOP853, 450 s, rtol 2e-9, atol 2e-11, max_step .08 s, peak-aligned 64 samples',
                                   note='Compared with archived seed targets used during fitting; no refit implied.', results=rows),indent=2)+'\n')
    print('Saved',OUT,'valid',sum(r['valid'] for r in rows),'/',len(rows))
if __name__ == '__main__': main()
