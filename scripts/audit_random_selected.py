"""Spectral and finite-difference return-map checks of the working selection."""
import hashlib
import json
from pathlib import Path

import numpy as np
from continue_physical_fixed import Orbit, REGIONS
from check_selected_return_map import mapped

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'paper/data/random_selected/manifest.json'
OUT = ROOT / 'paper/data/random_selected/audit.json'

def main():
    result = {}
    for case, meta in json.loads(MANIFEST.read_text())['models'].items():
        path = ROOT / meta['candidate_path']
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != meta['candidate_sha256']:
            raise ValueError('candidate hash mismatch')
        d = json.loads(raw)
        report = json.loads((ROOT / meta['report_path']).read_text())
        orb = Orbit(d)
        T = []; D = []
        for j in range(3):
            n = d['alpha'][1] + sum(d['alpha'][3:3+j])
            h = d['beta'][1] + sum(d['beta'][3:3+j])
            T.append(-d['a']*n-d['b'])
            D.append(d['a']*d['b']*(n-h))
        T = np.asarray(T); D = np.asarray(D); disc = T*T - 4*D
        fold = report['best']['fold']; times = np.array(fold['times']); I = float(fold['I'])
        _, _, q, _ = orb.match(times, I, True); q0 = float(q[0])
        record = {'candidate_sha256': meta['candidate_sha256'], 'trace': T.tolist(),
                  'determinant': D.tolist(), 'discriminant': disc.tolist(), 'I_fold': I,
                  'stored_trace_max_error':float(np.max(np.abs(T-np.array(d['trace'])))),
                  'stored_determinant_max_error':float(np.max(np.abs(D-np.array(d['determinant'])))),
                  'matching_max': fold['matching_max'], 'logM': float(orb.T[list(REGIONS)] @ times),
                  'min_abs_crossing_speed': min(map(abs, fold['speeds'])), 'steps_q': [], 'steps_I': []}
        try:
            value, _ = mapped(orb, times, I, q0)
            record['return_residual'] = value-q0
            for h in (1e-5, 3e-6, 1e-6, 3e-7, 1e-7, 3e-8):
                try:
                    plus, _ = mapped(orb, times, I, q0+h); minus, _ = mapped(orb, times, I, q0-h)
                    record['steps_q'].append({'h':h, 'G_q':(plus-minus)/(2*h)-1, 'G_qq':(plus-2*value+minus)/(h*h)})
                except (ValueError, RuntimeError, OverflowError) as exc: record['steps_q'].append({'h':h, 'error':str(exc)})
            for h in (1e-5, 1e-6, 1e-7, 1e-8, 1e-9):
                try:
                    plus, _ = mapped(orb, times, I+h, q0); minus, _ = mapped(orb, times, I-h, q0)
                    record['steps_I'].append({'h':h, 'G_I':(plus-minus)/(2*h)})
                except (ValueError, RuntimeError, OverflowError) as exc: record['steps_I'].append({'h':h, 'error':str(exc)})
        except (ValueError, RuntimeError, OverflowError) as exc: record['return_error'] = str(exc)
        result[case] = record
        print(case, json.dumps(record), flush=True)
    OUT.write_text(json.dumps({'scope':'finite-difference checks for one itinerary; no completeness proof', 'results':result}, indent=2, allow_nan=False)+'\n')

if __name__ == '__main__': main()
