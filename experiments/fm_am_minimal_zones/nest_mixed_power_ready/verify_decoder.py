#!/usr/bin/env python3
"""Verify that upstream Candidate decoder reproduces saved regional spectra."""
import json, os, sys
from pathlib import Path
import numpy as np

base = Path(__file__).resolve().parent
root = Path(os.environ.get('PWL_NEST_ROOT', os.getcwd())).resolve()
bench = root / 'experiments/fm_am_minimal_zones/nest_benchmark'
if not (bench / 'latent_flow.py').exists():
    raise SystemExit(f'Cannot find research branch at {bench}; set PWL_NEST_ROOT.')
sys.path.insert(0, str(bench))
from latent_flow import Candidate  # Uses the actual research branch decoder.

for name in ['AM_3zones', 'FM_4zones_6', 'FM_4zones_7']:
    fn = base / 'models' / (name + '.json')
    saved = json.loads(fn.read_text())
    actual = Candidate(fn)
    regional = actual.coefficients
    A = regional[:, :, 1:3]
    trace = np.trace(A, axis1=1, axis2=2)
    determinant = np.linalg.det(A)
    trace_err = float(np.max(np.abs(trace - np.array(saved['trace']))))
    determinant_err = float(np.max(np.abs(determinant - np.array(saved['determinant']))))
    reencoded_thresholds = actual.inverse(actual.z_thresholds)
    threshold_err = float(np.max(np.abs(reencoded_thresholds - actual.thresholds)))
    print(f'{name}: traces Δ={trace_err:.3g}, determinants Δ={determinant_err:.3g}, '
          f'thresholds Δ={threshold_err:.3g}, z={actual.z_thresholds.tolist()}')
    assert trace_err < 1e-11 and determinant_err < 1e-11 and threshold_err < 1e-11
print('ALL MODELS DECODED CONSISTENTLY')
