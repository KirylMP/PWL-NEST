"""Paired NEST timings and independent per-model trajectory accuracy, constant IP3.

Run separately from other benchmarks. The two errors compare each implementation
against its own ODE, not LR against PWA. Times are seconds except NEST *_ms.
"""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import solve_ivp

from benchmark_constant_ip3 import ROOT, KER, load_case, parameters
from validate_selected_nest import rhs as pwa_rhs


def lr_rhs(case, I, y):
    # NEST v3.10 clips the calcium used in its RHS to [0, Ca_tot=2].
    c, h = y
    c = np.clip(c, 0., 2.)
    q = 1.049 * (I + .13) / (I + .9434)
    dc = (6. * (I/(I+.13))**3 * (c/(c+.08234))**3 * h**3 + .11) * (2.-1.185*c)
    dc -= .9*c*c/(c*c + KER[case]**2)
    return np.array([dc, .2*(q-(q+c)*h)])


def reference(case, kind, data, I, initial, times, *, fine):
    if kind == 'LR':
        field = lambda t, y: lr_rhs(case, I, y)
    else:
        field = lambda t, y: pwa_rhs(data, I, y)
    sol = solve_ivp(field, (0., float(times[-1])), initial, method='DOP853',
                    rtol=2e-12 if fine else 1e-10,
                    atol=2e-14 if fine else 1e-12,
                    max_step=.025 if fine else .05, t_eval=times)
    if not sol.success or sol.y.shape != (2, len(times)) or not np.isfinite(sol.y).all():
        raise RuntimeError(f'{case} {kind}: reference failed: {sol.message}')
    return sol.y


def error_metrics(actual, expected):
    diff = np.asarray(actual, float) - np.asarray(expected, float)
    if not np.isfinite(diff).all():
        raise ValueError('non-finite reference or diagnostic')
    return {'C_max_abs': float(np.max(np.abs(diff[0]))),
            'C_rmse': float(np.sqrt(np.mean(diff[0]**2))),
            'hidden_max_abs': float(np.max(np.abs(diff[1]))),
            'hidden_rmse': float(np.sqrt(np.mean(diff[1]**2)))}


