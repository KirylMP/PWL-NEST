"""C++ analytic/event-driven solver for vertical-threshold 3-D PWA strips."""
import ctypes as ct
from pathlib import Path

import numpy as np

from .models import ROOT


class EventSolver:
    """Propagate the active affine region exactly and localize C-strip exits.

    The current implementation supports identity state coordinates and vertical
    calcium thresholds. It handles exponential IP3 and instantaneous IP3 jumps.
    ``scan_dt`` controls the maximum interval used to bracket a first exit; once
    bracketed, the crossing time is refined by bisection on the exact flow.
    """
    def __init__(self, library=None):
        path = Path(library or ROOT / '.build/pwa_event.so')
        self.lib = ct.CDLL(str(path))
        ptr = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
        iptr = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
        self.lib.pwla_simulate.argtypes = [ptr, ct.c_int, ptr, ptr,
            ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_double,
            ptr, ptr, ct.c_int, ptr, ct.c_int, iptr]
        self.lib.pwla_simulate.restype = ct.c_int

    def run(self, model, initial, duration, tau, baseline, jumps=(), output_dt=.05, scan_dt=.02):
        if model.has_geometry or model.has_state_mix or model.coordinate.kind != 'identity' or model.h_kind != 'identity':
            raise ValueError('Solver currently requires un-mixed identity C/h coordinates and no I-dependent geometry')
        if np.any(model.threshold_slopes):
            raise ValueError('Oblique thresholds are not supported by this solver')
        local = np.ascontiguousarray(model.latent.local_coefficients(), dtype=np.float64)
        thresholds = np.ascontiguousarray(model.latent.thresholds, dtype=np.float64)
        y0 = np.ascontiguousarray(model.to_latent(np.asarray(initial, dtype=float)), dtype=np.float64)
        events = sorted((float(t), float(d)) for t, d in jumps)
        event_times = np.ascontiguousarray([x[0] for x in events], dtype=np.float64)
        event_deltas = np.ascontiguousarray([x[1] for x in events], dtype=np.float64)
        capacity = int(np.ceil(duration / output_dt)) + 2
        out = np.empty((capacity, 3), dtype=np.float64)
        counters = np.zeros(3, dtype=np.int32)
        count = self.lib.pwla_simulate(thresholds, model.latent.region_count,
            np.ascontiguousarray(local.ravel()), y0, baseline, tau, duration,
            output_dt, scan_dt, event_times, event_deltas, len(events),
            np.ascontiguousarray(out.ravel()), capacity, counters)
        if count < 0:
            raise RuntimeError(f'C++ PWA event solver failed with status {count}')
        y = out[:count].copy()
        indices = np.arange(count, dtype=float)
        t = np.minimum(indices * output_dt, duration)
        if count > 1 and t[-1] < duration - 1e-12:
            raise RuntimeError('C++ solver did not return the terminal state')
        physical = model.to_physical(y)
        return t, physical, dict(affine_flows=int(counters[0]),
            calcium_region_crossings=int(counters[1]),ip3_jumps=int(counters[2]),
            scan_dt=float(scan_dt),output_dt=float(output_dt))
