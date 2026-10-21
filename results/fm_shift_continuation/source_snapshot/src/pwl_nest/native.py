"""Rebuilt legacy native engines for autonomous observable reproduction."""
import ctypes as ct
import numpy as np
from .models import ROOT, load_model, coefficients


def observables(case, ip3, duration=2400.):
    if duration < 800:
        raise ValueError('Legacy statistics require at least 800 seconds')
    model = load_model(case)
    theta, mu, k, q, _ = coefficients(model, ip3)
    if np.min(np.abs(k)) < 1e-7:
        raise ValueError('Legacy native propagator is singular near k=0')
    library = ct.CDLL(str(ROOT / '.build' / f'{case.lower()}_engine.so'))
    ptr = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    library.simulate.argtypes = [ct.c_double, ptr, ptr, ptr, ptr, ptr, ct.c_int,
                                 ct.c_double, ct.c_double, ct.c_double, ptr]
    library.simulate.restype = ct.c_int
    # AM engine uses force intercept; FM engine expects center intercept+slope*I.
    qi = q if case == 'AM' else np.array(model['q_intercept'])
    qs = np.zeros_like(q) if case == 'AM' else np.array(model['q_slope'])
    arrays = [np.ascontiguousarray(x, dtype=np.float64) for x in (theta, mu, k, qi, qs)]
    out = np.zeros(6)
    code = library.simulate(ip3, *arrays, len(k), .6 if case == 'AM' else 1.3,
                            0., duration, out)
    if code or not np.isfinite(out[[0, 1, 3, 4, 5]]).all():
        raise RuntimeError(f'{case} native simulation failed: code={code}')
    return dict(Cmin=float(out[0]), Cmax=float(out[1]), period=float(out[2]),
                amplitude_ratio=float(out[3]), Cmin_full=float(out[4]))
