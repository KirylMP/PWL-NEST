"""Cached affine flow backend for sampled fitting trajectories."""
import ctypes as ct
import numpy as np
from .models import ROOT


class FlowDynamic:
    def __init__(self):
        self.lib=ct.CDLL(str(ROOT/'.build/fit_flow.so'))
        ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        self.lib.fit_flow.argtypes=[ptr,ct.c_int,ptr,ct.c_int,ptr,ptr,
            ct.c_double,ct.c_double,ct.c_double,ct.c_int,ct.c_double,
            ct.c_double,ct.c_double,ct.c_double,ct.c_double,ptr]
        self.lib.fit_flow.restype=ct.c_int

    def run(self,model,initial,duration,tau,baseline,outputs=361,maxdt=.02):
        if model.has_state_mix or np.any(model.threshold_slopes):
            raise ValueError('Flow fitting backend requires vertical C thresholds')
        result=np.empty((outputs,2))
        code=self.lib.fit_flow(np.ascontiguousarray(model.latent.thresholds),model.latent.region_count,
            np.ascontiguousarray(model.input_thresholds),len(model.input_thresholds),
            np.ascontiguousarray(model.geometry_coefficients().ravel()),
            np.ascontiguousarray(model.to_latent(initial)),baseline,tau,duration,outputs,maxdt,
            model.coordinate.lower,model.coordinate.guard_upper,model.gate.lower,model.gate.upper,result)
        return None if code else result
