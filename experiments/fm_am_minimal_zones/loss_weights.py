"""Common scaling for Poincare preparation and final closed-cycle losses."""
import math
import os


def poincare_scale():
    scale=float(os.environ.get('POINCARE_WEIGHT_SCALE','1'))
    if not math.isfinite(scale) or scale<=0:
        raise ValueError('POINCARE_WEIGHT_SCALE must be finite and positive')
    return scale


def closed_components(geometry,equilibria,cycles_raw,base_weight,scale):
    weight=base_weight*scale
    return dict(geometry=float(geometry),equilibrium_sets=float(equilibria),
                cycles_raw=float(cycles_raw),cycle_weight=weight,
                cycles_weighted=float(weight*cycles_raw),
                total=float(geometry+equilibria+weight*cycles_raw))
