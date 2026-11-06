"""Add one continuous affine repelling zone below a calcium cut.

The old vector field is retained exactly at and above the cut. A fixed finite
gain is a local safeguard, not a proof of non-negativity for arbitrary states.
"""
import copy
import numpy as np

def add_guard(base,cut=.025,gain=20.):
    if not np.isfinite(cut) or not np.isfinite(gain) or not 0<cut<min(base['physical_thresholds_C']) or gain<=0:
        raise ValueError('invalid guard cut or gain')
    out=copy.deepcopy(base);out['physical_thresholds_C']=[float(cut),*base['physical_thresholds_C']]
    for key in ('alpha','beta'):
        coeff=np.asarray(base[key],float).copy();delta=-gain/base['a'] if key=='alpha' else 0.
        coeff[0]+=delta*cut;coeff[1]-=delta
        out[key]=[float(x) for x in np.r_[coeff[:3],delta,coeff[3:]]]
    trace=[];det=[]
    for j in range(len(out['physical_thresholds_C'])+1):
        n=out['alpha'][1]+sum(out['alpha'][3:3+j]);h=out['beta'][1]+sum(out['beta'][3:3+j])
        trace.append(float(-out['a']*n-out['b']));det.append(float(out['a']*out['b']*(n-h)))
    out['trace']=trace;out['determinant']=det;out['lower_guard']={'cut_C':float(cut),'gain_per_s':float(gain),'formula':'dC_new=dC_old+gain*(cut-C)_+'}
    return out
