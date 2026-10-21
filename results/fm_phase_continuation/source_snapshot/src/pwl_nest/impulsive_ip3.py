"""Exact exponential trace driven by Dirac impulses of specified area.

dI = -(I-baseline)/tau dt + sum_k amplitude_k delta(t-t_k) dt.
The distribution is represented by event times and jump amplitudes, never by
a tall finite pulse or an ordinary floating-point state value.
"""
import numpy as np


def exponential_ip3(times,initial,tau,baseline,impulses=(),*,side='right'):
    """Closed-form trace; side selects I(t_k-) or I(t_k+) at exact events."""
    t=np.asarray(times,dtype=float)
    if (side not in ('left','right') or not np.isfinite(t).all() or np.any(t<0)
            or not np.isfinite([initial,tau,baseline]).all() or tau<=0):
        raise ValueError('Invalid IP3 trace arguments')
    events=np.asarray(list(impulses),dtype=float)
    if events.size and (events.ndim!=2 or events.shape[1]!=2 or not np.isfinite(events).all() or np.any(events[:,0]<0)):
        raise ValueError('Require nonnegative finite event times and finite amplitudes')
    result=baseline+(initial-baseline)*np.exp(-t/tau)
    for when,amplitude in events.reshape(-1,2):
        active=t>=when if side=='right' else t>when
        # Avoid evaluating exp of a large positive number before future events.
        elapsed=np.maximum(t-when,0.)
        result=result+np.where(active,amplitude*np.exp(-elapsed/tau),0.)
    return result


def regular_ip3_velocity(ip3,tau,baseline):
    """Regular derivative between impulses; does not include delta masses."""
    if not np.isfinite(tau) or tau<=0:raise ValueError('Require positive finite tau')
    return -(np.asarray(ip3)-baseline)/tau
