"""Matched physical initial states and instantaneous IP3-only jumps.

DOP853 here validates the candidate model, not an event-driven speed claim.
Regional analytic propagation is provided separately in strict_pwa.py.
"""
import numpy as np
from scipy.integrate import solve_ivp
from .strict_pwa import lr_field


def simulate(case, initial, duration, tau=20., baseline=.2, jumps=(), model=None, dt=.2,
             rtol=2e-7, atol=2e-9, max_step=.5, enforce_domain=True):
    initial = np.asarray(initial,dtype=float)
    if (initial.shape != (3,) or not np.isfinite(initial).all() or duration <= 0
            or dt <= 0 or not np.isfinite(tau) or tau <= 0 or not 0 <= baseline <= 1.3):
        raise ValueError('Invalid initial state or protocol')
    events = sorted((float(t),float(delta)) for t,delta in jumps)
    if any(not 0 < t < duration or not np.isfinite(delta) for t,delta in events):
        raise ValueError('Jumps must be finite and strictly inside the simulation interval')
    if len({t for t,_ in events}) != len(events):
        raise ValueError('Combine simultaneous IP3 jumps explicitly')
    if model is None:
        def rhs(t,x):
            f = lr_field(x,case)
            return [f[0],f[1],-(x[2]-baseline)/tau]
    else:
        if model.case != case:
            raise ValueError('Model/protocol case mismatch')
        rhs = model.rhs(tau,baseline,stage_extrapolation=True)
    time_parts, state_parts, jump_log = [],[],[]
    state, left = initial.copy(),0.
    ip3_error = 0.
    evaluations = 0
    for right,delta in events+[(float(duration),0.)]:
        times = np.linspace(left,right,int(np.ceil((right-left)/dt))+1)
        sol = solve_ivp(rhs,(left,right),state,t_eval=times,method='DOP853',
                        rtol=rtol,atol=atol,max_step=max_step)
        if not sol.success or sol.t[-1] != right or not np.isfinite(sol.y).all():
            raise RuntimeError(sol.message)
        if model is not None and enforce_domain:
            for d,axis in enumerate(model.axes):
                if sol.y[d].min() < axis[0]-1e-7 or sol.y[d].max() > axis[-1]+1e-7:
                    raise RuntimeError('Accepted numerical trajectory left fitted box')
        expected_i = baseline+(state[2]-baseline)*np.exp(-(sol.t-left)/tau)
        ip3_error = max(ip3_error,float(np.max(abs(sol.y[2]-expected_i))))
        evaluations += sol.nfev
        time_parts.append(sol.t if not time_parts else sol.t[1:])
        state_parts.append(sol.y.T if not state_parts else sol.y.T[1:])
        state = sol.y[:,-1].copy()
        if right != duration:
            before = state.copy()
            state[2] += delta
            if not 0 <= state[2] <= 1.3:
                raise ValueError('IP3 jump outside fitted [0,1.3] domain')
            jump_log.append(dict(time=right,before=before.tolist(),after=state.tolist()))
        left = right
    return np.concatenate(time_parts),np.vstack(state_parts),dict(
        ip3_max_abs_error=ip3_error,rhs_evaluations=evaluations,jumps=jump_log)
