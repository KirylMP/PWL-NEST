"""C in micromolar, time in seconds. Legacy second state is NOT receptor h."""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[2]


def load_model(case):
    if case not in ('AM', 'FM'):
        raise ValueError('case must be AM or FM')
    return json.loads((ROOT / 'models' / f'{case}.json').read_text())


def coefficients(model, ip3):
    """Return thresholds, damping mu, restoring k, force intercept q, G offset g."""
    if not np.isfinite(ip3) or not 0 <= ip3 <= 1.3:
        raise ValueError('Legacy coefficient domain is 0 <= IP3 <= 1.3 uM')
    t = np.asarray(model['thresholds_C'], dtype=float)
    if model['case'] == 'AM':
        mu = np.array(model['mu_a']) + ip3 * np.array(model['mu_b'])
        d = ip3 - np.array(model['anchor_I'])
        force = np.array(model['force_a']) * d + model['force_quadratic'] * d**2
        tail = np.array(model['tail_k_a']) + ip3 * np.array(model['tail_k_b'])
        k = np.r_[tail[0], -np.diff(force) / np.diff(t), tail[1]]
        q = np.r_[force[0] + k[0]*t[0], force + k[1:]*t]
        guard = model['rest_guard']
        s = np.clip((ip3-guard['start_I'])/(guard['full_I']-guard['start_I']), 0, 1)
        mu[:-1] -= guard['damping'] * s*s*(3-2*s)
    else:
        mu = np.array(model['mu_intercept']) + ip3*np.array(model['mu_slope'])
        k = np.array(model['k_intercept']) + ip3*np.array(model['k_slope'])
        center = np.array(model['q_intercept']) + ip3*np.array(model['q_slope'])
        q = k*center
    g = np.r_[0., np.cumsum(-np.diff(mu)*t)]
    return t, mu, k, q, g


def pwa_rhs(model, ip3):
    t, mu, k, q, g = coefficients(model, ip3)
    def rhs(time, state):
        c, second = state
        j = np.searchsorted(t, c, side='right')
        if model['case'] == 'AM':
            return np.array([second + mu[j]*c + g[j], q[j]-k[j]*c])
        return np.array([second, q[j]-k[j]*c + mu[j]*second])
    return rhs


def lr_rhs(case, ip3):
    if case not in ('AM', 'FM') or not np.isfinite(ip3) or not 0 <= ip3 <= 1.3:
        raise ValueError('Invalid case or IP3')
    ker = .1 if case == 'AM' else .05
    receptor_q = 1.049*(ip3+.13)/(ip3+.9434)
    def rhs(time, state):
        c, h = state
        dc = (6*(ip3/(ip3+.13))**3*(c/(c+.08234))**3*h**3+.11)*(2-1.185*c)
        dc -= .9*c*c/(c*c+ker*ker)
        return np.array([dc, .2*(receptor_q-(receptor_q+c)*h)])
    return rhs


def trajectory(case, ip3, *, kind='pwa', duration=1600., dt=.1):
    """Numerical reference trajectories, NOT an event-driven performance benchmark.

    Large-excursion initialization follows legacy attracting-branch protocol.
    LR starts at (C0, .8); PWA starts at (C0, Cdot=0). These are not matched h states.
    """
    if duration <= 0 or dt <= 0:
        raise ValueError('duration and dt must be positive')
    model = load_model(case)
    c0 = .6 if case == 'AM' else 1.3
    if kind == 'pwa':
        rhs = pwa_rhs(model, ip3)
        _, mu, _, _, g = coefficients(model, ip3)
        j = np.searchsorted(model['thresholds_C'], c0, side='right')
        y0 = [c0, -mu[j]*c0-g[j] if case == 'AM' else 0.]
    elif kind == 'lr':
        rhs, y0 = lr_rhs(case, ip3), [c0, .8]
    else:
        raise ValueError('kind must be pwa or lr')
    times = np.linspace(0, duration, int(np.ceil(duration/dt))+1)
    sol = solve_ivp(rhs, (0, duration), y0, t_eval=times, method='DOP853',
                    rtol=2e-9, atol=2e-11, max_step=.2)
    if not sol.success or sol.t[-1] != duration or not np.isfinite(sol.y).all():
        raise RuntimeError(sol.message)
    return sol.t, sol.y.T
