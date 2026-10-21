"""Diagnostic time translation; input times and model parameters stay fixed."""
import numpy as np
from scipy.optimize import minimize_scalar


def single_shift_comparison(t,reference,candidate,fit_window,eval_window=None,max_shift=5.):
    """Fit one lag on a fixed window and evaluate on another fixed window.

    Positive lag means the candidate is late: compare candidate(t+lag) to
    reference(t). Fixed windows avoid favouring a lag by discarding a bad peak.
    No amplitude scaling, offset removal, or time stretching is performed.
    """
    t,reference,candidate=map(lambda x:np.asarray(x,dtype=float),(t,reference,candidate))
    if (t.ndim!=1 or len(t)<5 or reference.shape!=t.shape or candidate.shape!=t.shape
            or not np.isfinite(np.r_[t,reference,candidate,max_shift]).all()
            or np.any(np.diff(t)<=0) or max_shift<=0):raise ValueError('Invalid phase comparison')
    eval_window=fit_window if eval_window is None else eval_window
    def mask(window):
        left,right=window
        if left<t[0]+max_shift-1e-10 or right>t[-1]-max_shift+1e-10 or right<=left:
            raise ValueError('Window must support every allowed translation')
        chosen=(t>=left)&(t<=right)
        if chosen.sum()<5:raise ValueError('Too few evaluation samples')
        return chosen
    fit,check=mask(fit_window),mask(eval_window)
    def error(lag,selected):
        delta=np.interp(t[selected]+lag,t,candidate)-reference[selected]
        return float(np.mean(delta*delta))
    grid=np.linspace(-max_shift,max_shift,201)
    values=np.array([error(lag,fit) for lag in grid]);index=int(np.argmin(values))
    lower,upper=grid[max(0,index-1)],grid[min(len(grid)-1,index+1)]
    optimum=minimize_scalar(lambda lag:error(lag,fit),bounds=(lower,upper),method='bounded',options={'xatol':1e-8})
    lag=min([float(optimum.x),float(grid[index]),0.],key=lambda value:error(value,fit))
    return dict(shift_s=lag,raw_fit_rmse=float(np.sqrt(error(0.,fit))),
        aligned_fit_rmse=float(np.sqrt(error(lag,fit))),raw_eval_rmse=float(np.sqrt(error(0.,check))),
        aligned_eval_rmse=float(np.sqrt(error(lag,check))),fit_window=list(fit_window),
        eval_window=list(eval_window),max_shift_s=max_shift,at_search_boundary=bool(abs(lag)>.99*max_shift),
        convention='positive shift means candidate is late; compare candidate(t+shift) to LR(t)')
