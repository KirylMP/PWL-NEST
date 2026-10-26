"""Regional coordinates, explicit stationarity, and bounded descent recovery."""
import json
import time
import numpy as np
from scipy.optimize import least_squares,OptimizeResult
from regional_coordinates import RegionalCoordinates


def central_geometry_jacobian(objective,x):
    r=objective.residual(x);J=np.empty((len(r),len(x)))
    for j in range(len(x)):
        h=2e-6*max(1.,abs(x[j]));d=np.eye(len(x))[j]*h
        J[:,j]=(objective.residual(x+d)-objective.residual(x-d))/(2*h)
    return J


def solve_stage(objective,x0,max_nfev,*,label,log_path=None,freeze_transform=False,
                gtol=1e-6,rescue_evaluations=6,progress=None):
    geometry=getattr(objective,'geometry',objective)
    coordinates=RegionalCoordinates(geometry.thresholds,geometry.c_ref)
    lo,hi=coordinates.bounds();y0=coordinates.from_raw(x0)
    y0=np.clip(y0,lo+1e-9,hi-1e-9)
    active=np.arange(len(y0)-4 if freeze_transform else len(y0))
    best=None;history=[];previous=None;evaluations=0;started=time.perf_counter()
    def expand(v):
        y=y0.copy();y[active]=v;return y
    def record(y,r,admissible_override=None,error=None,evaluation_seconds=0.):
        nonlocal best,previous,evaluations
        evaluations+=1;loss=float(r@r)
        admissible=getattr(objective,'parts_value',{}).get('return_domain_admissible',True)
        if admissible_override is not None:admissible=admissible_override
        rank=(not admissible,loss)
        if best is None or rank<best[0]:best=(rank,y.copy(),r.copy())
        event=dict(event='objective',stage=label,evaluation=evaluations,loss=loss,admissible=bool(admissible),
                   step_norm=float(np.linalg.norm(y-previous)) if previous is not None else None,
                   seconds=time.perf_counter()-started,evaluation_seconds=evaluation_seconds,
                   freeze_transform=freeze_transform,regional_x=y.tolist())
        if hasattr(objective,'epsilon'):event['epsilon']=objective.epsilon
        if error is not None:event['error']=error
        if hasattr(objective,'report') and error is None:
            event['returns']=[dict(I=q['I'],regular_edges=q['regular_edges'],reference_edges=q['reference_edges'],
                admissible=q['admissible'],section=q['section'],error=q['error']) for q in objective.report]
        history.append(event);previous=y.copy()
        if log_path is not None:
            with log_path.open('a') as stream:stream.write(json.dumps(event,allow_nan=False)+'\n')
        if progress is not None:progress.evaluation(evaluations,loss,admissible,evaluation_seconds)
    def residual(v):
        y=expand(v);evaluation_started=time.perf_counter()
        if progress is not None:progress.activity('objective')
        try:
            r=objective.residual(coordinates.to_raw(y))
            if not np.all(np.isfinite(r)):raise FloatingPointError('nonfinite residual')
            record(y,r,evaluation_seconds=time.perf_counter()-evaluation_started);return r
        except (ValueError,FloatingPointError,OverflowError) as exc:
            if best is None:raise
            magnitude=max(1e6,10*float(np.max(abs(best[2]))))
            r=np.full_like(best[2],magnitude)
            record(y,r,False,repr(exc),time.perf_counter()-evaluation_started);return r
    def jacobian(v):
        y=expand(v);x=coordinates.to_raw(y)
        if progress is not None:progress.activity('jacobian')
        raw=objective.jac(x) if hasattr(objective,'jac') else central_geometry_jacobian(objective,x)
        matrix=(raw@coordinates.jacobian(y))[:,active]
        if log_path is not None:
            singular=np.linalg.svd(matrix,compute_uv=False)
            condition=float(np.exp(min(700.,np.log(max(singular[0],np.finfo(float).tiny))
                -np.log(max(singular[-1],np.finfo(float).tiny)))))
            gradient=matrix.T@objective.residual(x)
            event=dict(event='jacobian',stage=label,evaluation=evaluations,condition=condition,
                gradient_inf=float(np.max(abs(gradient))),singular_values=singular.tolist())
            with log_path.open('a') as stream:stream.write(json.dumps(event,allow_nan=False)+'\n')
        return matrix
    result=least_squares(residual,y0[active],jac=jacobian,bounds=(lo[active],hi[active]),
        method='trf',x_scale=np.ones(len(active)),max_nfev=max_nfev,
        ftol=None,xtol=None,gtol=gtol)
    # Retain the lowest admissible evaluated point, including stage entry.
    if progress is not None:progress.activity('retained_point_check')
    y=best[1];r=objective.residual(coordinates.to_raw(y));J=jacobian(y[active])
    g=J.T@r
    def optimality(v,gradient):
        dist=np.where(gradient<0,hi[active]-v,v-lo[active])
        return float(np.max(abs(gradient*dist)))
    # A bounded, verified line search supplies a fallback when a numerically
    # small trust-region step leaves an available direction of descent.
    rescues=0
    if optimality(y[active],g)>gtol and max_nfev>3:
        direction=-g/max(1.,float(np.max(abs(g))))
        for power in range(rescue_evaluations):
            proposal=np.clip(y[active]+(1e-3*.2**power)*direction,lo[active]+1e-10,hi[active]-1e-10)
            residual(proposal);rescues+=1
    y=best[1];raw_x=coordinates.to_raw(y);r=objective.residual(raw_x);J=jacobian(y[active]);g=J.T@r
    opt=optimality(y[active],g)
    domain_admissible=getattr(objective,'parts_value',{}).get('return_domain_admissible',True)
    stationary=opt<=gtol and not freeze_transform and domain_admissible
    singular=np.linalg.svd(J,compute_uv=False)
    condition=float(np.exp(min(700.,np.log(max(singular[0],np.finfo(float).tiny))
                              -np.log(max(singular[-1],np.finfo(float).tiny)))))
    # `success` in v5 means checked stationarity, never merely a small step.
    if progress is not None:progress.stage_done()
    return OptimizeResult(x=raw_x,fun=r,jac=J,grad=g,cost=float(.5*(r@r)),nfev=evaluations,
        optimality=opt,success=stationary,status=1 if stationary else 0,
        message='stationarity checked in regional coordinates' if stationary else 'evaluation budget exhausted or nonstationary termination',
        solver_success=bool(result.success),solver_status=int(result.status),solver_message=str(result.message),
        converged_stationary=stationary,active_stationarity=opt<=gtol,transform_frozen=freeze_transform,
        jacobian_condition=condition,singular_values=singular.tolist(),
        rescue_evaluations=rescues,iteration_history=history,regional_x=y.tolist())
