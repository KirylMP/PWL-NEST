"""Audited moving-section FM fit; all final candidates use one common grid."""
import json,time
import numpy as np
from scipy.optimize import OptimizeResult
from guarded_optimizer import solve_stage
from regional_coordinates import RegionalCoordinates
from cycle_training_audit import nested_nodes,audit_candidate
from trial_progress import TrialProgress,optimizer_plan


def model_from_vector(x,objective):
    n=objective.m+3
    return dict(case=objective.run['case'],K_ER=objective.run['K_ER'],
        physical_thresholds_C=objective.thresholds.tolist(),alpha=x[:n].tolist(),beta=x[n:2*n].tolist(),
        a=float(np.exp(x[2*n])),b=float(np.exp(x[2*n+1])),
        transform=dict(p=float(x[-4]),e=float(x[-3]),n=float(np.exp(x[-2])),k=float(x[-1]),C_ref=objective.c_ref))


def finalize(objective,result,checkpoint_dir=None):
    """Stationarity and audit both refer to the shared final objective."""
    progress=getattr(objective,'progress',None)
    if progress is not None:progress.phase('final_stationarity')
    objective.set_levels(objective.cfg['I_nodes'])
    objective.set_nodes(nested_nodes(objective.cfg['u_nodes'],objective.cfg['selection_nodes']))
    objective.epsilon=objective.epsilon_schedule[-1]
    x=result.x;parts=objective.parts(x).copy();r=objective.residual(x)
    coordinates=RegionalCoordinates(objective.thresholds,objective.c_ref);y=coordinates.from_raw(x)
    Jraw=objective.jac(x);J=Jraw@coordinates.jacobian(y);g=J.T@r;lo,hi=coordinates.bounds()
    result.raw_gradient_inf=float(np.max(abs(Jraw.T@r)))
    result.regional_x=y.tolist()
    dist=np.where(g<0,hi-y,y-lo);optimality=float(np.max(abs(g*dist)))
    result.bounds_feasible=bool(np.all(y>=lo) and np.all(y<=hi))
    gtol=objective.cfg.get('stationarity_gtol',1e-6)
    result.stationarity_regular=all(q['section'] is not None and not q['section'].get('equilibrium_on_threshold',False)
        for q in objective.report)
    result.converged_stationary=optimality<=gtol and result.stationarity_regular and result.bounds_feasible;result.success=result.converged_stationary
    result.optimality=optimality;result.cost=float(.5*(r@r));result.fun=r;result.jac=J
    singular=np.linalg.svd(J,compute_uv=False)
    result.jacobian_condition=float(np.exp(min(700.,np.log(max(singular[0],np.finfo(float).tiny))
        -np.log(max(singular[-1],np.finfo(float).tiny)))))
    result.singular_values=singular.tolist()
    result.status=1 if result.converged_stationary else 0
    if not result.converged_stationary:result.message='nonstationary on the common final grid; inspect iteration log and audit'
    result.return_report=objective.report
    if progress is not None:progress.phase('training_audit')
    audit=audit_candidate(objective.run_dir,objective.spec,model_from_vector(x,objective),
        label=checkpoint_dir.name if checkpoint_dir is not None else None,checkpoint_dir=checkpoint_dir,
        progress=progress)
    result.training_audit=audit
    parts.update(cycle_resolution_admissible=audit['parts']['cycle_resolution_admissible'],
                 cycle_coverage_admissible=audit['parts']['cycle_coverage_admissible'])
    parts['return_domain_admissible'] &= audit['parts']['return_domain_admissible']
    result.gradient_stationarity=bool(result.converged_stationary)
    numerically_admissible=all(parts[k] for k in
        ['return_domain_admissible','cycle_resolution_admissible','cycle_coverage_admissible'])
    result.converged_stationary &= numerically_admissible;result.success=result.converged_stationary
    result.status=1 if result.success else 0
    if not numerically_admissible:result.message='cycle coverage/resolution audit failed; diagnostic candidate only'
    result.parts_value=parts
    result.resolution_history=objective.resolution_history
    if checkpoint_dir is not None:
        (checkpoint_dir/'training_audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n')
        (checkpoint_dir/'collocation_history.json').write_text(json.dumps(objective.resolution_history,indent=2)+'\n')
    return result


def optimize_trial_v5(objective,x0,max_nfev,checkpoint_dir=None):
    label=checkpoint_dir.name if checkpoint_dir is not None else 'trial'
    with TrialProgress(objective.run_dir/'logs' if checkpoint_dir is not None else None,label) as progress:
        objective.progress=progress
        try:return _optimize_trial_v5(objective,x0,max_nfev,checkpoint_dir,progress)
        finally:del objective.progress


def _optimize_trial_v5(objective,x0,max_nfev,checkpoint_dir,progress):
    started=time.perf_counter();history=[]
    if checkpoint_dir is not None:checkpoint_dir.mkdir(parents=True,exist_ok=True)
    log=checkpoint_dir/'iterations.jsonl' if checkpoint_dir is not None else None
    label=checkpoint_dir.name if checkpoint_dir is not None else 'trial'
    geometry=objective.geometry
    print(f'{label}: regional geometry fit',flush=True)
    first=max(1,max_nfev//2)
    plans=optimizer_plan(objective.cfg,objective.epsilon_schedule,max_nfev)
    progress.configure(plans);progress.stage(0)
    result=solve_stage(geometry,x0,first,label='precycle_fixed_transform',log_path=log,
        freeze_transform=True,rescue_evaluations=0,progress=progress)
    history.append(dict(stage='precycle_fixed_transform',nfev=result.nfev,loss=float(2*result.cost),
        success=bool(result.success),converged_stationary=bool(result.converged_stationary),transform_frozen=True))
    if max_nfev>1:
        progress.stage(1)
        result=solve_stage(geometry,result.x,max_nfev-first,label='precycle_joint',log_path=log,rescue_evaluations=0,progress=progress)
        history.append(dict(stage='precycle_joint',nfev=result.nfev,loss=float(2*result.cost),
            success=bool(result.success),converged_stationary=bool(result.converged_stationary),transform_frozen=False))
    for index,epsilon in enumerate(objective.epsilon_schedule):
        progress.stage((2 if max_nfev>1 else 1)+index)
        coarse_stage=index<min(2,len(objective.epsilon_schedule)-1)
        levels=objective.cfg.get('coarse_I_nodes',objective.cfg['I_nodes']) if coarse_stage else objective.cfg['I_nodes']
        objective.set_levels(levels);objective.epsilon=epsilon
        objective.refine_nodes(result.x,objective.cfg['max_stage_nodes'])
        progress.units(len(levels)*len(objective.nodes))
        print(f'{label}: epsilon={epsilon}, I nodes={len(levels)}, u nodes={len(objective.nodes)}',flush=True)
        result=solve_stage(objective,result.x,objective.cfg['max_nfev_per_epsilon'],
            label=f'cycles_{epsilon}',log_path=log,freeze_transform=index==0 and len(objective.epsilon_schedule)>1,
            gtol=objective.cfg['stationarity_gtol'],rescue_evaluations=objective.cfg.get('rescue_evaluations',6),progress=progress)
        parts=objective.parts(result.x)
        event=dict(stage='cycles',epsilon=epsilon,nfev=result.nfev,loss=parts['full_total'],
            success=bool(result.success),converged_stationary=bool(result.converged_stationary),
            active_stationarity=bool(result.active_stationarity),optimality=result.optimality,
            condition=result.jacobian_condition,I_nodes=len(levels),u_nodes=len(objective.nodes),
            transform_frozen=bool(result.transform_frozen),return_domain_admissible=parts['return_domain_admissible'])
        history.append(event)
        if checkpoint_dir is not None:
            (checkpoint_dir/f'epsilon_{epsilon}.json').write_text(json.dumps(dict(x=result.x.tolist(),history=history),indent=2,allow_nan=False)+'\n')
    objective.set_levels(objective.cfg['I_nodes'])
    objective.set_nodes(nested_nodes(objective.cfg['u_nodes'],objective.cfg['selection_nodes']))
    objective.epsilon=objective.epsilon_schedule[-1]
    progress.stage(len(plans)-1)
    print(f'{label}: joint polish on common final grid',flush=True)
    result=solve_stage(objective,result.x,objective.cfg.get('common_polish_nfev',objective.cfg['max_nfev_per_epsilon']),
        label='common_grid_joint_polish',log_path=log,gtol=objective.cfg['stationarity_gtol'],
        rescue_evaluations=objective.cfg.get('rescue_evaluations',6),progress=progress)
    history.append(dict(stage='common_grid_joint_polish',epsilon=objective.epsilon,nfev=result.nfev,
        loss=objective.parts(result.x)['full_total'],optimality=result.optimality,success=bool(result.success),
        converged_stationary=bool(result.converged_stationary),I_nodes=len(objective.levels),u_nodes=len(objective.nodes),transform_frozen=False))
    if checkpoint_dir is not None:
        (checkpoint_dir/'common_grid_checkpoint.json').write_text(json.dumps(dict(x=result.x.tolist(),history=history),indent=2,allow_nan=False)+'\n')
    result.history=history;result=finalize(objective,result,checkpoint_dir)
    result.total_optimizer_evaluations=sum(q['nfev'] for q in history)
    result.elapsed_seconds=time.perf_counter()-started
    coordinates=RegionalCoordinates(objective.thresholds,objective.c_ref);y=coordinates.from_raw(result.x);lo,hi=coordinates.bounds()
    result.bound_hits=int(np.sum(np.minimum(y-lo,hi-y)<1e-6))
    return result


def evaluate_incumbent(objective,model,checkpoint_dir=None):
    label=checkpoint_dir.name if checkpoint_dir is not None else 'incumbent'
    with TrialProgress(objective.run_dir/'logs' if checkpoint_dir is not None else None,label) as progress:
        objective.progress=progress
        try:return _evaluate_incumbent(objective,model,checkpoint_dir)
        finally:del objective.progress


def _evaluate_incumbent(objective,model,checkpoint_dir):
    from analyze_candidate import vector
    started=time.perf_counter()
    compatible=(model['transform'].get('C_ref',objective.c_ref)==objective.c_ref
                and len(model['physical_thresholds_C'])==len(objective.thresholds)
                and np.allclose(model['physical_thresholds_C'],objective.thresholds,rtol=0,atol=1e-12))
    if compatible:x=vector(model)
    else:
        from fit_stage import initial_vector
        x=initial_vector(objective,0,model)
    result=OptimizeResult(x=x,nfev=0,status=0,success=False,
        message='supplied incumbent retained and rescored; no optimization',history=[dict(stage='incumbent',nfev=0)])
    if checkpoint_dir is not None:checkpoint_dir.mkdir(parents=True,exist_ok=True)
    result=finalize(objective,result,checkpoint_dir)
    result.incumbent_projected=not compatible
    result.elapsed_seconds=time.perf_counter()-started
    coordinates=RegionalCoordinates(objective.thresholds,objective.c_ref);lo,hi=coordinates.bounds();y=np.asarray(result.regional_x)
    result.bound_hits=int(np.sum(np.minimum(y-lo,hi-y)<1e-6))
    result.total_optimizer_evaluations=0
    return result
