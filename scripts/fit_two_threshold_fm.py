"""Fit two calcium knots inside the dominant FM error interval.

The incumbent is represented exactly at initialization: the original region
is copied into both newly created subregions. Only the two new knot positions
and the PWA coefficients are then optimized. Existing training gates are kept.
"""
import copy, hashlib, json, os, shutil, time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

from pwl_nest.calcium_coordinates import CalciumTransformPWA, decode_coordinates
from pwl_nest.protocols import simulate
from pwl_nest.thresholds import OrderedThresholds
from pwl_nest.equilibrium_constraints import anchor_equilibrium
from pwl_nest.rest_protocols import stable_rest
from fit_anchored_shapes import ShapeObjective, shape_checks
from refine_server_candidates import diagnostics, aggregate, rebuild_training_field
from overnight_coordinate_grid import atomic, source_rest

ROOT=Path('/workspace/PWL-NEST')
SEED=ROOT/'models/phase_shape_candidates/FM_control.json'
SOURCE_CACHE=ROOT/'results/fm_phase_continuation/cache/FM'
REFERENCE_CACHE=Path(os.environ.get('PWL_FM_TWO_THRESHOLD_REFERENCE_CACHE',ROOT/'results/fm_two_thresholds_20261023_rerun1/cache/FM'))
OUT=Path(os.environ.get('PWL_FM_TWO_THRESHOLD_OUT',ROOT/'results/fm_two_thresholds_20261023_rerun1'))
EVAL_BUDGET=int(os.environ.get('PWL_FM_TWO_THRESHOLD_BUDGET','800'))
START_ID=int(os.environ.get('PWL_FM_TWO_THRESHOLD_START','0'))
LOW_C,HIGH_C=.05774933375817452,.1690186499338946

def fresh_protocols():
    specs=[dict(name=f'two_knot_heldout_tau_{tau}',duration=540. if tau==205. else 340.,
        tau=tau,baseline=.45,jumps=[[20.,.7]]) for tau in [13.,37.,76.,128.,205.]]
    rng=np.random.default_rng(20261023)
    times=20.+np.r_[0.,np.cumsum(rng.uniform(14.,25.,8))]
    specs.append(dict(name='two_knot_heldout_irregular',duration=360.,tau=26.,baseline=.45,
        jumps=list(map(list,zip(times.tolist(),rng.uniform(.2,.3,len(times)).tolist())))))
    return specs

def prepare():
    if OUT.exists() and any(OUT.iterdir()): raise RuntimeError(f'Output exists: {OUT}')
    cache=OUT/'cache/FM';cache.mkdir(parents=True)
    spec=json.loads((SOURCE_CACHE/'data.json').read_text())
    spec['previous_tests_not_blind']=spec.get('test',[])
    spec['test']=fresh_protocols()
    for p in spec['fixed']+spec['train']:
        shutil.copyfile(SOURCE_CACHE/f'{p["name"]}.npz',cache/f'{p["name"]}.npz')
    for p in spec['test']:
        ref=REFERENCE_CACHE/f'{p["name"]}.npz'
        if ref.exists():
            shutil.copyfile(ref,cache/ref.name)
        else:
            t,x,_=simulate('FM',spec['rest'],**{k:v for k,v in p.items() if k!='name'},
                           dt=.05,rtol=2e-11,atol=2e-13,max_step=.05)
            np.savez_compressed(cache/f'{p["name"]}.npz',t=t,x=x)
    atomic(cache/'data.json',spec);rebuild_training_field(cache)
    atomic(OUT/'config.json',dict(timing_weight=3.,dynamic_width_gate=True,
        dynamic_width_residual_weight=1.,phase_ceiling=None,regions_before=8,regions_after=10,
        original_interval_C_uM=[LOW_C,HIGH_C],tests=spec['test'],seed_model=str(SEED.relative_to(ROOT)),
        seed_sha256=hashlib.sha256(SEED.read_bytes()).hexdigest(),
        selection='same training eligibility gates as preceding FM phase fit; fresh protocols kept out of fitting'))
    return spec

def main():
    spec=prepare();seed=CalciumTransformPWA.load(SEED)
    old_theta=seed.latent.thresholds.copy();n0=seed.latent.region_count;n=n0+2;nc=2*n+6
    # The interval C=[LOW_C,HIGH_C] is between old knots 3 and 4 (zero based).
    left=float(seed.coordinate.forward(LOW_C));right=float(seed.coordinate.forward(HIGH_C))
    initial_knots=np.array([left+(right-left)/3.,left+2*(right-left)/3.])
    threshold_map=OrderedThresholds(initial_knots,left,right)
    oldz=np.asarray(seed.metadata['parameters'],float)
    # Expand the old 8-region coefficient vector to 10 regions by triplicating
    # old region 4 (index 4); this leaves its field exactly unchanged initially.
    region=4
    newz=np.r_[oldz[:region],np.repeat(oldz[region],3),oldz[region+1:n0],
               oldz[n0:n0+region],np.repeat(oldz[n0+region],3),oldz[n0+region+1:2*n0],oldz[2*n0:]]
    assert len(newz)==nc
    mu_sign=np.asarray(seed.metadata['mu_sign']);k_sign=np.asarray(seed.metadata['k_sign'])
    mu_sign=np.r_[mu_sign[:region],np.repeat(mu_sign[region],3),mu_sign[region+1:]]
    k_sign=np.r_[k_sign[:region],np.repeat(k_sign[region],3),k_sign[region+1:]]
    target_types=seed.metadata['target_types']
    target_types=target_types[:region]+[target_types[region]]*3+target_types[region+1:]
    active=np.array([j for j in range(nc) if j not in [2*n,2*n+5]])
    start=np.r_[newz[active],np.zeros(2)]
    incumbent_start=start.copy()
    full_lo=np.r_[np.full(n,-12.),np.full(n,-40.),-.5,-40.,-28.,-10.,-10.,-10.]
    full_hi=np.r_[np.full(n,5.),np.full(n,10.),.5,8.,8.,2.,10.,10.]
    for j,label in enumerate(target_types):
        if label.endswith('node'):full_hi[n+j]=12.
    lo=np.r_[full_lo[active],[-np.log(4)]*2];hi=np.r_[full_hi[active],[np.log(4)]*2]
    # Break the exact symmetry between duplicated child regions before moving
    # their boundaries. Otherwise threshold locations have almost no gradient.
    # Several deterministic starts cover both knot placement and coefficient
    # asymmetry; start 0 remains the exact incumbent warm start for comparison.
    if START_ID:
        rng=np.random.default_rng(91000+START_ID)
        fractions=[(.18,.62),(.28,.76),(.38,.82),(.12,.84),(.24,.54),(.46,.70)]
        f1,f2=fractions[(START_ID-1)%len(fractions)]
        physical_knots=np.array([left+(right-left)*f1,left+(right-left)*f2])
        start[len(active):]=threshold_map.encode(physical_knots)
        coeff_by_full={int(full):j for j,full in enumerate(active)}
        # Perturb the four local coordinates of each child segment in opposite
        # directions; global parameters and surrounding regions stay untouched.
        for offset in (region, n+region):
            width=3
            for coordinate in range(2):
                idx=[coeff_by_full[offset+k] for k in range(width) if offset+k in coeff_by_full]
                if idx:
                    delta=rng.normal(0.,.015,len(idx))
                    delta-=delta.mean()
                    start[idx]+=delta
        start=np.minimum(np.maximum(start,lo+1e-9),hi-1e-9)
    data=spec
    base_meta=copy.deepcopy(seed.metadata)
    for key in ['training_checks','fine_training_shape_passes','fine_training_phase_check','heldout_shape_passes',
                'heldout_protocol_count','verified_train_loss','recommended_by_training_only']:
        base_meta.pop(key,None)
    def make(p):
        z=np.zeros(nc);z[active]=p[:len(active)]
        extra=threshold_map.decode(p[len(active):])
        theta=np.r_[old_theta[:4],extra,old_theta[4:]]
        meta=copy.deepcopy(base_meta)
        meta.update(parameters=z.tolist(),mu_sign=mu_sign.tolist(),k_sign=k_sign.tolist(),target_types=target_types,
            regions=n,free_parameters=nc,accepted=False,extra_C_thresholds_uM=seed.coordinate.inverse(extra).tolist(),
            threshold_fit_interval_C_uM=[LOW_C,HIGH_C],threshold_shifts=p[len(active):].tolist(),
            threshold_parameter_count=2,coefficient_parameter_count=len(active),
            thresholds_policy='incumbent knots fixed; two ordered optimized knots within the identified high-error C interval')
        model=decode_coordinates('FM',theta,z,mu_sign,k_sign,meta,seed.coordinate,seed.h_reference,seed.h_kind)
        model.mix_linear=seed.mix_linear;model.mix_quadratic=seed.mix_quadratic;model.mix_center=seed.mix_center
        model.__post_init__()
        return anchor_equilibrium(model,spec['rest'])
    # Guard the exact warm start before any optimization.
    probe=np.column_stack([np.linspace(.02,.9,101),np.linspace(.12,.98,101),np.linspace(.2,1.15,101)])
    warm=make(incumbent_start);preservation=float(np.max(np.abs(warm.field(warm.to_latent(probe))-seed.field(seed.to_latent(probe)))))
    if preservation>2e-8:raise RuntimeError(f'Warm start changed incumbent field: {preservation:g}')
    objective=ShapeObjective(OUT,'FM',seed)
    r0=objective(seed,precise=True);base_loss=float(r0@r0)
    # The inherited spectral-anchor residual has one target per knot. Preserve
    # the original seven targets; use the exact warm-start branch roots for the
    # two inserted knots so their initial anchor residuals are zero.
    old_anchors=np.asarray(objective.static.anchors,float)
    warm_roots=warm.equilibrium_inputs_at_thresholds()
    objective.static.anchors=np.r_[old_anchors[:4],warm_roots[4:6],old_anchors[4:]]
    rstart=objective(warm,precise=True);warm_loss=float(rstart@rstart)
    print(json.dumps(dict(stage='preflight',base_loss=base_loss,warm_loss=warm_loss,field_preservation=preservation)),flush=True)
    # Bound the total residual evaluations, including finite-difference Jacobians.
    budget=EVAL_BUDGET;count=0;best=start.copy();bestloss=np.inf;started=time.monotonic();history=[]
    class Limit(Exception):pass
    def residual(p):
        nonlocal count,best,bestloss
        if count>=budget:raise Limit()
        count+=1;model=make(p);r=objective(model)
        reg=np.r_[.005*(p[:len(active)]-start[:len(active)]),.03*p[len(active):]]
        full=np.r_[r,reg];loss=float(full@full)
        if loss<bestloss:
            bestloss=loss;best=p.copy()
            if count%25==0:
                atomic(OUT/'checkpoint.json',dict(evaluations=count,loss=loss,seconds=time.monotonic()-started,
                    thresholds_C=seed.coordinate.inverse(np.r_[old_theta[:4],threshold_map.decode(p[len(active):]),old_theta[4:]]).tolist()))
                print(json.dumps(dict(stage='fit',evaluations=count,loss=loss,seconds=time.monotonic()-started)),flush=True)
        return full
    initial_residual=residual(start)
    status=''
    def jacobian(p):
        base=residual(p);columns=[]
        for j in range(len(p)):
            step=1e-4*max(1.,abs(p[j])) if j<len(active) else 1e-3
            if p[j]+step>hi[j]:step=-step
            q=p.copy();q[j]+=step
            columns.append((residual(q)-base)/step)
        return np.column_stack(columns)
    try:
        fit=least_squares(residual,start,jac=jacobian,bounds=(lo,hi),max_nfev=100,x_scale='jac',
                          ftol=1e-8,xtol=1e-8,gtol=1e-8,verbose=0)
        status=str(fit.message)
    except Limit:status='least_squares evaluation budget exhausted'
    try:
        remaining=budget-count
        if remaining>1:
            from scipy.optimize import minimize
            result=minimize(lambda p: float(np.dot(r:=residual(p),r)),best,method='Powell',
                bounds=list(zip(lo,hi)),options=dict(maxfev=remaining,xtol=1e-5,ftol=1e-7))
            status=(status+'; Powell: '+str(result.message)).strip('; ')
    except Limit:status=(status+'; Powell evaluation budget exhausted').strip('; ')
    candidate=make(best);rfit=objective(candidate,precise=True);fit_loss=float(rfit@rfit)
    checks=shape_checks(objective,candidate)
    candidate.metadata.update(accepted=False,training_loss=fit_loss,training_eligible=checks['eligible'],
        fit_status=status,fit_evaluations=count,incumbent_field_preservation_max_error=preservation)
    OUT.mkdir(parents=True,exist_ok=True)
    outmodel=OUT/'FM_two_threshold_candidate.json';candidate.save(outmodel)
    # Fresh test diagnostics use the same protocols for incumbent and candidate.
    rows=[]
    for name,model in [('incumbent',seed),('two_threshold',candidate)]:
        rec=diagnostics(model,objective,spec['test'],objective.refs,OUT/'diagnostics'/name)
        for item,p in zip(rec,spec['test']):
            if item['completed']:
                with np.load(OUT/'diagnostics'/name/f'{p["name"]}.npz') as dat:
                    from pwl_nest.peak_shape import peak_shape_check
                    item['shape']=peak_shape_check(dat['t'],dat['candidate'],dat['t'],dat['lr'])
        rows.append(dict(name=name,aggregate=aggregate(rec),protocols=rec,
            shape_passes=sum(x.get('shape',{}).get('passes',False) for x in rec)))
    report=dict(seed=str(SEED.relative_to(ROOT)),seed_sha256=hashlib.sha256(SEED.read_bytes()).hexdigest(),
        fitted_model=str(outmodel.resolve()),thresholds_C=seed.coordinate.inverse(candidate.latent.thresholds).tolist(),
        thresholds_added_C=seed.coordinate.inverse(threshold_map.decode(best[len(active):])).tolist(),
        field_preservation_max_error=preservation,training_loss_incumbent=base_loss,training_loss_warm_start=warm_loss,
        training_loss_candidate=fit_loss,training_checks=checks,fit_status=status,fit_evaluations=count,
        heldout_comparison=rows,accepted=False,
        method='Two optimized ordered C knots inside 0.0577493–0.169019 uM; original other knots fixed; all active regional coefficients refit; same phase/shape/rest gates; fresh protocols excluded from fitting.')
    atomic(OUT/'summary.json',report)
    print(json.dumps(dict(stage='complete',thresholds_C=report['thresholds_added_C'],training_loss=fit_loss,
        eligible=checks['eligible'],heldout=[dict(name=r['name'],aggregate=r['aggregate'],shape_passes=r['shape_passes']) for r in rows])),flush=True)

if __name__=='__main__':main()
