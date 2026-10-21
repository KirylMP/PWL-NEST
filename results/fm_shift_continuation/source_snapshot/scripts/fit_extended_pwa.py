"""Matched pilots: cube gate, input hinges and ordered oblique C planes."""
import argparse,copy,time
import numpy as np
from scipy.optimize import least_squares,minimize
from pwl_nest.models import ROOT
from pwl_nest.strict_pwa import C_MAX
from pwl_nest.calcium_coordinates import CalciumCoordinate,CalciumTransformPWA,decode_coordinates
from pwl_nest.thresholds import OrderedThresholds,oblique_thresholds
from fit_calcium_coordinates import training_data,dump
from fit_movable_thresholds import CommonResidual,EvaluationLimit
from validate_calcium_coordinates import validate

OUT=ROOT/'results/extended_pwa';MODELS=ROOT/'models/extended_pwa'
VARIANTS=['control','sqrt_cube','log_cube','hill_cube','input_one','input_two','tilted']
INPUT_KNOTS={'AM':[.35,.64],'FM':[.53,1.1]}


def fit_variant(case,variant,seed,data,objective):
    n=seed.latent.region_count;ncoef=2*n+6;m=n-1
    physical=np.array(seed.metadata['physical_thresholds_C'])
    hkind='cube' if variant.endswith('_cube') else seed.h_kind
    ckind={'log_cube':'log','hill_cube':'hill'}.get(variant,'sqrt')
    if ckind==seed.coordinate.kind:co=seed.coordinate
    else:
        half=(.1 if case=='AM' else .05) if ckind=='hill' else None
        raw=CalciumCoordinate(ckind,seed.coordinate.reference,half_saturation=half)
        co=CalciumCoordinate(ckind,seed.coordinate.reference,float(raw.forward(C_MAX)/C_MAX),half)
    theta=co.forward(physical)
    mapping=OrderedThresholds(theta,float(co.forward(data[4])),float(co.forward(data[5])))
    knots=np.array(INPUT_KNOTS[case][-1:] if variant=='input_one' else INPUT_KNOTS[case] if variant=='input_two' else [])
    extra=2*len(knots) if len(knots) else m if variant=='tilted' else 0
    initial=np.r_[seed.metadata['parameters'],np.zeros(extra)]
    # Exact tangent affine scaling between centered calcium coordinates at Cref.
    factor=seed.coordinate.derivative(co.reference)/co.derivative(co.reference)
    initial[2*n+1]-=np.log(factor);initial[2*n+2]-=np.log(factor);initial[2*n+4:2*n+6]/=factor
    lower=np.r_[np.full(n,-9.),np.full(n,-12.),-.5,-12.,-8.,-4.,-8.,-8.]
    upper=np.r_[np.full(n,5.),np.full(n,5.),.5,2.,3.,1.5,8.,8.]
    for j,label in enumerate(seed.metadata['target_types']):
        if label.endswith('node'):upper[n+j]=12.
    if len(knots):
        lower=np.r_[lower,np.full(len(knots),-2.),np.full(len(knots),-np.log(4))]
        upper=np.r_[upper,np.full(len(knots),2.),np.full(len(knots),np.log(4))]
    elif variant=='tilted':
        lower=np.r_[lower,np.full(m,-np.log(2))];upper=np.r_[upper,np.full(m,np.log(2))]
    initial=np.clip(initial,lower+1e-9,upper-1e-9)
    metadata=copy.deepcopy(seed.metadata)
    metadata.update(class_name='PWA_in_transformed_calcium_and_gate',experiment_variant=variant,calcium_transform=ckind,h_transform=hkind,
        coordinate_scale=co.scale,physical_thresholds_C=physical.tolist(),
        parameter_count=len(initial),free_parameters=len(initial),active_fit_parameters=len(initial),
        coefficient_parameter_count=ncoef,new_geometry_parameters=extra,
        held_incumbent_C_thresholds=physical.tolist(),input_thresholds=knots.tolist(),
        input_knot_locations_fitted=False,reference_input=.65,accepted=False,
        thresholds_policy='ordered affine planes with antisymmetric endpoint gaps' if variant=='tilted' else 'incumbent C thresholds frozen',
        pilot='standalone branches, not all Cartesian feature combinations',seed_model=f'models/movable_thresholds/selected_{case}.json')
    def make(z):
        th,slopes=oblique_thresholds(mapping,z[ncoef:]) if variant=='tilted' else (theta,np.zeros(m))
        meta={**metadata,'physical_thresholds_C':co.inverse(th).tolist()}
        base=decode_coordinates(case,th,z[:ncoef],seed.metadata['mu_sign'],seed.metadata['k_sign'],meta,co,seed.h_reference,hkind)
        weights=np.zeros((2,len(knots)))
        if len(knots):
            B,D=base.latent.matrix[:,2];qb=base.latent.spectral_coefficients()[3]
            slopes_input=qb*np.r_[1.,np.exp(z[ncoef+len(knots):])]
            p=z[ncoef:ncoef+len(knots)];weights=np.array([p,(np.diff(slopes_input)+D*p)/B])
        return CalciumTransformPWA(base.latent,co,seed.h_reference,hkind,knots,weights,slopes)
    points=data[2];template=make(initial);transformed=template.to_latent(points);truth=template.source_field(transformed)
    inverse_derivatives=np.column_stack([1/co.derivative(points[:,0]),1/template.gate.derivative(points[:,1])])
    def field_residual(z):
        error=(make(z).field(transformed)-truth)*inverse_derivatives
        return np.r_[(error/np.array([.2,.05])).ravel()/np.sqrt(len(points)),.01*(z-initial)]
    print(f'{case} {variant}: field seed, active parameters={len(initial)}',flush=True)
    field=least_squares(field_residual,initial,bounds=(lower,upper),max_nfev=80,ftol=1e-8,xtol=1e-8,gtol=1e-8)
    rng=np.random.default_rng(20261020)
    perturb=initial.copy();perturb[:ncoef]+=rng.normal(0,.025,ncoef)
    if extra:perturb[ncoef:]+=rng.normal(0,.04,extra)
    perturb=np.clip(perturb,lower+1e-9,upper-1e-9)
    directory=MODELS/variant;directory.mkdir(parents=True,exist_ok=True)
    make(initial).save(directory/f'{case}_initial.json');make(field.x).save(directory/f'{case}_field_seed.json')
    count=0;limit=10;best=np.inf;bestz=initial.copy();history=[];started=time.monotonic()
    def residual(z):
        nonlocal count,best,bestz
        if count>=limit:raise EvaluationLimit()
        count+=1;data_r=objective(make(z));score=float(data_r@data_r)
        reg=np.r_[.005*(z[:ncoef]-initial[:ncoef]),.05*(z[ncoef:]-initial[ncoef:])]
        full=np.r_[data_r,reg];loss=float(full@full)
        if loss<best:
            best,bestz=loss,z.copy()
            if not history or count-history[-1]['evaluation']>=150:
                history.append(dict(evaluation=count,data_loss=score,total_loss=loss,seconds=time.monotonic()-started))
                print(f'{case} {variant}: eval={count}, data_loss={score:.7g}',flush=True)
        return full
    candidates=[('incumbent_tangent',initial),('perturbed',perturb),('field',field.x)]
    scores=[]
    for label,z in candidates:
        r=residual(z);scores.append(dict(start=label,total_loss=float(r@r)))
    starts=sorted(zip(candidates,scores),key=lambda pair:pair[1]['total_loss'])[:2];runs=[]
    def jac(z):
        base=residual(z);columns=[]
        for j in range(len(z)):
            step=1e-4*max(1.,abs(z[j])) if j<ncoef else 1e-3
            if z[j]+step>upper[j]:step=-step
            other=z.copy();other[j]+=step;columns.append((residual(other)-base)/step)
        return np.column_stack(columns)
    for (label,z),_ in starts:
        limit=count+1500
        try:
            fit=least_squares(residual,z,jac=jac,bounds=(lower,upper),x_scale='jac',max_nfev=120,ftol=1e-8,xtol=1e-8,gtol=1e-8)
            status=dict(status=int(fit.status),nfev=int(fit.nfev),message=fit.message)
        except EvaluationLimit:status=dict(status='evaluation_limit',message='1500 residual evaluations exhausted')
        runs.append(dict(method='least_squares',start=label,**status))
    limit=count+2000
    try:
        fit=minimize(lambda z:float(np.dot(r:=residual(z),r)),bestz,method='Powell',bounds=list(zip(lower,upper)),options=dict(maxfev=2000,xtol=1e-5,ftol=1e-7))
        status=dict(status=int(fit.status),nfev=int(fit.nfev),message=str(fit.message))
    except EvaluationLimit:status=dict(status='evaluation_limit',message='2000 residual evaluations exhausted')
    runs.append(dict(method='Powell',**status))
    model=make(bestz);r=objective(model);score=float(r@r)
    model.metadata.update(parameters=bestz[:ncoef].tolist(),extra_parameters=bestz[ncoef:].tolist(),
        training_data_loss=score,training_total_loss=best,accepted=False,
        selected_by='common physical residual and source equilibrium-I anchors, excluding regularization')
    model.save(directory/f'{case}.json');(OUT/variant).mkdir(parents=True,exist_ok=True)
    dump(OUT/variant/f'{case}_fit.json',dict(training_data_loss=score,training_total_loss=best,
        initial_scores=scores,optimizer_runs=runs,evaluations=count,history=history,
        field_nfev=int(field.nfev),bounds=[lower.tolist(),upper.tolist()],parameters=bestz.tolist(),
        max_evaluations=dict(LS_per_start=1500,Powell=2000,field_max_nfev=80),
        active_parameters=len(initial),input_knots=knots.tolist()))
    return model,score


def run(case):
    data=training_data(case);seed=CalciumTransformPWA.load(ROOT/f'models/movable_thresholds/selected_{case}.json')
    objective=CommonResidual(data,seed);r=objective(seed);score=float(r@r)
    seed.metadata.update(experiment_variant='baseline',reported_previous_training_loss=seed.metadata['training_data_loss'],training_data_loss=score)
    (MODELS/'baseline').mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    seed.save(MODELS/'baseline'/f'{case}.json')
    print(f'{case} baseline data_loss={score:.8g}',flush=True)
    pairs=[(seed,score)]+[fit_variant(case,variant,seed,data,objective) for variant in VARIANTS]
    selected,score=min(pairs,key=lambda pair:pair[1]);selected.save(MODELS/f'selected_{case}.json')
    dump(OUT/f'{case}_selection.json',dict(selected=selected.variant,variants=[m.variant for m,s in pairs],
        training_scores={m.variant:s for m,s in pairs},selection='training only; standalone branches, not all feature combinations',
        controls='preserved movable-threshold incumbent plus equally refitted control; C geometry frozen except tilted branch',
        input_knots=INPUT_KNOTS[case]))
    validate(case,[m for m,s in pairs],data[0],data[1],output_root=OUT)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=['AM','FM','both'],default='both');args=parser.parse_args()
    for case in (['AM','FM'] if args.case=='both' else [args.case]):run(case)
