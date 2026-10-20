"""Equal-budget fixed/joint continuation from preserved coordinate incumbents."""
import argparse,copy,json,time
import numpy as np
from scipy.optimize import least_squares,minimize
from pwl_nest.models import ROOT
from pwl_nest.strict_pwa import C_MAX
from pwl_nest.thresholds import OrderedThresholds
from pwl_nest.calcium_coordinates import CalciumTransformPWA,decode_coordinates,NativeCoordinateDynamic
from pwl_nest.calcium_strips import NativeStatistics
from fit_calcium_coordinates import training_data,dump
from fit_strict import stats
from validate_calcium_coordinates import validate

MODELS=ROOT/'models/movable_thresholds';OUT=ROOT/'results/movable_thresholds'
SEEDS={'AM':'models/coordinate_families/selected_AM.json','FM':'models/calcium_powers/sqrt/FM.json'}


class CommonResidual:
    """Previous physical residual, including unchanged soft spectral-I anchors."""
    def __init__(self,data,seed):
        self.training,self.refs,*_=data;self.case=seed.case
        self.fixed=[p for p in self.training if p['name'].startswith('fixed')]
        self.decays=[p for p in self.training if p['name'].startswith('decay')]
        self.targets={p['name']:stats(*self.refs[p['name']][:2]) for p in self.fixed}
        self.htargets={p['name']:[self.refs[p['name']][1][:,1].min(),self.refs[p['name']][1][:,1].max()] for p in self.fixed}
        self.sampled={}
        for p in self.decays:
            t,x,_=self.refs[p['name']];grid=np.linspace(0,p['duration'],361)
            self.sampled[p['name']]=np.column_stack([np.interp(grid,t,x[:,d]) for d in [0,1]])
        self.anchors=np.array([e['I'] for e in seed.metadata['events']])
        self.static=NativeStatistics();self.dynamic=NativeCoordinateDynamic()

    def __call__(self,model):
        co,gate=model.coordinate,model.gate;theta=model.latent.thresholds
        upper_guard=float(co.forward(20.));rr=[]
        for p in self.fixed:
            effective=model.at_input(p['baseline'])
            obs=self.static.run(effective.latent,p['baseline'],model.to_latent(p['initial']),p['duration'],diagnostics=True)
            if obs is None:rr.extend([100.]*7);continue
            ob,diag=obs;mn,mx,period,ratio,minall,stopped=ob
            if minall<=co.lower or diag['max_C']>=upper_guard:
                gap=max(0.,co.lower-minall,diag['max_C']-upper_guard);rr.extend([50.+1000*gap]*7);continue
            if diag['min_h']<=gate.lower or diag['max_h']>=gate.upper:
                gap=max(0.,gate.lower-diag['min_h'],diag['max_h']-gate.upper);rr.extend([50.+1000*gap]*7);continue
            try:
                cmin,cmax,cminall,cmaxall=map(float,co.inverse(np.array([mn,mx,minall,diag['max_C']])))
                hmin,hmax=map(float,gate.inverse(np.array([diag['min_h'],diag['max_h']])))
            except (ValueError,FloatingPointError):rr.extend([100.]*7);continue
            target=self.targets[p['name']]
            rr.extend([(cmin-target['Cmin'])/.02,(cmax-target['Cmax'])/.05,
                (period-target['period'])/(1 if self.case=='AM' else 3) if target['period'] is not None and np.isfinite(period)
                else (15. if target['period'] is not None else (cmax-cmin)/.02),
                min(0,cminall)/.005,max(0,cmaxall-C_MAX)/.02,
                (hmin-self.htargets[p['name']][0])/.1,(hmax-self.htargets[p['name']][1])/.1])
        for p in self.decays:
            y=self.dynamic.run(model,p['initial'],p['duration'],p['tau'],p['baseline'])
            if y is None:rr.extend(np.full(724,100.));continue
            try:x=np.column_stack([co.inverse(y[:,0]),gate.inverse(y[:,1])])
            except (ValueError,FloatingPointError):rr.extend(np.full(724,100.));continue
            rr.extend(((x-self.sampled[p['name']])/np.array([.03,.05])/np.sqrt(361)).ravel())
            rr.extend([min(0,x[:,0].min())/.005,max(0,x[:,0].max()-C_MAX)/.02])
        rr.extend((model.equilibrium_inputs_at_thresholds(self.anchors)-self.anchors)/.03)
        return np.array(rr)


class EvaluationLimit(Exception):pass


def fit_arm(seed,data,objective,variant,per_start=1500,powell_budget=2000):
    moving=variant=='moving_joint';case=seed.case;n=seed.latent.region_count;ncoef=2*n+6;m=n-1
    ref=np.array(seed.metadata['physical_thresholds_C']);mapping=OrderedThresholds(ref,data[4],data[5])
    coefficients=np.array(seed.metadata['parameters']);ms=seed.metadata['mu_sign'];ks=seed.metadata['k_sign']
    initial=np.r_[coefficients,np.zeros(m)] if moving else coefficients.copy()
    lower=np.r_[np.full(n,-9.),np.full(n,-12.),-.5,-12.,-8.,-4.,-8.,-8.]
    upper=np.r_[np.full(n,5.),np.full(n,5.),.5,2.,3.,1.5,8.,8.]
    for j,label in enumerate(seed.metadata['target_types']):
        if label.endswith('node'):upper[n+j]=12.
    if moving:lower=np.r_[lower,np.full(m,-np.log(2))];upper=np.r_[upper,np.full(m,np.log(2))]
    if np.any(initial<lower) or np.any(initial>upper):raise ValueError('Incumbent outside coefficient bounds')
    meta=copy.deepcopy(seed.metadata)
    meta.update(experiment_variant=variant,threshold_reference_C=ref.tolist(),
        thresholds_policy='ordered interval widths fitted around source spectral seeds' if moving else 'fixed source spectral C thresholds',
        threshold_domain_C=[mapping.lower,mapping.upper],coefficient_parameter_count=ncoef,
        threshold_parameter_count=m if moving else 0,parameter_count=len(initial),free_parameters=len(initial),
        coefficient_prior_strength=.005,threshold_gap_prior_strength=.05 if moving else 0.,
        accepted=False,seed_model=SEEDS[case])
    def make(z):
        physical=mapping.decode(z[ncoef:]) if moving else ref.copy()
        metadata={**meta,'physical_thresholds_C':physical.tolist()}
        return decode_coordinates(case,seed.coordinate.forward(physical),z[:ncoef],ms,ks,metadata,seed.coordinate,seed.h_reference,seed.h_kind)
    count=0;limit=1;best=np.inf;bestz=initial.copy();history=[];started=time.monotonic()
    def residual(z):
        nonlocal count,best,bestz
        if count>=limit:raise EvaluationLimit()
        count+=1;r=objective(make(z));data_loss=float(r@r)
        reg=.005*(z[:ncoef]-initial[:ncoef])
        if moving:reg=np.r_[reg,.05*z[ncoef:]]
        full=np.r_[r,reg];loss=float(full@full)
        if loss<best:
            best,bestz=loss,z.copy()
            if not history or count-history[-1]['evaluation']>=100:
                history.append(dict(evaluation=count,data_loss=data_loss,total_loss=loss,seconds=time.monotonic()-started))
                print(f'{case} {variant}: eval={count}, data_loss={data_loss:.7g}',flush=True)
        return full
    residual(initial)
    rng=np.random.default_rng(20261020)
    perturbed=initial.copy();perturbed[:ncoef]+=rng.normal(0,.025,ncoef)
    if moving:perturbed[ncoef:]=rng.normal(0,.06,m)
    perturbed=np.clip(perturbed,lower+1e-9,upper-1e-9)
    starts=[initial,perturbed];runs=[]
    # Absolute steps avoid near-zero threshold shifts using a vanishing relative FD step.
    def jacobian(z):
        base=residual(z);columns=[]
        for j in range(len(z)):
            step=(1e-4*max(1.,abs(z[j]))) if j<ncoef else 1e-3
            if z[j]+step>upper[j]:step=-step
            value=z.copy();value[j]+=step
            columns.append((residual(value)-base)/step)
        return np.column_stack(columns)
    for label,z in zip(['incumbent','perturbed'],starts):
        limit=count+per_start
        try:
            fit=least_squares(residual,z,jac=jacobian,bounds=(lower,upper),x_scale='jac',
                              max_nfev=120,ftol=1e-8,xtol=1e-8,gtol=1e-8)
            status=dict(status=int(fit.status),nfev=int(fit.nfev),message=fit.message)
        except EvaluationLimit:status=dict(status='evaluation_limit',message=f'{per_start} residual evaluations exhausted')
        runs.append(dict(method='least_squares',start=label,**status))
    limit=count+powell_budget
    try:
        fit=minimize(lambda z:float(np.dot(r:=residual(z),r)),bestz,method='Powell',bounds=list(zip(lower,upper)),
                     options=dict(maxfev=powell_budget,xtol=1e-5,ftol=1e-7))
        status=dict(status=int(fit.status),nfev=int(fit.nfev),message=str(fit.message))
    except EvaluationLimit:status=dict(status='evaluation_limit',message=f'{powell_budget} residual evaluations exhausted')
    runs.append(dict(method='Powell',**status))
    model=make(bestz);r=objective(model);score=float(r@r)
    model.metadata.update(parameters=bestz[:ncoef].tolist(),threshold_shifts=bestz[ncoef:].tolist() if moving else [],
        training_data_loss=score,training_total_loss=best,
        selected_by='common physical residual plus unchanged spectral-I anchors; excludes regularization')
    directory=MODELS/variant;directory.mkdir(parents=True,exist_ok=True);model.save(directory/f'{case}.json')
    destination=OUT/variant;destination.mkdir(parents=True,exist_ok=True)
    dump(destination/f'{case}_fit.json',dict(training_data_loss=score,training_total_loss=best,
        evaluations=count,history=history,optimizer_runs=runs,bounds=[lower.tolist(),upper.tolist()],
        thresholds_initial_C=ref.tolist(),thresholds_final_C=model.metadata['physical_thresholds_C'],
        threshold_domain_C=[mapping.lower,mapping.upper],parameters=bestz.tolist(),
        max_residual_evaluations_per_LS_start=per_start,Powell_max_evaluations=powell_budget))
    return model,score


def run(case):
    data=training_data(case);seed=CalciumTransformPWA.load(ROOT/SEEDS[case]);objective=CommonResidual(data,seed)
    r=objective(seed);score=float(r@r)
    print(f'{case} incumbent: recomputed={score:.8g}, previous={seed.metadata["training_data_loss"]:.8g}',flush=True)
    seed.metadata.update(experiment_variant='baseline',seed_model=SEEDS[case],
                         reported_previous_training_loss=seed.metadata['training_data_loss'],training_data_loss=score,
                         threshold_parameter_count=0,coefficient_parameter_count=len(seed.metadata['parameters']))
    (MODELS/'baseline').mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    seed.save(MODELS/'baseline'/f'{case}.json')
    pairs=[(seed,score)]+[fit_arm(seed,data,objective,variant) for variant in ['fixed_refit','moving_joint']]
    selected,score=min(pairs,key=lambda pair:pair[1]);selected.save(MODELS/f'selected_{case}.json')
    dump(OUT/f'{case}_selection.json',dict(selected=selected.variant,variants=[m.variant for m,s in pairs],
        training_scores={m.variant:s for m,s in pairs},source_seed=SEEDS[case],
        selection='lowest common training data-loss; diagnostics excluded',
        control='same incumbent and two coefficient starts; fixed/joint same maximum residual budgets',
        threshold_bounds='m independent log interval ratios bounded by +/-ln(2); endpoints source C equilibrium at I=0,1.3'))
    validate(case,[m for m,s in pairs],data[0],data[1],output_root=OUT)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=['AM','FM','both'],default='both');args=parser.parse_args()
    for case in (['AM','FM'] if args.case=='both' else [args.case]):run(case)
