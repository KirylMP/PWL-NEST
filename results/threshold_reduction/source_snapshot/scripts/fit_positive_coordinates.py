"""Continuation with sixth powers, movable C knots and explicit failure cases.

Historical diagnostics promoted to calibration are labelled as such. New test
protocols are never used for fitting or candidate selection. No C clipping.
"""
import argparse, copy, json, time
import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.signal import find_peaks
from pwl_nest.models import ROOT
from pwl_nest.strict_pwa import C_MAX
from pwl_nest.calcium_coordinates import (CalciumCoordinate, CalciumTransformPWA,
    NativeCoordinateDynamic, decode_coordinates, simulate_coordinates)
from pwl_nest.calcium_strips import NativeStatistics
from pwl_nest.thresholds import OrderedThresholds
from pwl_nest.protocols import simulate
from fit_calcium_coordinates import training_data, dump
from fit_strict import protocols, stats
from validate_logit import fresh_protocols as initial_h_protocols
from validate_calcium_coordinates import new_initial_C

OUT=ROOT/'results/positive_coordinates'
MODELS=ROOT/'models/positive_coordinates'
VARIANTS=['control','sixth_C','sixth_root_C','sixth_h','bounded_C','sqrt_tails']


def calibration(case,data):
    old=list(data[0]);fixed=[.637,.638,.64] if case=='AM' else [.53,1.12,1.13]
    promoted=[dict(name=f'fixed_{i}',initial=[.6 if case=='AM' else 1.3,.8,i],
        duration=1600.,tau=20.,baseline=i) for i in fixed]
    promoted.append(initial_h_protocols(case)[1])
    return old+promoted, [p['name'] for p in promoted]


def fresh_tests(case):
    start,base=(.65,.3) if case=='AM' else (1.15,.45)
    fixed=[.405,.515,.625] if case=='AM' else [.575,.825,1.08]
    result=[dict(name=f'fresh_fixed_{i}',initial=[.6 if case=='AM' else 1.3,.8,i],
        duration=1600.,tau=20.,baseline=i) for i in fixed]
    result += [dict(name=f'fresh_tau_{tau}',initial=[.055,.86,start],duration=180.,
        tau=tau,baseline=base) for tau in [7.,65.]]
    rng=np.random.default_rng(20261021)
    events=list(zip(np.cumsum(rng.uniform(11,22,6)).tolist(),
        rng.uniform(.10,.19,6).tolist() if case=='AM' else rng.uniform(.2,.34,6).tolist()))
    result.append(dict(name='fresh_irregular',initial=[.055,.86,base],duration=220.,
        tau=17.,baseline=base,jumps=events))
    return result


def is_fixed(p):return 'fixed' in p['name']


class CalibrationResidual:
    def __init__(self,case,training,refs,seed,peak_weight=25.,explicit_status=True,spectral_anchors=True):
        self.use_spectral_anchors=bool(spectral_anchors)
        self.case=case;self.training=training;self.refs=refs
        self.peak_weight=peak_weight;self.explicit_status=explicit_status
        self.static=NativeStatistics();self.dynamic=NativeCoordinateDynamic()
        self.anchors=(np.array([e['I'] for e in seed.metadata['events']])
                      if self.use_spectral_anchors else np.empty(0))
        self.targets={p['name']:stats(*refs[p['name']][:2]) for p in training if is_fixed(p)}
        self.samples={}
        for p in training:
            if not is_fixed(p):
                t,x,_=refs[p['name']];grid=np.linspace(0,p['duration'],721)
                self.samples[p['name']]=np.column_stack([np.interp(grid,t,x[:,d]) for d in [0,1]])

    def __call__(self,model,replay=None,parts=False):
        if model.has_state_mix and replay is None:
            raise ValueError('Mixed coordinates require statistics from physical trajectories')
        co,gate=model.coordinate,model.gate;groups={};rr=[]
        for p in self.training:
            name=p['name'];r=[]
            try:
                if is_fixed(p):
                    if replay is None:
                        obs=self.static.run(model.at_input(p['baseline']).latent,p['baseline'],
                            model.to_latent(p['initial']),p['duration'],diagnostics=True)
                        if obs is None:raise ValueError('Static solver failed')
                        ob,diag=obs
                        cmin,cmax,cminall,cmaxall=co.inverse(np.array([ob[0],ob[1],ob[4],diag['max_C']]))
                        hmin,hmax=gate.inverse(np.array([diag['min_h'],diag['max_h']]))
                        period=float(ob[2]) if np.isfinite(ob[2]) else None
                    else:
                        t,x,_=replay[name];st=stats(t,x)
                        cmin,cmax=st['Cmin'],st['Cmax'];period=st['period']
                        cminall,cmaxall=x[:,0].min(),x[:,0].max();hmin,hmax=x[:,1].min(),x[:,1].max()
                    target=self.targets[name];truth=self.refs[name][1]
                    r=[(cmin-target['Cmin'])/.02,(cmax-target['Cmax'])/.05]
                    r.append((period-target['period'])/(1 if self.case=='AM' else 3)
                        if target['period'] is not None and period is not None
                        else (25. if target['period'] is not None else (cmax-cmin)/.02))
                    # Wrong regime cannot be compensated by a small waveform improvement.
                    status='cycling' if period is not None else 'rest' if cmax-cmin<.001 else 'unresolved'
                    mismatch=(target['observed_status']!='unresolved' and status!=target['observed_status']) if self.explicit_status else ((period is None)!=(target['period'] is None))
                    r.append(25.*float(mismatch))
                    r.extend([max(0,cmaxall-C_MAX)/.005,min(0,cminall)/.002,
                        (hmin-truth[:,1].min())/.1,(hmax-truth[:,1].max())/.1])
                else:
                    grid=np.linspace(0,p['duration'],721)
                    if replay is None:
                        y=self.dynamic.run(model,p['initial'],p['duration'],p['tau'],p['baseline'],outputs=721,maxdt=.005)
                        if y is None:raise ValueError('RK solver failed')
                        x=np.column_stack([co.inverse(y[:,0]),gate.inverse(y[:,1])])
                    else:
                        t,path,_=replay[name]
                        x=np.column_stack([np.interp(grid,t,path[:,d]) for d in [0,1]])
                    target=self.samples[name]
                    # Preserve input-locked timing, at half the previous waveform weight.
                    r=list((.5*(x-target)/np.array([.03,.05])/np.sqrt(len(x))).ravel())
                    a=find_peaks(target[:,0],prominence=.02)[0];b=find_peaks(x[:,0],prominence=.02)[0]
                    r.extend([self.peak_weight*(len(b)-len(a)),max(0,x[:,0].max()-C_MAX)/.002])
                if not np.isfinite(r).all():raise ValueError('Nonfinite residual')
            except (ValueError,RuntimeError,FloatingPointError,OverflowError):
                r=[100.]* (8 if is_fixed(p) else 1444)
            rr.extend(r);groups[name]=float(np.dot(r,r))
        if self.use_spectral_anchors:
            anchors=(model.equilibrium_inputs_at_thresholds(self.anchors)-self.anchors)/.03
            rr.extend(anchors);groups['spectral_I_anchors']=float(anchors@anchors)
        result=np.array(rr)
        return (result,groups) if parts else result


def fit(case,variant,seed,data,objective,budget,*,coordinate_kind=None,gate_kind=None,rng_seed=20261021,checkpoint_path=None,input_threshold_bounds=None,powell_radius=None):
    if powell_radius is not None and (not np.isfinite(powell_radius) or powell_radius<=0):
        raise ValueError('Invalid local Powell radius')
    if 'exact_rest_constraint' in seed.metadata:
        raise ValueError('Use fit_anchored_shapes to preserve the exact rest constraint')
    if seed.has_state_mix:
        raise ValueError('Use fit_mixed_coordinates for coupled coordinate transforms')
    n=seed.latent.region_count;nc=2*n+6;m=n-1;knots=seed.input_thresholds
    move_input=input_threshold_bounds is not None
    if move_input:
        bounds=np.asarray(input_threshold_bounds,dtype=float)
        if (len(knots)!=1 or bounds.shape!=(2,) or not np.isfinite(bounds).all()
                or not 0<bounds[0]<knots[0]<bounds[1]<1.3):
            raise ValueError('Movable input knot requires one threshold inside valid bounds')
    physical=np.array(seed.metadata['physical_thresholds_C']);mapping=OrderedThresholds(physical,data[4],data[5])
    ckind={'sixth_C':'sixth','sixth_root_C':'sixth_root','bounded_C':'bounded_logit','sqrt_tails':'sqrt_tails'}.get(variant,seed.coordinate.kind)
    hkind='sixth' if variant=='sixth_h' else seed.h_kind
    ckind=coordinate_kind or ckind;hkind=gate_kind or hkind
    if ckind==seed.coordinate.kind:co=seed.coordinate
    elif ckind=='sqrt_tails':co=CalciumCoordinate(ckind,seed.coordinate.reference,seed.coordinate.scale)
    else:
        half=(.1 if case=='AM' else .05) if ckind=='hill' else None
        raw=CalciumCoordinate(ckind,seed.coordinate.reference,half_saturation=half)
        endpoint=.9*C_MAX if ckind=='bounded_logit' else C_MAX
        co=CalciumCoordinate(ckind,raw.reference,float(raw.forward(endpoint)/endpoint),half)
    z=np.r_[seed.metadata['parameters'],np.zeros(m),seed.metadata.get('extra_parameters',[]) if len(knots) else [],knots if move_input else []]
    if len(z)!=nc+m+2*len(knots)+int(move_input):raise ValueError('Unexpected input geometry parameter count')
    factor=seed.coordinate.derivative(co.reference)/co.derivative(co.reference)
    z[2*n+1:2*n+3]-=np.log(factor);z[2*n+4:2*n+6]/=factor
    lower=np.r_[np.full(n,-12.),np.full(n,-40.),-.5,-40.,-28.,-10.,-10.,-10.,np.full(m,-np.log(2)),np.full(len(knots),-2.),np.full(len(knots),-np.log(4))]
    upper=np.r_[np.full(n,5.),np.full(n,10.),.5,8.,8.,2.,10.,10.,np.full(m,np.log(2)),np.full(len(knots),2.),np.full(len(knots),np.log(4))]
    if move_input:
        lower=np.r_[lower,bounds[0]];upper=np.r_[upper,bounds[1]]
    for j,label in enumerate(seed.metadata['target_types']):
        if label.endswith('node'):upper[n+j]=12.
    z=np.clip(z,lower+1e-8,upper-1e-8)
    meta=copy.deepcopy(seed.metadata)
    meta.update(experiment_variant=variant,calcium_transform=ckind,h_transform=hkind,
        coordinate_scale=co.scale,threshold_parameter_count=m,coefficient_parameter_count=nc,
        parameter_count=len(z),active_fit_parameters=len(z),accepted=False,
        thresholds_policy='ordered physical C gaps fitted; +/-ln(2)',
        input_knot_locations_fitted=move_input,calcium_clipping=False,
        seed_model=f'models/extended_pwa/selected_{case}.json',
        selected_by='event-weighted calibration with promoted failure cases; fresh tests excluded',
        dynamic_peak_count_residual_weight=objective.peak_weight,fixed_status_residual_weight=25.,
        fixed_status_policy='explicit known status; unresolved references carry no hard label' if objective.explicit_status else 'period presence only')
    def make(p):
        ph=mapping.decode(p[nc:nc+m]);eta=p[-1:] if move_input else knots
        metadata={**meta,'physical_thresholds_C':ph.tolist(),'input_thresholds':eta.tolist()}
        model=decode_coordinates(case,co.forward(ph),p[:nc],seed.metadata['mu_sign'],seed.metadata['k_sign'],
            metadata,co,seed.h_reference,hkind)
        weights=np.zeros((2,len(knots)))
        if len(knots):
            extra=p[nc+m:nc+m+2*len(knots)];B,D=model.latent.matrix[:,2];qb=model.latent.spectral_coefficients()[3]
            slopes=qb*np.r_[1.,np.exp(extra[len(knots):])];a=extra[:len(knots)]
            weights=np.array([a,(np.diff(slopes)+D*a)/B])
        return CalciumTransformPWA(model.latent,co,seed.h_reference,hkind,eta,weights)
    directory=MODELS/variant;directory.mkdir(parents=True,exist_ok=True)
    make(z).save(directory/f'{case}_initial.json')
    points=data[2];template=make(z);transformed=template.to_latent(points);truth=template.source_field(transformed)
    inverse=np.column_stack([1/co.derivative(points[:,0]),1/template.gate.derivative(points[:,1])])
    def field_residual(p):
        return np.r_[((make(p).field(transformed)-truth)*inverse/np.array([.2,.05])).ravel()/np.sqrt(len(points)),.01*(p-z)]
    field=least_squares(field_residual,z,bounds=(lower,upper),max_nfev=60)
    make(field.x).save(directory/f'{case}_field_seed.json')
    count=0;limit=10;best=np.inf;bestz=z.copy();history=[];started=time.monotonic()
    class Limit(Exception):pass
    def residual(p):
        nonlocal count,best,bestz
        if count>=limit:raise Limit()
        count+=1;r=objective(make(p));full=np.r_[r,.005*(p[:nc]-z[:nc]),.03*p[nc:nc+m],.03*(p[nc+m:]-z[nc+m:])]
        loss=float(full@full)
        if loss<best:
            best,bestz=loss,p.copy()
            if not history or count-history[-1]['evaluation']>=100:
                history.append(dict(evaluation=count,calibration_loss=float(r@r),seconds=time.monotonic()-started))
                print(f'{case} {variant} {count}: {r@r:.7g}',flush=True)
                if checkpoint_path is not None:
                    checkpoint=make(p);checkpoint.metadata.update(checkpoint_unverified=True,raw_fit_vector=p.tolist())
                    checkpoint.save(checkpoint_path)
        return full
    rng=np.random.default_rng(rng_seed);perturb=np.clip(z+rng.normal(0,.02,len(z)),lower+1e-8,upper-1e-8)
    candidates=[('tangent',z),('field',field.x),('perturbed',perturb)]
    ranks=sorted([(float(np.dot(r:=residual(p),r)),name,p) for name,p in candidates],key=lambda q:q[0])
    runs=[]
    def jac(p):
        base=residual(p);columns=[]
        for j in range(len(p)):
            step=1e-4*max(1,abs(p[j])) if j<nc else 1e-3
            if p[j]+step>upper[j]:step=-step
            other=p.copy();other[j]+=step;columns.append((residual(other)-base)/step)
        return np.column_stack(columns)
    for _,name,p in ranks[:2]:
        limit=count+budget
        try:
            result=least_squares(residual,p,jac=jac,bounds=(lower,upper),x_scale='jac',max_nfev=100)
            status=str(result.message)
        except Limit:status='residual evaluation limit'
        runs.append(dict(method='least_squares',start=name,status=status))
    limit=count+budget
    powell_lower,powell_upper=lower,upper
    if powell_radius is not None:
        radius=powell_radius*np.maximum(1.,np.abs(bestz))
        powell_lower=np.maximum(lower,bestz-radius);powell_upper=np.minimum(upper,bestz+radius)
    try:
        result=minimize(lambda p:float(np.dot(r:=residual(p),r)),bestz,method='Powell',bounds=list(zip(powell_lower,powell_upper)),options=dict(maxfev=budget,xtol=1e-5,ftol=1e-7))
        status=str(result.message)
    except Limit:status='residual evaluation limit'
    runs.append(dict(method='Powell',status=status))
    model=make(bestz);r,parts=objective(model,parts=True)
    model.metadata.update(parameters=bestz[:nc].tolist(),threshold_shifts=bestz[nc:nc+m].tolist(),
        extra_parameters=bestz[nc+m:nc+m+2*len(knots)].tolist(),optimizer_calibration_loss=float(r@r))
    model.save(directory/f'{case}.json')
    destination=OUT/variant;destination.mkdir(parents=True,exist_ok=True)
    dump(destination/f'{case}_fit.json',dict(evaluations=count,history=history,runs=runs,
        optimizer_loss=float(r@r),parts=parts,bounds=[lower.tolist(),upper.tolist()],
        parameters=bestz.tolist(),initial_scores=[dict(score=s,start=name) for s,name,_ in ranks],
        LS_per_start=budget,Powell_budget=budget,active_parameters=len(z),powell_radius=powell_radius,
        powell_bounds=[powell_lower.tolist(),powell_upper.tolist()]))
    return model


def run(case,budget,variants,pilot=False):
    OUT.mkdir(parents=True,exist_ok=True);MODELS.mkdir(parents=True,exist_ok=True)
    data=training_data(case);seed=CalciumTransformPWA.load(ROOT/f'models/extended_pwa/selected_{case}.json')
    training,promoted=calibration(case,data);refs=dict(data[1])
    tests=fresh_tests(case)
    dump(OUT/f'{case}_protocols.json',dict(calibration=training,promoted_failure_cases=promoted,
        fresh_validation=tests,selection='verified calibration loss only; no fresh validation',
        objective='static extrema/period, known rest/cycling status x25 (unresolved targets unlabelled), half waveform weight, peak count x25, C-bound penalty; no clipping' if not pilot else 'pilot: period presence, half waveform weight, peak count x3, C-bound penalty; no clipping',
        peak_count_residual_weight=3. if pilot else 25.))
    for p in training:
        if p['name'] not in refs:refs[p['name']]=simulate(case,**{k:v for k,v in p.items() if k!='name'},dt=.1 if not is_fixed(p) else .2)
    objective=CalibrationResidual(case,training,refs,seed,peak_weight=3. if pilot else 25.,explicit_status=not pilot)
    for variant in variants:
        if variant=='baseline':
            model=copy.deepcopy(seed);model.metadata.update(experiment_variant='baseline',active_fit_parameters=0,accepted=False)
            (MODELS/'baseline').mkdir(parents=True,exist_ok=True);model.save(MODELS/'baseline'/f'{case}.json')
        else:fit(case,variant,seed,data,objective,budget)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=['AM','FM','both'],default='both')
    parser.add_argument('--budget',type=int,default=800);parser.add_argument('--variants',nargs='+',default=['baseline']+VARIANTS)
    parser.add_argument('--pilot',action='store_true',help='reproduce the archived weaker event penalty')
    args=parser.parse_args()
    if args.pilot:
        OUT=ROOT/'results/positive_coordinates_pilot';MODELS=ROOT/'models/positive_coordinates_pilot'
    for case in (['AM','FM'] if args.case=='both' else [args.case]):run(case,args.budget,args.variants,args.pilot)
