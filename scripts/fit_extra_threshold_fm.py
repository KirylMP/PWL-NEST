"""Exploratory FM fit with one additional affine calcium threshold.

Runs the same objective for a split inside the slow low-C band and for a new
upper-C band. Original candidate and all data remain untouched.
"""
import json, time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from pwl_nest.models import ROOT
from pwl_nest.calcium_strips import CalciumStrips, decode, NativeStatistics, NativeDynamic
from pwl_nest.protocols import simulate
from fit_strict import protocols, stats

OUT=ROOT/'results/extra_threshold_fm_20261023'
SEED=ROOT/'models/calcium_strips/FM.json'

def fit_arm(label, insert, bounds_C, budget=100):
    old=CalciumStrips.load(SEED); oldmeta=old.metadata.copy()
    theta=old.thresholds.copy(); n=len(theta)+1
    z=np.asarray(oldmeta['parameters'],float)
    # New knot initially duplicates the neighboring region, so the inserted
    # threshold does not alter the incumbent vector field.
    newtheta=np.insert(theta,insert,bounds_C[0]+.5*(bounds_C[1]-bounds_C[0]))
    duplicate=min(insert,n-1)
    nz=np.r_[z[:insert],z[duplicate],z[insert:n],
             z[n:n+insert],z[n+duplicate],z[n+insert:2*n],z[2*n:]]
    # z layout: log|mu|[n], log|k|[n], then six shared parameters.
    target=list(oldmeta['target_types']); target.insert(insert,target[duplicate])
    ms=np.sign(old.spectral_coefficients()[0]); ks=np.sign(old.spectral_coefficients()[1])
    ms=np.insert(ms,insert,ms[duplicate]);ks=np.insert(ks,insert,ks[duplicate])
    meta={**oldmeta,'target_types':target,'regions':n+1,'free_parameters':2*(n+1)+6,
          'thresholds_policy':'original thresholds plus fitted extra knot','accepted':False}
    # Inserting one region parameter means inserting into both spectral blocks.
    def make(p):
        zz=p[:-1]; knots=newtheta.copy();knots[insert]=p[-1]
        return decode('FM',knots,zz,ms,ks,meta)
    train=protocols('FM'); held=protocols('FM',heldout=True)
    refs={}
    for proto in train:
        refs[proto['name']]=simulate('FM',**{k:v for k,v in proto.items() if k!='name'})
    static=NativeStatistics();dynamic=NativeDynamic()
    tgrid=np.linspace(0,1,361)
    def residual(p):
        model=make(p); rr=[]
        for proto in train:
            name=proto['name']; tr,truth,_=refs[name]
            if name.startswith('fixed'):
                target_stats=stats(tr,truth)
                got=static.run(model,proto['baseline'],proto['initial'],proto['duration'])
                if got is None: rr.extend([100.]*4);continue
                mn,mx,period,ratio,minall,stopped=got
                rr.extend([(mn-target_stats['Cmin'])/.02,(mx-target_stats['Cmax'])/.05,
                    (period-target_stats['period'])/3 if target_stats['period'] is not None and np.isfinite(period) else 15.,
                    min(0.,minall)/.01])
            else:
                out=dynamic.run(model,proto['initial'],proto['duration'],proto['tau'],proto['baseline'],outputs=361)
                if out is None: rr.extend(np.full(722,100.));continue
                # Match the training reference at its actual output times.
                sample=np.column_stack([np.interp(np.linspace(tr[0],tr[-1],361),tr,truth[:,j]) for j in (0,1)])
                rr.extend(((out-sample)/np.array([.04,.08])/np.sqrt(361)).ravel())
        # A modest trust region keeps the original fit as a useful anchor;
        # threshold itself is free throughout the selected source interval.
        rr.extend(.01*(p[:-1]-nz))
        return np.asarray(rr)
    # Duplicate insertion can change indexing across the two parameter blocks;
    # construct a 2*(n+1)+6 vector explicitly from the original encoding.
    # The first insertion above is used for mu; rebuild k block independently.
    oldn=n
    nz=np.r_[z[:insert],z[duplicate],z[insert:oldn],
             z[oldn:oldn+insert],z[oldn+duplicate],z[oldn+insert:2*oldn],z[2*oldn:]]
    p0=np.r_[nz,newtheta[insert]]
    initial_model=make(p0)
    probes=np.column_stack([np.linspace(.01,.9,80),np.linspace(.2,.95,80),np.linspace(.1,1.1,80)])
    preservation_error=float(np.max(np.abs(initial_model.field(probes)-old.field(probes))))
    if preservation_error>2e-9:
        raise RuntimeError(f'Inserted knot changed incumbent field at initialization: {preservation_error:g}')
    lo=np.r_[np.full(2*(n+1),-12.),[-4.,-12.,-5.,-7.,-5.,-5.],bounds_C[0]]
    hi=np.r_[np.full(2*(n+1),5.),[4.,2.,4.,2.,5.,5.],bounds_C[1]]
    started=time.monotonic(); history=[]
    def counted(p):
        r=residual(p); loss=float(r@r)
        if not history or loss<history[-1]['loss']:
            history.append(dict(loss=loss,seconds=time.monotonic()-started,threshold_C=float(p[-1])))
        return r
    result=least_squares(counted,p0,bounds=(lo,hi),max_nfev=budget,x_scale='jac',
                         ftol=1e-6,xtol=1e-6,gtol=1e-6,verbose=0)
    candidate=make(result.x)
    baseline_loss=float(residual(p0)@residual(p0)); fit_loss=float(residual(result.x)@residual(result.x))
    validation=[]
    for proto in held:
        kw={k:v for k,v in proto.items() if k!='name'}
        ref=simulate('FM',**kw); pred=simulate('FM',**kw,model=candidate,enforce_domain=False,rtol=2e-9,atol=2e-11,max_step=.15)
        a,b=ref[1][:,0],pred[1][:,0]
        validation.append(dict(name=proto['name'],rmse_C=float(np.sqrt(np.mean((a-b)**2))),
            peaks_ref=int(np.sum((a[1:-1]>a[:-2])&(a[1:-1]>=a[2:]))),
            peaks_fit=int(np.sum((b[1:-1]>b[:-2])&(b[1:-1]>=b[2:]))),
            C_range=[float(b.min()),float(b.max())]))
    candidate.metadata.update(parameters=result.x[:-1].tolist(),extra_threshold_C=float(result.x[-1]),
        extra_threshold_arm=label,training_loss=fit_loss,accepted=False,
        optimizer=dict(status=int(result.status),nfev=int(result.nfev),message=result.message))
    dest=OUT/label;dest.mkdir(parents=True,exist_ok=True);candidate.save(dest/'FM.json')
    (dest/'summary.json').write_text(json.dumps(dict(label=label,threshold_C=float(result.x[-1]),
        training_loss_before=baseline_loss,training_loss_after=fit_loss,optimizer_nfev=int(result.nfev),
        optimizer_message=result.message,incumbent_field_preservation_max_error=preservation_error,
        validation=validation,history=history),indent=2,allow_nan=False)+'\n')
    print(label,json.dumps(dict(threshold_C=float(result.x[-1]),before=baseline_loss,after=fit_loss,validation=validation)),flush=True)

if __name__=='__main__':
    fit_arm('split_slow_region',4,(.06129,.13570))
    fit_arm('add_upper_region',7,(.39263,.95))
