"""Frozen sqrt_tails models: initial preparation, latency, and phase drift."""
import argparse,hashlib,json
import numpy as np
from scipy.optimize import brentq
from scipy.signal import find_peaks,peak_widths
from pwl_nest.models import ROOT
from pwl_nest.strict_pwa import C_MAX,lr_field,equilibrium_h
from pwl_nest.protocols import simulate
from pwl_nest.calcium_coordinates import CalciumTransformPWA,simulate_coordinates
from pwl_nest.phase_comparison import single_shift_comparison
from fit_strict import protocols
from fit_calcium_coordinates import dump

OUT=ROOT/'results/rest_and_phase'
MODES=['original','own_rest','common_LR_rest']


def equilibrium_and_warmup(case,model,baseline):
    grid=np.unique(np.r_[np.geomspace(1e-8,.1,300),np.linspace(.1,C_MAX-1e-8,700)])
    force=lambda c:lr_field([c,equilibrium_h(c,baseline),baseline],case)[0]
    roots=[]
    for left,right in zip(grid[:-1],grid[1:]):
        if force(left)*force(right)<0:
            c=brentq(force,left,right,xtol=1e-14);roots.append([c,float(equilibrium_h(c,baseline)),baseline])
    candidates=[];local=model.at_input(baseline).latent.local_coefficients()
    limits=np.r_[-np.inf,model.latent.thresholds,np.inf]
    for j,a in enumerate(local):
        uv=np.linalg.solve(a[:,1:3],-a[:,0]-baseline*a[:,3])
        if limits[j]<=uv[0]<=limits[j+1]:
            x=model.to_physical(np.r_[uv,baseline])
            if 0<x[0]<C_MAX and 0<x[1]<1:candidates.append(x.tolist())
    if not roots or not candidates:raise RuntimeError('Missing physical equilibrium')
    initial=[.08,.8,baseline]
    lr=simulate(case,initial,1200.,baseline=baseline,dt=2.,rtol=2e-12,atol=2e-14,max_step=.5)
    pwa=simulate_coordinates(model,initial,1200.,baseline=baseline,dt=2.,rtol=2e-12,atol=2e-14,max_step=.5)
    xlr=np.array(min(roots,key=lambda x:np.linalg.norm(x-lr[1][-1])))
    xpwa=np.array(min(candidates,key=lambda x:np.linalg.norm(x-pwa[1][-1])))
    errors=[float(abs(lr[1][-1]-xlr).max()),float(abs(pwa[1][-1]-xpwa).max())]
    if max(errors)>1e-8:raise RuntimeError('Warmup did not reach the selected equilibrium')
    jac=np.column_stack([lr_field(xlr.astype(complex)+1e-25j*np.eye(3)[d],case).imag/1e-25 for d in [0,1]])
    a,_,_=model.region(model.to_latent(xpwa),20.,baseline)
    eigen=[np.linalg.eigvals(jac),np.linalg.eigvals(a[:2,:2])]
    if any(np.max(v.real)>=0 for v in eigen):raise RuntimeError('Selected equilibrium is not stable')
    np.savez_compressed(OUT/f'{case}_warmup.npz',t=lr[0],lr=lr[1],pwa=pwa[1])
    return dict(LR=xlr.tolist(),PWA=xpwa.tolist(),LR_equilibria=roots,PWA_equilibria=candidates,
        warmup_duration=1200.,warmup_max_error=errors,
        max_equilibrium_residual=[float(abs(lr_field(xlr,case)).max()),float(abs(model.field(model.to_latent(xpwa))).max())],
        eigenvalues=[[[float(z.real),float(z.imag)] for z in v] for v in eigen])


def metrics(t,truth,candidate,onset):
    lo,hi=onset+5.,float(t[-1])-5.;middle=(lo+hi)/2
    global_shift=single_shift_comparison(t,truth[:,0],candidate[:,0],(lo,hi))
    early_shift=single_shift_comparison(t,truth[:,0],candidate[:,0],(lo,middle),(middle,hi))
    window=t>=onset
    peaks_a=find_peaks(truth[:,0],prominence=.02)[0];peaks_a=peaks_a[t[peaks_a]>=onset]
    peaks_b=find_peaks(candidate[:,0],prominence=.02)[0];peaks_b=peaks_b[t[peaks_b]>=onset]
    a,b=t[peaks_a],t[peaks_b]
    peak_info=dict(LR_times=a.tolist(),PWA_times=b.tolist(),counts=[len(a),len(b)],
        first_peak_latency_LR=float(a[0]-onset) if len(a) else None,
        first_peak_latency_PWA=float(b[0]-onset) if len(b) else None)
    for label,path,peaks in [('LR',truth,peaks_a),('PWA',candidate,peaks_b)]:
        _,_,left,right=peak_widths(path[:,0],peaks,rel_height=.5)
        widths=np.interp(right,np.arange(len(t)),t)-np.interp(left,np.arange(len(t)),t)
        peak_info[f'{label}_widths_half_prominence_s']=widths.tolist()
        peak_info[f'{label}_heights']=path[peaks,0].tolist()
    if len(a)==len(b) and len(a):
        lag=b-a
        peak_info.update(lags_s=lag.tolist(),first_lag_s=float(lag[0]),last_lag_s=float(lag[-1]),
            lag_span_s=float(np.ptp(lag)),peak_amplitude_MAE=float(abs(candidate[peaks_b,0]-truth[peaks_a,0]).mean()))
        if len(a)>1:peak_info['lag_trend_s_per_100s']=float(np.polyfit(a,lag,1)[0]*100)
        if len(a)>1:peak_info['peak_amplitude_MAE_after_first']=float(abs(candidate[peaks_b[1:],0]-truth[peaks_a[1:],0]).mean())
        # Additional morphology diagnostic only: each peak gets its own centre.
        # This does not replace the single-shift or input-locked response errors.
        relative=np.linspace(-3,3,301);errors=[];used=[]
        for k,(ta,tb) in enumerate(zip(a,b)):
            if min(ta,tb)>=t[0]+3 and max(ta,tb)<=t[-1]-3:
                delta=np.interp(tb+relative,t,candidate[:,0])-np.interp(ta+relative,t,truth[:,0])
                errors.append(float(np.mean(delta*delta)));used.append(k)
        peak_info['peak_centered_shape']=dict(RMSE=float(np.sqrt(np.mean(errors))) if errors else None,
            used_peak_indices=used,half_window_s=3.,amplitude_rescaled=False,
            scope='individual peak centering is a morphology diagnostic, not causal response accuracy')
    return dict(raw_full_RMSE=np.sqrt(np.mean((candidate[window]-truth[window])**2,axis=0)).tolist(),
        global_shift=global_shift,early_shift_late_test=early_shift,peaks=peak_info,
        C_ranges=[list(map(float,[x[window,0].min(),x[window,0].max()])) for x in [truth,candidate]],
        physical=bool((candidate[:,0]>0).all() and (candidate[:,0]<C_MAX).all() and
            (candidate[:,1]>0).all() and (candidate[:,1]<1).all()))


def run(case):
    OUT.mkdir(parents=True,exist_ok=True)
    path=ROOT/f'models/positive_coordinates/sqrt_tails/{case}.json';digest=hashlib.sha256(path.read_bytes()).hexdigest()
    model=CalciumTransformPWA.load(path);start,base=(.65,.3) if case=='AM' else (1.15,.45)
    rest=equilibrium_and_warmup(case,model,base)
    specs=[dict(name=f'decay_tau_{tau}',initial=[.08,.8,start],duration=180.,tau=tau,baseline=base) for tau in [20.,50.,100.]]
    specs += [p for p in protocols(case,heldout=True) if 'jumps' in p['name']]
    fresh=json.loads((ROOT/f'results/positive_coordinates/{case}_protocols.json').read_text())['fresh_validation']
    specs.append(next(p for p in fresh if p['name']=='fresh_tau_65.0'))
    records=[];lr_cache={}
    atol=np.array([min(2e-14,float(model.coordinate.derivative(.005))*2e-14),
        min(2e-14,float(model.gate.derivative(.03))*2e-14),2e-14])
    for mode in MODES:
        for p in specs:
            print(case,mode,p['name'],flush=True)
            common={k:v for k,v in p.items() if k not in ['initial','name']}
            xlr=np.array(p['initial'],dtype=float);xpwa=xlr.copy()
            if mode!='original':
                xlr[:2]=rest['LR'][:2];xpwa[:2]=rest['PWA' if mode=='own_rest' else 'LR'][:2]
            cache_key=(p['name'],tuple(xlr))
            if cache_key not in lr_cache:lr_cache[cache_key]=simulate(case,xlr,**common,dt=.02,rtol=2e-12,atol=2e-14,max_step=.03)
            t,truth,_=lr_cache[cache_key]
            tp,candidate,meta=simulate_coordinates(model,xpwa,**common,dt=.02,rtol=2e-12,atol=atol,max_step=.03)
            np.testing.assert_allclose(t,tp,atol=1e-12,rtol=0)
            np.testing.assert_allclose(truth[:,2],candidate[:,2],atol=1e-12,rtol=0)
            onset=p['jumps'][0][0] if p.get('jumps') else 0.
            result=dict(case=case,mode=mode,protocol=p['name'],onset=onset,
                LR_initial=xlr.tolist(),PWA_initial=xpwa.tolist(),**metrics(t,truth,candidate,onset))
            if mode!='original':
                result['onset_pre_jump_C_h']=dict(LR=rest['LR'][:2],PWA=rest['PWA' if mode=='own_rest' else 'LR'][:2])
                if p.get('jumps'):result['onset_pre_jump_C_h']=None
            records.append(result)
            np.savez_compressed(OUT/f'{case}_{mode}_{p["name"]}.npz',t=t,lr=truth,pwa=candidate,
                latent=meta['latent'])
            dump(OUT/f'{case}.json',dict(case=case,model=str(path.relative_to(ROOT)),model_sha256=digest,
                equilibrium_preparation=rest,protocols=specs,results=records,
                controls='physical initial conditions fixed by protocol or equilibrium; no parameter fitting',
                alignment='diagnostic scalar time shift only; physical inputs and outputs are not retimed'))
    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise AssertionError('Model changed during the experiment')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=['AM','FM','both'],default='both');args=parser.parse_args()
    for case in (['AM','FM'] if args.case=='both' else [args.case]):run(case)
