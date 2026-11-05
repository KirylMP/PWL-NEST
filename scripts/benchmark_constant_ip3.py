"""Constant-IP3 LR/PWA sweep using the runtime-configurable NEST PWA module.

Times in seconds except NEST API values marked ms. Hidden q is not LR h.
"""
import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'paper/data/physical_models/no_fold'
KER = {'AM': .1, 'AFM': .075, 'FM': .05}
SELECTED = ROOT / 'paper/data/selected_lr_candidates/manifest.json'
RANDOM_SELECTED = ROOT / 'paper/data/random_selected/manifest.json'


def load_case(case, model_set='selected'):
    if model_set in ('selected', 'random_selected'):
        manifest_path = SELECTED if model_set == 'selected' else RANDOM_SELECTED
        manifest = json.loads(manifest_path.read_text())['models'][case]
        path = ROOT / manifest['candidate_path']
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if sha != manifest['candidate_sha256']:
            raise ValueError(f'{case}: candidate hash mismatch')
    elif model_set == 'no_fold':
        path = MODEL_DIR / case / 'parameters.json'
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
    else:
        raise ValueError(f'Unknown model set: {model_set}')
    data = json.loads(raw)
    if data.get('case') not in ({'AFM', 'INTERMEDIATE'} if case == 'AFM' else {case}):
        raise ValueError(f'{case}: candidate case mismatch')
    if abs(float(data['K']) - KER[case]) > 1e-10:
        raise ValueError(f'{case}: candidate SERCA parameter mismatch')
    return data, sha


def parameters(data, I, C0, q0):
    """Encode C=cref+u, q=href+href*(1-href)*v into module coefficients."""
    N, H = np.array(data['alpha']), np.array(data['beta'])
    cuts = np.array(data['physical_thresholds_C'])
    a, b, cref, href = float(data['a']), float(data['b']), 1e-6, .5
    scale = href*(1-href)
    rows = []
    for j in range(3):
        n = N[1] + N[3:3+j].sum(); h = H[1] + H[3:3+j].sum()
        nc = N[0] - N[3:3+j] @ cuts[:j]
        hc = H[0] - H[3:3+j] @ cuts[:j]
        rows.extend([a*(href-nc-n*cref), -a*n, a*scale, -a*N[2],
                     b*(hc+h*cref-href)/scale, b*h/scale, -b, b*H[2]/scale])
    return dict(thresholds=(cuts-cref).tolist(), coefficients=rows, input_thresholds=[],
                c_coordinate=0, c_reference=cref, c_scale=1., h_coordinate=0,
                h_reference=href, u=C0-cref, v=(q0-href)/scale, IP3=I, IP3_0=I,
                tau_IP3=20000., delta_IP3=0.)


def peaks(t, c, transient):
    mask=t>=transient; x=c[mask]; tt=t[mask]
    if len(x)<3: return dict(status='insufficient_data', peaks=0)
    if not np.isfinite(x).all(): return dict(status='nonfinite', peaks=0, physical_calcium=False)
    ind,_=find_peaks(x, prominence=.001)
    periods=np.diff(tt[ind]); cmin=float(x.min()); cmax=float(x.max())
    if len(ind)<3: status='insufficient_peaks'
    elif periods[-1] <= 0 or abs(periods[-1]-periods[-2]) > max(.1,.01*periods[-1]): status='not_converged'
    else: status='oscillatory'
    return dict(status=status, peaks=int(len(ind)), Cmin=cmin, Cmax=cmax,
                last_period_s=float(periods[-1]) if len(periods) else None,
                physical_calcium=bool(np.isfinite(x).all() and np.all((x>0))))


def run(case, model, data, I, args, diagnostic=False):
    import nest
    nest.ResetKernel()
    nest.SetKernelStatus({'resolution':args.dt_ms, 'local_num_threads':args.threads, 'print_time':False})
    if 'astrocyte_lr_pwa' not in nest.node_models:
        nest.Install(str(ROOT / '.build/pwl_astro_module'))
    C0,q0=map(float,data['rest_target'])
    if model=='PWA':
        params=parameters(data,I,C0,q0); name='astrocyte_lr_pwa'
    else:
        params=dict(Ca_astro=C0,h_IP3R=q0,IP3=I,IP3_0=I,tau_IP3=20000.,delta_IP3=0.,Km_SERCA=KER[case]);name='astrocyte_lr_1994'
    start=time.perf_counter(); nodes=nest.Create(name,n=1 if diagnostic else args.nodes,params=params)
    meter=None
    if diagnostic:
        meter=nest.Create('multimeter',params={'interval':args.interval_ms,'record_from':['Ca_astro','IP3']})
        nest.Connect(meter,nodes)
    build=time.perf_counter()-start
    start=time.perf_counter();nest.Prepare();prepare=time.perf_counter()-start
    try:
        start=time.perf_counter();nest.Run(args.duration_s*1000);runtime=time.perf_counter()-start
    finally:
        start=time.perf_counter();nest.Cleanup();cleanup=time.perf_counter()-start
    out=dict(build_s=build,prepare_s=prepare,run_s=runtime,cleanup_s=cleanup,
             node_count=1 if diagnostic else args.nodes, initial_C=C0,initial_hidden=q0)
    if model=='PWA':
        out['counters']={k:int(np.sum(nodes.get(k))) for k in ('affine_flows','region_crossings','input_crossings','input_spikes')}
    if diagnostic:
        ev=meter.get('events'); t=np.asarray(ev['times'])/1000.; C=np.asarray(ev['Ca_astro']); ip=np.asarray(ev['IP3'])
        out['dynamics']=peaks(t,C,args.transient_s)
        out['max_ip3_drift']=float(np.max(np.abs(ip-I)))
        out['trace']={'time_s':t.tolist(),'C':C.tolist(),'IP3':ip.tolist()}
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases',nargs='+',choices=sorted(KER),default=['AM','AFM','FM'])
    p.add_argument('--model-set',choices=['selected','random_selected','no_fold'],default='selected')
    p.add_argument('--inputs',nargs='+',type=float,default=[.05,.2,.3,.34,.35,.4,.5,.53,.6,.64,.7,.77,.8,.86,.9,1.,1.12,1.2])
    p.add_argument('--nodes',type=int,default=32);p.add_argument('--threads',type=int,default=1)
    p.add_argument('--duration-s',type=float,default=300.);p.add_argument('--transient-s',type=float,default=150.)
    p.add_argument('--dt-ms',type=float,default=.1);p.add_argument('--interval-ms',type=float,default=50.)
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--warmup',type=int,default=1)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if min(args.nodes,args.threads,args.repeats)<1 or args.warmup<0 or not 0<=args.transient_s<args.duration_s or min(args.inputs)<0 or max(args.inputs)>1.3: p.error('Invalid arguments')
    for x in (args.duration_s*1000,args.interval_ms):
        if x<=0 or abs(x/args.dt_ms-round(x/args.dt_ms))>1e-7:p.error('Duration and recording interval must align with resolution')
    import nest
    module=ROOT/'.build/pwl_astro_module.so'
    if not module.is_file():raise RuntimeError(f'Module missing: {module}; run scripts/build_nest_module.py')
    args.output.mkdir(parents=True,exist_ok=True)
    meta=dict(nest_version=nest.__version__,platform=platform.platform(),processor=platform.processor(),
              module_sha256=hashlib.sha256(module.read_bytes()).hexdigest(),settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              caveats=['Constant IP3; finite forward observation; q is hidden and not LR h.',
                       'Per-run crossing counters include transient; timing includes PWA counter overhead.',
                       'Fixed-dt NEST; no recording in timed runs; diagnostic run has a separate single node.'])
    output=args.output/'runs.jsonl'
    if output.exists():raise FileExistsError(f'Refusing to overwrite {output}; choose a new output directory')
    (args.output/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    with output.open('x') as f:
        for case in args.cases:
            data,sha=load_case(case,args.model_set)
            for I in args.inputs:
                diagnostic={}
                for model in ('LR','PWA'):
                    try:diagnostic[model]=run(case,model,data,I,args,True)
                    except Exception as exc: diagnostic[model]={'error':f'{type(exc).__name__}: {exc}'}
                for repeat in range(-args.warmup,args.repeats):
                    for model in (('LR','PWA') if repeat%2==0 else ('PWA','LR')):
                        try:result=run(case,model,data,I,args)
                        except Exception as exc:result={'error':f'{type(exc).__name__}: {exc}'}
                        row=dict(case=case,I=I,model=model,repeat=repeat,warmup=repeat<0,candidate_sha256=sha,result=result,diagnostics=diagnostic if repeat==0 else None)
                        f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
                print(case,I,diagnostic.get('LR',{}).get('dynamics',{}).get('status'),diagnostic.get('PWA',{}).get('dynamics',{}).get('status'),flush=True)
    print('Saved',output)


if __name__=='__main__':main()
