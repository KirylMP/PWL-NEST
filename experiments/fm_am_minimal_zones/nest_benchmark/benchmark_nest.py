#!/usr/bin/env python3
"""Benchmark AM/FM PWA and matched Li-Rinzel in NEST 3.10.

The compiled extension must be on LD_LIBRARY_PATH and generated from the
candidate files being validated. By default, timings cover nest.Run after Prepare.
"""
import argparse,ctypes,hashlib,json,os,platform,statistics,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import nest
from scipy.optimize import brentq
from latent_flow import Candidate
from validate_latent_flow import reference

HERE=Path(__file__).resolve().parent
PROFILE_NAMES=('advance_calls','advance_ns','crossings','crossing_ns','inverse_calls','inverse_ns','sic_calls','sic_ns','sic_active_calls','exponential_calls','exponential_ns','cached_full_steps')


def profile_library():
    if os.environ.get('PWA_PROFILE','0')!='1':return None
    path=next((Path(d)/'pwa_astrocyte_module.so' for d in os.environ.get('LD_LIBRARY_PATH','').split(os.pathsep) if d and (Path(d)/'pwa_astrocyte_module.so').is_file()),None)
    if path is None:raise RuntimeError('Instrumented PWA module not found in LD_LIBRARY_PATH')
    lib=ctypes.CDLL(str(path.resolve()))
    try:lib.pwa_profile_reset.argtypes=[];lib.pwa_profile_snapshot.argtypes=[ctypes.POINTER(ctypes.c_uint64)]
    except AttributeError as e:raise RuntimeError('Module lacks profiling symbols; rebuild with PWA_PROFILE=1') from e
    return lib


def profile_snapshot(lib):
    out=(ctypes.c_uint64*len(PROFILE_NAMES))();lib.pwa_profile_snapshot(out)
    return dict(zip(PROFILE_NAMES,map(int,out)))


def work_initial(case,I):
    K=.1 if case=='am' else .05
    def h(c):
        q=1.049*(I+.13)/(I+.9434);return q/(q+c)
    def field(c):
        return (6*(I/(I+.13))**3*(c/(c+.08234))**3*h(c)**3+.11)*(2-1.185*c)-.9*c*c/(c*c+K*K)
    c=brentq(field,1e-6,.2);return float(c),float(h(c))


def pwa_initial(case,I):
    """Choose a stable physical equilibrium of the compiled PWA candidate."""
    manifest=json.loads((HERE/'module'/'generated'/'manifest.json').read_text())
    candidate=Candidate(manifest[case]['path']);solutions=[]
    for j,coeff in enumerate(candidate.coefficients):
        A=coeff[:,1:3]
        if abs(np.linalg.det(A))<1e-12 or np.max(np.real(np.linalg.eigvals(A)))>=-1e-9:
            continue
        z,h=np.linalg.solve(A,-coeff[:,0]-coeff[:,3]*I)
        low=-np.inf if j==0 else candidate.z_thresholds[j-1]
        high=np.inf if j==len(candidate.coefficients)-1 else candidate.z_thresholds[j]
        if not (low-1e-10<=z<=high+1e-10 and np.isfinite(h) and 0<=h<=1):
            continue
        try:C=float(candidate.inverse(np.asarray([z]))[0])
        except ValueError:continue
        solutions.append((C,float(h)))
    if not solutions:raise RuntimeError(f'{case}: no stable physical PWA equilibrium at IP3={I}')
    reference_C,_=work_initial(case,I)
    return min(solutions,key=lambda q:abs(q[0]-reference_C))


def run(model,case,duration_ms,dt_ms,threads,nodes,connected,record,timing_mode="simulate",protocol="dynamic"):
    nest.ResetKernel()
    nest.SetKernelStatus({'resolution':dt_ms,'local_num_threads':threads,'print_time':False})
    nest.Install('pwa_astrocyte_module')
    profiler=profile_library()
    if protocol=='work-pulse':
        baseline=.2 if case=='am' else .45
        init_mode=os.environ.get('PWA_INIT_MODE','equilibrium')
        if init_mode not in ('equilibrium','lr'):raise ValueError('PWA_INIT_MODE must be equilibrium or lr')
        c0,h0=(pwa_initial(case,baseline) if model!='astrocyte_lr_1994' and init_mode=='equilibrium' else work_initial(case,baseline));tau_ms=20000.;delta=.7;delay=10.
        if duration_ms<=20000.:raise ValueError('work-pulse duration must exceed 20 seconds')
        if abs(delay/dt_ms-round(delay/dt_ms))>1e-9:raise ValueError('10 ms connection delay must be an integer number of simulation steps')
        spike_ms=[20000.-delay]
    else:
        baseline=.4 if case=='am' else .55
        c0,h0=.1,.75;tau_ms=30000.;delta=.15;delay=dt_ms
        nsteps=int(round(duration_ms/dt_ms));spike_steps=sorted(set(int(nsteps*x) for x in (.2,.45,.7)))
        spike_ms=[float(x*dt_ms) for x in spike_steps if x>=1 and x+1<nsteps]
    pars={'IP3_0':baseline,'IP3':baseline,'tau_IP3':tau_ms,'Ca_astro':c0,'h_IP3R':h0,'delta_IP3':delta}
    if model=='astrocyte_lr_1994':pars['Km_SERCA']=.1 if case=='am' else .05
    astro=nest.Create(model,n=nodes,params=pars)
    generator=nest.Create('spike_generator',params={'spike_times':spike_ms})
    nest.Connect(generator,astro,syn_spec={'weight':1.,'delay':delay})
    if connected:
        neurons=nest.Create('aeif_cond_alpha_astro',n=nodes)
        nest.Connect(astro,neurons,conn_spec='one_to_one',syn_spec={'synapse_model':'sic_connection','weight':1.})
    meter=None;post_meter=None
    if record:
        meter=nest.Create('multimeter',params={'interval':dt_ms,'record_from':['Ca_astro','IP3','h_IP3R']})
        nest.Connect(meter,astro[:1])
        if connected:
            post_meter=nest.Create('multimeter',params={'interval':dt_ms,'record_from':['I_SIC']})
            nest.Connect(post_meter,neurons[:1])
    parts={}
    if timing_mode=='run':
        start=time.perf_counter();nest.Prepare();parts['prepare_seconds']=time.perf_counter()-start
        try:
            if profiler:profiler.pwa_profile_reset()
            start=time.perf_counter();nest.Run(duration_ms);elapsed=time.perf_counter()-start
            if profiler:parts['pwa_profile']=profile_snapshot(profiler)
        finally:
            start=time.perf_counter();nest.Cleanup();parts['cleanup_seconds']=time.perf_counter()-start
    else:
        if profiler:profiler.pwa_profile_reset()
        start=time.perf_counter();nest.Simulate(duration_ms);elapsed=time.perf_counter()-start
        if profiler:parts['pwa_profile']=profile_snapshot(profiler)
    if profiler and model!='astrocyte_lr_1994' and not parts['pwa_profile']['advance_calls']:
        raise RuntimeError('PWA profiling reported zero propagation calls; check the loaded module')
    parts['timed_seconds']=elapsed;parts['timing_mode']=timing_mode
    events=None
    if meter:
        ev=meter.get('events')
        events={k:np.asarray(ev[k],dtype=float) for k in ('times','Ca_astro','IP3','h_IP3R')}
        if post_meter:
            pe=post_meter.get('events');events['sic_times']=np.asarray(pe['times'],dtype=float);events['I_SIC']=np.asarray(pe['I_SIC'],dtype=float)
    return elapsed,events,dict(spike_times_ms=spike_ms,delay_ms=delay,baseline_ip3=baseline,delta_ip3=delta,tau_ip3_ms=tau_ms,initial_C=c0,initial_h=h0,initialization=(os.environ.get('PWA_INIT_MODE','equilibrium') if protocol=='work-pulse' and model!='astrocyte_lr_1994' else ('lr_equilibrium' if protocol=='work-pulse' else 'shared_dynamic')),protocol=protocol,case=case,model=model,connected_sic=connected,nodes=nodes,threads=threads,resolution_ms=dt_ms,timing_breakdown=parts)


def validate_candidate(case,manifest,duration_ms,dt_ms,protocol):
    data=manifest[case];path=Path(data['path']);raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=data['sha256']:raise ValueError(f'{case}: candidate changed after compilation; regenerate module')
    model=Candidate(path);elapsed,ev,proto=run(f'astrocyte_{case}_pwa',case,duration_ms,dt_ms,1,1,True,True,protocol=protocol)
    times=ev['times']/1000.;jump_times=[(s+proto['delay_ms'])/1000. for s in proto['spike_times_ms']]
    jumps=[(s,proto['delta_ip3']) for s in jump_times]
    expected=reference(model,[proto['initial_C'],proto['initial_h'],proto['baseline_ip3']],proto['baseline_ip3'],proto['tau_ip3_ms']/1000.,duration_ms/1000.,jumps,times)
    got=np.column_stack([ev['Ca_astro'],ev['h_IP3R'],ev['IP3']]);err=np.abs(got-expected)
    maxerr=err.max(axis=0);rms=np.sqrt(np.mean(err**2,axis=0))
    if maxerr[0]>2e-4 or maxerr[1]>2e-4 or maxerr[2]>2e-8:raise RuntimeError(f'{case}: NEST differs from independent solver: max error {maxerr}')
    return dict(protocol=proto,samples=len(times),max_abs_error_C_h_I=maxerr.tolist(),rmse_C_h_I=rms.tolist(),validation_seconds=elapsed),ev


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--duration-s',type=float,default=20.);p.add_argument('--resolution-ms',type=float,default=.1);p.add_argument('--nodes',type=int,default=4);p.add_argument('--threads',type=int,default=1);p.add_argument('--repeats',type=int,default=3);p.add_argument('--validation-s',type=float,default=20.);p.add_argument('--protocol',choices=['dynamic','work-pulse'],default='dynamic');p.add_argument('--timing-mode',choices=['run','simulate'],default='run');p.add_argument('--warmup',type=int,default=1);p.add_argument('--output',type=Path,default=HERE/'results'/'nest_benchmark.json');a=p.parse_args()
    if a.duration_s<=0 or a.validation_s<=0 or a.resolution_ms<=0 or min(a.nodes,a.threads,a.repeats)<1 or a.warmup<0:p.error('positive durations, resolution, node/thread and repeat counts required')
    if a.protocol=='work-pulse' and min(a.duration_s,a.validation_s)<=20:p.error('work-pulse requires durations longer than 20 seconds')
    if min(a.duration_s,a.validation_s)*1000/a.resolution_ms<10:p.error('each simulation must contain at least 10 resolution steps')
    for value in (a.duration_s*1000,a.validation_s*1000):
        if abs(value/a.resolution_ms-round(value/a.resolution_ms))>1e-7:p.error('durations must be integer multiples of resolution')
    if nest.__version__.split('.')[:2]!=['3','10']:raise RuntimeError('This module is adapted to NEST 3.10')
    manifest=json.loads((HERE/'module'/'generated'/'manifest.json').read_text());validation={};approximation={};traces={}
    module_file=next((Path(d)/'pwa_astrocyte_module.so' for d in os.environ.get('LD_LIBRARY_PATH','').split(os.pathsep) if d and (Path(d)/'pwa_astrocyte_module.so').is_file()),None)
    if module_file is None:raise RuntimeError('PWA module not found in LD_LIBRARY_PATH; run the provided run_benchmarks.sh')
    module_info=dict(path=str(module_file.resolve()),sha256=hashlib.sha256(module_file.read_bytes()).hexdigest())
    for case in ('am','fm'):
        validation[case],pwa_trace=validate_candidate(case,manifest,a.validation_s*1000,a.resolution_ms,a.protocol)
        print('VALIDATION',case,json.dumps(validation[case]),flush=True)
        _,lr_trace,lr_proto=run('astrocyte_lr_1994',case,a.validation_s*1000,a.resolution_ms,1,1,True,True,protocol=a.protocol)
        if not np.array_equal(lr_trace['times'],pwa_trace['times']) or not np.array_equal(lr_trace['sic_times'],pwa_trace['sic_times']):
            raise RuntimeError('LR and PWA recording grids differ')
        errors={}
        for key in ('Ca_astro','h_IP3R','IP3','I_SIC'):
            delta=pwa_trace[key]-lr_trace[key]
            errors[key]=dict(rmse=float(np.sqrt(np.mean(delta**2))),max_abs=float(np.max(abs(delta))))
        approximation[case]=dict(errors=errors,lr_initial_C=lr_proto['initial_C'],lr_initial_h=lr_proto['initial_h'],pwa_initial_C=validation[case]['protocol']['initial_C'],pwa_initial_h=validation[case]['protocol']['initial_h'],pwa_initialization=validation[case]['protocol']['initialization'],scope='One driven protocol; baseline NEST uses its built-in GSL tolerances')
        for model_name,events in [('pwa',pwa_trace),('lr',lr_trace)]:
            for key,values in events.items():traces[f'{case}_{model_name}_{key}']=values
        print('LR COMPARISON',case,json.dumps(errors),flush=True)
    report=[]
    for connected in (False,True):
      for case in ('am','fm'):
        models=('astrocyte_lr_1994',f'astrocyte_{case}_pwa')
        samples={name:[] for name in models};details={name:[] for name in models};protocols={}
        for repeat in range(a.warmup+a.repeats):
          # Alternate order to reduce drift between LR and PWA observations.
          order=models if repeat%2==0 else models[::-1]
          for model in order:
            t,_,proto=run(model,case,a.duration_s*1000,a.resolution_ms,a.threads,a.nodes,connected,False,a.timing_mode,a.protocol)
            kind='WARMUP' if repeat<a.warmup else 'TIMING'
            print(f'{kind} case={case} model={model} sic={connected} repeat={repeat+1} seconds={t:.6g}',flush=True)
            if repeat>=a.warmup:
                samples[model].append(t);details[model].append(proto['timing_breakdown']);protocols[model]=proto
        for model in models:
          proto=protocols[model].copy();proto.pop('timing_breakdown',None)
          report.append(dict(protocol=proto,seconds=samples[model],timing_breakdowns=details[model],median_seconds=statistics.median(samples[model])))
    speeds=[]
    for case in ('am','fm'):
      for connected in (False,True):
        lr=next(x['median_seconds'] for x in report if x['protocol']['case']==case and x['protocol']['connected_sic']==connected and x['protocol']['model']=='astrocyte_lr_1994')
        pwa=next(x['median_seconds'] for x in report if x['protocol']['case']==case and x['protocol']['connected_sic']==connected and x['protocol']['model']!= 'astrocyte_lr_1994')
        speeds.append(dict(case=case,connected_sic=connected,LR_seconds=lr,PWA_seconds=pwa,speedup=lr/pwa))
    output=dict(generated_utc=datetime.now(timezone.utc).isoformat(),nest_version=nest.__version__,platform=platform.platform(),logical_cpu_count=os.cpu_count(),duration_s=a.duration_s,validation_s=a.validation_s,resolution_ms=a.resolution_ms,nodes=a.nodes,threads=a.threads,module=module_info,candidate_manifest=manifest,validation=validation,approximation_vs_LR=approximation,protocol=a.protocol,timing_mode=a.timing_mode,warmup_repeats=a.warmup,timings=report,speedups=speeds,caveats=['Local numerical verification covers one finite protocol; it does not establish approximation quality against LR.', 'CurrentEvent inputs are unsupported by the PWA adapter.', 'Finite step threshold crossing bracketing may miss crossings.', 'Wall time uses nest.Run after nest.Prepare by default; prepare and cleanup are reported separately. Simulate mode includes them.'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    trace_path=a.output.with_name(a.output.stem+'_traces.npz');np.savez_compressed(trace_path,**traces);output['trace_file']=str(trace_path)
    tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(output,indent=2,allow_nan=False));tmp.replace(a.output)
    print('SPEEDUPS',json.dumps(speeds,indent=2),flush=True);print(f'Saved {a.output}',flush=True)

if __name__=='__main__':main()
