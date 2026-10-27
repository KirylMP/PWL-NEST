#!/usr/bin/env python3
"""Benchmark AM/FM PWA and matched Li-Rinzel in NEST 3.10.

The compiled extension must be on LD_LIBRARY_PATH and generated from the
candidate files being validated. Timings cover nest.Simulate, not model setup.
"""
import argparse,hashlib,json,os,platform,statistics,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import nest
from latent_flow import Candidate
from validate_latent_flow import reference

HERE=Path(__file__).resolve().parent


def run(model,case,duration_ms,dt_ms,threads,nodes,connected,record):
    nest.ResetKernel()
    nest.SetKernelStatus({'resolution':dt_ms,'local_num_threads':threads,'print_time':False})
    nest.Install('pwa_astrocyte_module')
    baseline=.4 if case=='am' else .55
    pars={'IP3_0':baseline,'IP3':baseline,'tau_IP3':30000.,'Ca_astro':.1,'h_IP3R':.75,'delta_IP3':.15}
    if model=='astrocyte_lr_1994':pars['Km_SERCA']=.1 if case=='am' else .05
    astro=nest.Create(model,n=nodes,params=pars)
    # All comparisons get the same deterministic spike times and one-step delay.
    nsteps=int(round(duration_ms/dt_ms));spike_steps=sorted(set(int(nsteps*x) for x in (.2,.45,.7)))
    spike_ms=[float(x*dt_ms) for x in spike_steps if x>=1 and x+1<nsteps]
    generator=nest.Create('spike_generator',params={'spike_times':spike_ms})
    nest.Connect(generator,astro,syn_spec={'weight':1.,'delay':dt_ms})
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
    start=time.perf_counter();nest.Simulate(duration_ms);elapsed=time.perf_counter()-start
    events=None
    if meter:
        ev=meter.get('events')
        events={k:np.asarray(ev[k],dtype=float) for k in ('times','Ca_astro','IP3','h_IP3R')}
        if post_meter:
            pe=post_meter.get('events');events['sic_times']=np.asarray(pe['times'],dtype=float);events['I_SIC']=np.asarray(pe['I_SIC'],dtype=float)
    return elapsed,events,dict(spike_times_ms=spike_ms,delay_ms=dt_ms,baseline_ip3=baseline,delta_ip3=.15,tau_ip3_ms=30000.,case=case,model=model,connected_sic=connected,nodes=nodes,threads=threads,resolution_ms=dt_ms)


def validate_candidate(case,manifest,duration_ms,dt_ms):
    data=manifest[case];path=Path(data['path']);raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=data['sha256']:raise ValueError(f'{case}: candidate changed after compilation; regenerate module')
    model=Candidate(path);elapsed,ev,proto=run(f'astrocyte_{case}_pwa',case,duration_ms,dt_ms,1,1,True,True)
    times=ev['times']/1000.;jump_times=[(s+proto['delay_ms'])/1000. for s in proto['spike_times_ms']]
    jumps=[(s,proto['delta_ip3']) for s in jump_times]
    expected=reference(model,[.1,.75,proto['baseline_ip3']],proto['baseline_ip3'],30.,duration_ms/1000.,jumps,times)
    got=np.column_stack([ev['Ca_astro'],ev['h_IP3R'],ev['IP3']]);err=np.abs(got-expected)
    maxerr=err.max(axis=0);rms=np.sqrt(np.mean(err**2,axis=0))
    if maxerr[0]>2e-4 or maxerr[1]>2e-4 or maxerr[2]>2e-8:raise RuntimeError(f'{case}: NEST differs from independent solver: max error {maxerr}')
    return dict(protocol=proto,samples=len(times),max_abs_error_C_h_I=maxerr.tolist(),rmse_C_h_I=rms.tolist(),validation_seconds=elapsed),ev


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--duration-s',type=float,default=20.);p.add_argument('--resolution-ms',type=float,default=.1);p.add_argument('--nodes',type=int,default=4);p.add_argument('--threads',type=int,default=1);p.add_argument('--repeats',type=int,default=3);p.add_argument('--validation-s',type=float,default=20.);p.add_argument('--output',type=Path,default=HERE/'results'/'nest_benchmark.json');a=p.parse_args()
    if a.duration_s<=0 or a.validation_s<=0 or a.resolution_ms<=0 or min(a.nodes,a.threads,a.repeats)<1:p.error('positive durations, resolution, node/thread and repeat counts required')
    if min(a.duration_s,a.validation_s)*1000/a.resolution_ms<10:p.error('each simulation must contain at least 10 resolution steps')
    for value in (a.duration_s*1000,a.validation_s*1000):
        if abs(value/a.resolution_ms-round(value/a.resolution_ms))>1e-7:p.error('durations must be integer multiples of resolution')
    if nest.__version__.split('.')[:2]!=['3','10']:raise RuntimeError('This module is adapted to NEST 3.10')
    manifest=json.loads((HERE/'module'/'generated'/'manifest.json').read_text());validation={};approximation={};traces={}
    for case in ('am','fm'):
        validation[case],pwa_trace=validate_candidate(case,manifest,a.validation_s*1000,a.resolution_ms)
        print('VALIDATION',case,json.dumps(validation[case]),flush=True)
        _,lr_trace,_=run('astrocyte_lr_1994',case,a.validation_s*1000,a.resolution_ms,1,1,True,True)
        if not np.array_equal(lr_trace['times'],pwa_trace['times']) or not np.array_equal(lr_trace['sic_times'],pwa_trace['sic_times']):
            raise RuntimeError('LR and PWA recording grids differ')
        errors={}
        for key in ('Ca_astro','h_IP3R','IP3','I_SIC'):
            delta=pwa_trace[key]-lr_trace[key]
            errors[key]=dict(rmse=float(np.sqrt(np.mean(delta**2))),max_abs=float(np.max(abs(delta))))
        approximation[case]=dict(errors=errors,scope='One driven protocol; baseline NEST uses its built-in GSL tolerances')
        for model_name,events in [('pwa',pwa_trace),('lr',lr_trace)]:
            for key,values in events.items():traces[f'{case}_{model_name}_{key}']=values
        print('LR COMPARISON',case,json.dumps(errors),flush=True)
    report=[]
    for connected in (False,True):
      for case in ('am','fm'):
        for model in ('astrocyte_lr_1994',f'astrocyte_{case}_pwa'):
          timings=[];proto=None
          for repeat in range(a.repeats):
            t,_,proto=run(model,case,a.duration_s*1000,a.resolution_ms,a.threads,a.nodes,connected,False)
            timings.append(t);print(f'TIMING case={case} model={model} sic={connected} repeat={repeat+1}/{a.repeats} seconds={t:.6g}',flush=True)
          report.append(dict(protocol=proto,seconds=timings,median_seconds=statistics.median(timings)))
    speeds=[]
    for case in ('am','fm'):
      for connected in (False,True):
        lr=next(x['median_seconds'] for x in report if x['protocol']['case']==case and x['protocol']['connected_sic']==connected and x['protocol']['model']=='astrocyte_lr_1994')
        pwa=next(x['median_seconds'] for x in report if x['protocol']['case']==case and x['protocol']['connected_sic']==connected and x['protocol']['model']!= 'astrocyte_lr_1994')
        speeds.append(dict(case=case,connected_sic=connected,LR_seconds=lr,PWA_seconds=pwa,speedup=lr/pwa))
    output=dict(generated_utc=datetime.now(timezone.utc).isoformat(),nest_version=nest.__version__,platform=platform.platform(),logical_cpu_count=os.cpu_count(),candidate_manifest=manifest,validation=validation,approximation_vs_LR=approximation,timings=report,speedups=speeds,caveats=['Local numerical verification covers one finite protocol; it does not establish approximation quality against LR.', 'CurrentEvent inputs are unsupported by the PWA adapter.', 'Finite step threshold crossing bracketing may miss crossings.', 'Wall time covers nest.Simulate, including NEST preparation/finalization; model build and network creation are excluded.'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    trace_path=a.output.with_name(a.output.stem+'_traces.npz');np.savez_compressed(trace_path,**traces);output['trace_file']=str(trace_path)
    tmp=a.output.with_suffix('.tmp');tmp.write_text(json.dumps(output,indent=2,allow_nan=False));tmp.replace(a.output)
    print('SPEEDUPS',json.dumps(speeds,indent=2),flush=True);print(f'Saved {a.output}',flush=True)

if __name__=='__main__':main()
