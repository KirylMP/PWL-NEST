"""Closed-loop network scaling: one astrocyte per selected directed synapse."""
import argparse, hashlib, json, platform, time
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
from benchmark_constant_ip3 import KER, parameters, load_case

ROOT=Path(__file__).resolve().parents[1]


def graph(neurons, count, seed):
    edges=np.array([(i,j) for i in range(neurons) for j in range(neurons) if i!=j],dtype=np.int64)
    if count<1 or count>len(edges):raise ValueError('astrocytes exceed directed synapses without autapses')
    rng=np.random.default_rng(seed); chosen=rng.permutation(len(edges))[:count]
    return edges,edges[chosen]


def run(case,kind,count,seed,args,diagnostic):
    import nest
    from pwl_nest.nest_adapter import install
    data,_=load_case(case,args.model_set);C,q=map(float,data['rest_target'])
    edges, selected=graph(args.neurons,count,seed)
    nest.ResetKernel();nest.SetKernelStatus({'resolution':args.dt_ms,'local_num_threads':args.threads,'rng_seed':seed,'print_time':False});install()
    start=time.perf_counter()
    neurons=nest.Create('aeif_cond_alpha_astro',args.neurons,params={'tau_syn_ex':2.,'tau_syn_in':4.})
    params=(parameters(data,args.ip3,C,q) if kind=='PWA' else dict(Ca_astro=C,h_IP3R=q,IP3=args.ip3,IP3_0=args.ip3,tau_IP3=20000.,delta_IP3=args.delta_ip3,Km_SERCA=KER[case]))
    if kind=='PWA':params['delta_IP3']=args.delta_ip3
    astro=nest.Create('astrocyte_lr_pwa' if kind=='PWA' else 'astrocyte_lr_1994',count,params=params)
    noise=nest.Create('poisson_generator',params={'rate':args.poisson_rate})
    nest.Connect(noise,neurons,syn_spec={'weight':args.neuron_weight,'delay':args.delay_ms})
    syn={'synapse_model':'static_synapse','weight':args.neuron_weight,'delay':args.delay_ms}
    for pre,post in edges:
        nest.Connect(neurons[int(pre):int(pre)+1],neurons[int(post):int(post)+1],syn_spec=syn)
    for k,(pre,post) in enumerate(selected):
        nest.Connect(neurons[int(pre):int(pre)+1],astro[k:k+1],syn_spec={'weight':1.,'delay':args.delay_ms})
        nest.Connect(astro[k:k+1],neurons[int(post):int(post)+1],syn_spec={'synapse_model':'sic_connection','weight':args.sic_weight,'delay':args.delay_ms})
    meters=[];recorder=None
    if diagnostic:
        sample=min(count,args.sample);m=nest.Create('multimeter',params={'interval':args.record_ms,'record_from':['Ca_astro','IP3']});nest.Connect(m,astro[:sample]);meters.append(m)
        s=nest.Create('multimeter',params={'interval':args.record_ms,'record_from':['I_SIC']});nest.Connect(s,neurons);meters.append(s)
        recorder=nest.Create('spike_recorder');nest.Connect(neurons,recorder)
    build=time.perf_counter()-start;start=time.perf_counter();nest.Prepare();prep=time.perf_counter()-start
    try:
        start=time.perf_counter();nest.Run(args.duration*1000);elapsed=time.perf_counter()-start
    finally:
        start=time.perf_counter();nest.Cleanup();cleanup=time.perf_counter()-start
    result={'build_s':build,'prepare_s':prep,'run_s':elapsed,'cleanup_s':cleanup,'neurons':args.neurons,'synapses':len(edges),'astrocytes':count,
            'selected_edges_sha256':hashlib.sha256(selected.tobytes()).hexdigest()}
    if kind=='PWA': result['crossings']=int(np.sum(astro.get('region_crossings')))
    if diagnostic:
        ev=meters[0].get('events');sic=meters[1].get('events');sp=recorder.get('events');ca=np.asarray(ev['Ca_astro']);ip=np.asarray(ev['IP3']);ids=np.asarray(ev['senders']);t=np.asarray(ev['times'])/1000.
        if len(ca)==0:raise RuntimeError('No diagnostic samples')
        active=0; physical=True
        for node in np.unique(ids):
            z=ca[(ids==node)&(t>=args.duration/2)];peaks,_=find_peaks(z,prominence=.001)
            if len(peaks)>=3:active+=1
            physical &= bool(np.isfinite(z).all() and np.all((z>0)))
        result['diagnostic']={'sampled_astrocytes':len(np.unique(ids)),'at_least_3_peaks':active,
             'observed_C_min':float(ca.min()),'observed_C_max':float(ca.max()),'physical_sampled_C':physical,
             'observed_IP3_min':float(ip.min()),'observed_IP3_max':float(ip.max()),
             'mean_firing_hz':len(sp['times'])/(args.neurons*args.duration),'spike_count':len(sp['times']),
             'max_abs_SIC':float(np.max(np.abs(sic['I_SIC']))) if len(sic['I_SIC']) else None,
             'scope':'finite diagnostic, peaks only in second half; >=3 peaks is not convergence proof'}
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cases',nargs='+',choices=sorted(KER),default=['AM','AFM','FM']);p.add_argument('--neurons',type=int,default=100)
    p.add_argument('--model-set',choices=['selected','no_fold'],default='selected');p.add_argument('--counts',nargs='+',type=int,default=[100,1000,5000,9900]);p.add_argument('--seeds',nargs='+',type=int,default=[101,102,103,104,105]);p.add_argument('--repeats',type=int,default=3);p.add_argument('--warmup',type=int,default=1)
    p.add_argument('--duration',type=float,default=60.);p.add_argument('--dt-ms',type=float,default=.1);p.add_argument('--threads',type=int,default=1);p.add_argument('--record-ms',type=float,default=50.);p.add_argument('--sample',type=int,default=32)
    p.add_argument('--ip3',type=float,default=.2);p.add_argument('--delta-ip3',type=float,default=.008);p.add_argument('--poisson-rate',type=float,default=2000.);p.add_argument('--neuron-weight',type=float,default=1.);p.add_argument('--sic-weight',type=float,default=.05);p.add_argument('--delay-ms',type=float,default=2.);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.neurons<2 or min(a.counts)<1 or max(a.counts)>a.neurons*(a.neurons-1) or min(a.seeds)<1 or min(a.repeats,a.threads,a.sample)<1 or a.warmup<0 or min(a.duration,a.dt_ms,a.record_ms,a.delay_ms)<=0 or a.delay_ms<a.dt_ms or a.delta_ip3<0 or not 0<=a.ip3<=1.3:p.error('invalid configuration')
    if any(abs(x/a.dt_ms-round(x/a.dt_ms))>1e-7 for x in (a.duration*1000,a.record_ms,a.delay_ms)):p.error('duration, recording interval and delay must align to resolution')
    import nest
    module=ROOT/'.build/pwl_astro_module.so'
    if not module.is_file():raise FileNotFoundError(module)
    output=a.output/'runs.jsonl'
    if output.exists():raise FileExistsError(output)
    a.output.mkdir(parents=True,exist_ok=True)
    meta={'settings':vars(a)|{'output':str(a.output)},'nest_version':nest.__version__,'platform':platform.platform(),'module_sha256':hashlib.sha256(module.read_bytes()).hexdigest(),
          'candidate_sha256':{c:load_case(c,a.model_set)[1] for c in a.cases},
          'scope':'all-to-all directed without autapses; fixed SIC weight per astrocyte, overall feedback changes with population; closed-loop spike histories may diverge'}
    (a.output/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    with output.open('x') as out:
      for case in a.cases:
       for count in a.counts:
        for seed in a.seeds:
         for repeat in range(-a.warmup,a.repeats):
          for kind in (('LR','PWA') if repeat%2==0 else ('PWA','LR')):
           print('run',case,count,seed,repeat,kind,flush=True)
           try:r=run(case,kind,count,seed,a,False)
           except Exception as e:r={'error':f'{type(e).__name__}: {e}'}
           out.write(json.dumps({'case':case,'seed':seed,'repeat':repeat,'warmup':repeat<0,'count':count,'kind':kind,'result':r},allow_nan=False)+'\n');out.flush()
         for kind in ('LR','PWA'):
          try:r=run(case,kind,count,seed,a,True)
          except Exception as e:r={'error':f'{type(e).__name__}: {e}'}
          out.write(json.dumps({'case':case,'seed':seed,'repeat':None,'warmup':False,'count':count,'kind':kind,'diagnostic':True,'result':r},allow_nan=False)+'\n');out.flush()
    print('Saved',output)


if __name__=='__main__':main()
