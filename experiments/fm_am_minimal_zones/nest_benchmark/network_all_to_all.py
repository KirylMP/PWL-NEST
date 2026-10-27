#!/usr/bin/env python3
"""All-excitatory network with one independent astrocyte per directed edge."""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nest
import numpy as np
from benchmark_nest import HERE, pwa_initial, work_initial


def fingerprint(source,target):
    d=nest.GetConnections(source=source,target=target).get(['source','target'])
    pairs=np.asarray([d['source'],d['target']],dtype=np.int64).T
    if len(pairs):pairs=pairs[np.lexsort((pairs[:,1],pairs[:,0]))]
    return dict(count=len(pairs),sha256=hashlib.sha256(pairs.tobytes()).hexdigest())


def run(case,kind,a,path):
    nest.ResetKernel();nest.SetKernelStatus(dict(resolution=a.dt,local_num_threads=a.threads,rng_seed=a.seed,print_time=False))
    nest.Install('pwa_astrocyte_module')
    baseline=.2 if case=='am' else .45
    c,h=work_initial(case,baseline) if kind=='lr' else pwa_initial(case,baseline)
    params=dict(IP3_0=baseline,IP3=baseline,tau_IP3=20000.,delta_IP3=a.delta_ip3,Ca_astro=c,h_IP3R=h)
    if kind=='lr':params['Km_SERCA']=.1 if case=='am' else .05
    t0=time.perf_counter()
    neurons=nest.Create('aeif_cond_alpha_astro',a.neurons,params=dict(tau_syn_ex=2.,tau_syn_in=4.))
    pairs=[(i,j) for i in range(a.neurons) for j in range(a.neurons) if i!=j]
    astro=nest.Create('astrocyte_lr_1994' if kind=='lr' else f'astrocyte_{case}_pwa',len(pairs),params=params)
    noise=nest.Create('poisson_generator',params={'rate':a.poisson_rate})
    nest.Connect(noise,neurons,syn_spec={'weight':1.})
    syn=dict(synapse_model='tsodyks_synapse',weight=a.weight,tau_psc=2.,delay=2.)
    for k,(i,j) in enumerate(pairs):
        nest.Connect(neurons[i:i+1],neurons[j:j+1],syn_spec=syn)
        nest.Connect(neurons[i:i+1],astro[k:k+1],syn_spec=syn)
        nest.Connect(astro[k:k+1],neurons[j:j+1],syn_spec={'synapse_model':'sic_connection','weight':.05})
    spikes=nest.Create('spike_recorder');nest.Connect(neurons,spikes)
    mm_astro=nest.Create('multimeter',params={'interval':a.record_interval,'record_from':['IP3','Ca_astro','h_IP3R']})
    mm_sic=nest.Create('multimeter',params={'interval':a.record_interval,'record_from':['I_SIC']})
    # One representative astrocyte per presynaptic neuron.
    sampled=nest.NodeCollection(astro.tolist()[::a.neurons-1])
    nest.Connect(mm_astro,sampled);nest.Connect(mm_sic,neurons)
    graph={'neuron_to_neuron':fingerprint(neurons,neurons),'neuron_to_astro':fingerprint(neurons,astro),'astro_to_neuron':fingerprint(astro,neurons)}
    if any(g['count']!=len(pairs) for g in graph.values()):raise RuntimeError(f'Incorrect connection counts: {graph}')
    build=time.perf_counter()-t0
    start=time.perf_counter();nest.Prepare();prepare=time.perf_counter()-start
    try:
        start=time.perf_counter();nest.Run(a.duration*1000.);elapsed=time.perf_counter()-start
    finally:
        start=time.perf_counter();nest.Cleanup();cleanup=time.perf_counter()-start
    events={}
    for name,device,keys in [('astro',mm_astro,['times','senders','IP3','Ca_astro','h_IP3R']),('sic',mm_sic,['times','senders','I_SIC']),('spikes',spikes,['times','senders'])]:
        ev=device.get('events')
        for key in keys:events[f'{name}_{key}']=np.asarray(ev[key])
    if path is not None:np.savez_compressed(path,**events)
    rate=len(events['spikes_times'])/(a.neurons*a.duration)
    sic_max=float(np.max(np.abs(events['sic_I_SIC'])))
    notices=[]
    if rate==0:notices.append('No recorded neuronal spikes.')
    if np.ptp(events['astro_IP3'])<1e-8:notices.append('No detectable IP3 change in sampled astrocytes.')
    if sic_max==0:notices.append('No SIC observed in recorded neurons.')
    return dict(case=case,model=kind,graph=graph,initial_C=c,initial_h=h,baseline_ip3=baseline,
        times_seconds=dict(build=build,prepare=prepare,run=elapsed,cleanup=cleanup),
        firing_rate_hz=rate,astro_C_range=[float(np.min(events['astro_Ca_astro'])),float(np.max(events['astro_Ca_astro']))],
        astro_IP3_range=[float(np.min(events['astro_IP3'])),float(np.max(events['astro_IP3']))],sic_max=sic_max,notices=notices),events


def plot(case,traces,path):
    fig,ax=plt.subplots(4,1,figsize=(11,10),sharex=True)
    for kind,color in [('lr','tab:purple'),('pwa','black')]:
        d=traces[kind];mask=d['astro_senders']==np.min(d['astro_senders']);t=d['astro_times'][mask]/1000
        ax[0].plot(t,d['astro_Ca_astro'][mask],color=color,label=kind.upper())
        ax[1].plot(t,d['astro_IP3'][mask],color=color,label=kind.upper())
        sm=d['sic_senders']==np.min(d['sic_senders']);ax[2].plot(d['sic_times'][sm]/1000,d['sic_I_SIC'][sm],color=color,label=kind.upper())
        ax[3].scatter(d['spikes_times']/1000,d['spikes_senders'],s=.4,alpha=.35,color=color,label=kind.upper())
    for a,label in zip(ax,['C (µM)','IP3 (µM)','SIC (pA)','Neuron ID']):a.set_ylabel(label);a.legend(loc='upper right')
    ax[-1].set_xlabel('Time (s)');fig.suptitle(f'{case.upper()}: all-to-all network');fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',choices=['am','fm','both'],default='both');p.add_argument('--neurons',type=int,default=10)
    p.add_argument('--duration',type=float,default=60.);p.add_argument('--dt',type=float,default=.1);p.add_argument('--threads',type=int,default=16)
    p.add_argument('--seed',type=int,default=100);p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--delta-ip3',type=float,default=.008);p.add_argument('--poisson-rate',type=float,default=2000.);p.add_argument('--weight',type=float,default=1.)
    p.add_argument('--record-interval',type=float,default=10.)
    a=p.parse_args()
    if a.neurons<2 or min(a.threads,a.repeats)<=0 or min(a.duration,a.dt,a.record_interval)<=0 or a.delta_ip3<0 or a.poisson_rate<0 or a.weight<=0:p.error('Invalid configuration')
    if a.duration*1000<a.record_interval or any(abs(v/a.dt-round(v/a.dt))>1e-7 for v in (a.duration*1000,a.record_interval,2.)):p.error('Duration, recording interval, and 2 ms delay must be integer multiples of dt')
    a.output.mkdir(parents=True,exist_ok=True);manifest=json.loads((HERE/'module/generated/manifest.json').read_text())
    for case in ('am','fm'):
        d=manifest[case]
        if hashlib.sha256(Path(d['path']).read_bytes()).hexdigest()!=d['sha256']:raise ValueError(f'{case}: candidate changed after build')
    summary=dict(settings=vars(a)|{'output':str(a.output)},nest_version=nest.__version__,candidate_manifest=manifest,
                 topology='all-to-all directed, no autapses, one independent astrocyte per connection',results={})
    for case in (['am','fm'] if a.case=='both' else [a.case]):
        runs={'lr':[],'pwa':[]};traces={};ref=None
        for n in range(a.repeats):
            for kind in (['lr','pwa'] if n%2==0 else ['pwa','lr']):
                print(f'NETWORK start case={case} model={kind} repeat={n+1}/{a.repeats}',flush=True)
                m,ev=run(case,kind,a,a.output/f'{case}_{kind}_traces.npz' if n==0 else None)
                if ref is None:ref=m['graph']
                elif m['graph']!=ref:raise RuntimeError('Connectivity changed; refusing timing comparison')
                runs[kind].append(m);(a.output/f'{case}_{kind}_repeat{n+1}.json').write_text(json.dumps(m,indent=2)+'\n')
                if n==0:traces[kind]=ev
                print(f'NETWORK done case={case} model={kind} time={m["times_seconds"]["run"]:.3f}s rate={m["firing_rate_hz"]:.3f}Hz IP3={m["astro_IP3_range"]} SICmax={m["sic_max"]:.5g}',flush=True)
                for notice in m['notices']:print('NETWORK NOTE',notice,flush=True)
        lr=statistics.median(x['times_seconds']['run'] for x in runs['lr']);pw=statistics.median(x['times_seconds']['run'] for x in runs['pwa'])
        summary['results'][case]=dict(lr=runs['lr'],pwa=runs['pwa'],median_lr_s=lr,median_pwa_s=pw,speedup=lr/pw,
            note='One seed and identical checked connectivity; closed-loop spike histories can diverge.')
        plot(case,traces,a.output/f'{case}_network.png');(a.output/'network_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(f'NETWORK RESULT {case}: LR={lr:.3f}s PWA={pw:.3f}s speedup={lr/pw:.3f}',flush=True)


if __name__=='__main__':main()
