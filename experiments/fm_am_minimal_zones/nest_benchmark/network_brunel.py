#!/usr/bin/env python3
"""Scaled NEST astrocyte Brunel example: compare LR and fitted PWA networks.

Connectivity and conductance parameters follow NEST 3.10's
astrocyte_brunel_fixed_indegree example; population sizes are configurable.
"""
import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nest
import numpy as np
from benchmark_nest import HERE, pwa_initial, work_initial


def topology_hash(exc,inh,neurons,astro):
    result={}
    for label,source,target in [('excitatory',exc,neurons),('inhibitory',inh,neurons),('neuron_to_astro',exc,astro),('astro_to_neuron',astro,neurons)]:
        connections=nest.GetConnections(source=source,target=target)
        data=connections.get(['source','target'])
        pairs=np.asarray([data['source'],data['target']],dtype=np.int64).T
        if len(pairs):pairs=pairs[np.lexsort((pairs[:,1],pairs[:,0]))]
        result[label]=dict(count=len(pairs),sha256=hashlib.sha256(pairs.tobytes()).hexdigest())
    return result


def run(case,kind,args,record,output):
    nest.ResetKernel()
    nest.SetKernelStatus(dict(resolution=args.dt,local_num_threads=args.threads,rng_seed=args.seed,print_time=False))
    nest.Install('pwa_astrocyte_module')
    I=.2 if case=='am' else .45
    c,h=work_initial(case,I) if kind=='lr' else pwa_initial(case,I)
    params=dict(IP3_0=I,IP3=I,tau_IP3=20000.,delta_IP3=args.delta_ip3,Ca_astro=c,h_IP3R=h)
    if kind=='lr':params['Km_SERCA']=.1 if case=='am' else .05
    astro_model='astrocyte_lr_1994' if kind=='lr' else f'astrocyte_{case}_pwa'
    t0=time.perf_counter()
    neuron_params=dict(tau_syn_ex=2.,tau_syn_in=4.)
    exc=nest.Create('aeif_cond_alpha_astro',args.ne,params=neuron_params)
    inh=nest.Create('aeif_cond_alpha_astro',args.ni,params=neuron_params)
    neurons=exc+inh
    astro=nest.Create(astro_model,args.astro,params=params)
    noise=nest.Create('poisson_generator',params={'rate':2000.})
    nest.Connect(noise,neurons,syn_spec={'weight':1.})
    nest.TripartiteConnect(exc,neurons,astro,
        conn_spec={'rule':'fixed_indegree','indegree':args.ce},
        third_factor_conn_spec={'rule':'third_factor_bernoulli_with_pool','p':.5,'pool_size':args.pool,'pool_type':'random'},
        syn_specs={'primary':{'synapse_model':'tsodyks_synapse','weight':1.,'tau_psc':2.,'delay':2.},
                   'third_in':{'synapse_model':'tsodyks_synapse','weight':1.,'tau_psc':2.,'delay':2.},
                   'third_out':{'synapse_model':'sic_connection','weight':.05}})
    nest.Connect(inh,neurons,{'rule':'fixed_indegree','indegree':args.ci},
        {'synapse_model':'tsodyks_synapse','weight':-4.,'tau_psc':4.,'delay':1.})
    spike_ex=nest.Create('spike_recorder');spike_in=nest.Create('spike_recorder')
    n_ex=min(args.record_neurons,args.ne);n_in=min(args.record_neurons,args.ni)
    nest.Connect(exc[:n_ex],spike_ex);nest.Connect(inh[:n_in],spike_in)
    mm_astro=nest.Create('multimeter',params={'interval':args.record_interval,'record_from':['IP3','Ca_astro','h_IP3R']})
    mm_sic=nest.Create('multimeter',params={'interval':args.record_interval,'record_from':['I_SIC']})
    nest.Connect(mm_astro,astro[:min(args.record_astro,args.astro)])
    nest.Connect(mm_sic,exc[:n_ex])
    graph=topology_hash(exc,inh,neurons,astro)
    build_seconds=time.perf_counter()-t0
    start=time.perf_counter();nest.Prepare();prepare=time.perf_counter()-start
    try:
        start=time.perf_counter();nest.Run(args.duration*1000.);elapsed=time.perf_counter()-start
    finally:
        start=time.perf_counter();nest.Cleanup();cleanup=time.perf_counter()-start
    events={}
    for name,device,keys in [('astro',mm_astro,['times','senders','IP3','Ca_astro','h_IP3R']),
                             ('sic',mm_sic,['times','senders','I_SIC']),
                             ('spikes_ex',spike_ex,['times','senders']),('spikes_in',spike_in,['times','senders'])]:
        ev=device.get('events')
        for k in keys:events[f'{name}_{k}']=np.asarray(ev[k])
    rate_ex=len(events['spikes_ex_times'])/(n_ex*args.duration)
    rate_in=len(events['spikes_in_times'])/(n_in*args.duration)
    notices=[]
    if rate_ex==0 or rate_in==0:notices.append('At least one sampled neuron population emitted no spikes.')
    if np.ptp(events['astro_IP3'])<1e-8:notices.append('Recorded astrocytes show no detectable IP3 change.')
    if np.max(np.abs(events['sic_I_SIC']))==0:notices.append('No SIC was observed in the sampled neurons.')
    metrics=dict(case=case,model=kind,initial_C=c,initial_h=h,baseline_ip3=I,graph=graph,
        times_seconds=dict(build=build_seconds,prepare=prepare,run=elapsed,cleanup=cleanup),
        recorded_firing_rate_hz=dict(excitatory=rate_ex,inhibitory=rate_in),
        astro_C_range=[float(np.min(events['astro_Ca_astro'])),float(np.max(events['astro_Ca_astro']))],
        astro_IP3_range=[float(np.min(events['astro_IP3'])),float(np.max(events['astro_IP3']))],
        sic_max=float(np.max(np.abs(events['sic_I_SIC']))),sample_counts=dict(excitatory=n_ex,inhibitory=n_in,astrocytes=min(args.record_astro,args.astro)),notices=notices)
    if record:
        np.savez_compressed(output,**events)
    return metrics,events


def make_plot(case,data,path):
    fig,ax=plt.subplots(4,1,figsize=(11,10),sharex=True)
    for kind,color in [('lr','tab:purple'),('pwa','black')]:
        ev=data[kind];senders=ev['astro_senders'];first=np.min(senders);mask=senders==first
        t=ev['astro_times'][mask]/1000.
        ax[0].plot(t,ev['astro_Ca_astro'][mask],color=color,label=kind.upper())
        ax[1].plot(t,ev['astro_IP3'][mask],color=color,label=kind.upper())
        sicmask=ev['sic_senders']==np.min(ev['sic_senders'])
        ax[2].plot(ev['sic_times'][sicmask]/1000.,ev['sic_I_SIC'][sicmask],color=color,label=kind.upper())
        ax[3].scatter(ev['spikes_ex_times']/1000.,ev['spikes_ex_senders'],s=.3,alpha=.3,color=color,label=kind.upper())
    ax[0].set_ylabel('C (µM)');ax[1].set_ylabel('IP3 (µM)');ax[2].set_ylabel('SIC (pA)');ax[3].set_ylabel('Exc. neuron ID');ax[3].set_xlabel('Time (s)')
    for a in ax:a.legend(loc='upper right')
    fig.suptitle(f'{case.upper()}: NEST Brunel-style network');fig.tight_layout();fig.savefig(path,dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--case',choices=['am','fm','both'],default='both')
    p.add_argument('--duration',type=float,default=10.);p.add_argument('--dt',type=float,default=.1)
    p.add_argument('--threads',type=int,default=16);p.add_argument('--seed',type=int,default=100)
    p.add_argument('--ne',type=int,default=800);p.add_argument('--ni',type=int,default=200);p.add_argument('--astro',type=int,default=1000)
    p.add_argument('--ce',type=int,default=80);p.add_argument('--ci',type=int,default=20);p.add_argument('--pool',type=int,default=10)
    p.add_argument('--delta-ip3',type=float,default=.0002);p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--record-neurons',type=int,default=100);p.add_argument('--record-astro',type=int,default=5);p.add_argument('--record-interval',type=float,default=10.)
    a=p.parse_args()
    if min(a.ne,a.ni,a.astro,a.ce,a.ci,a.pool,a.threads,a.repeats,a.record_neurons,a.record_astro)<=0 or a.ce>a.ne or a.ci>a.ni or a.pool>a.astro or a.duration<=0 or a.dt<=0 or a.record_interval<a.dt or a.delta_ip3<0:p.error('Invalid network or simulation parameters')
    if a.duration*1000<a.record_interval or abs(a.duration*1000/a.dt-round(a.duration*1000/a.dt))>1e-7 or abs(a.record_interval/a.dt-round(a.record_interval/a.dt))>1e-7:p.error('Duration and recording interval must be multiples of dt, with at least one recording')
    if any(abs(delay/a.dt-round(delay/a.dt))>1e-7 for delay in (1.,2.)):p.error('The example delays 1 ms and 2 ms must be multiples of dt')
    a.output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((HERE/'module/generated/manifest.json').read_text())
    for case in ('am','fm'):
        data=manifest[case]
        if hashlib.sha256(Path(data['path']).read_bytes()).hexdigest()!=data['sha256']:raise ValueError(f'{case}: candidate changed since module build')
    summary=dict(settings=vars(a)|{'output':str(a.output)},nest_version=nest.__version__,candidate_manifest=manifest,results={})
    for case in (['am','fm'] if a.case=='both' else [a.case]):
        results={'lr':[],'pwa':[]};traces={};reference_graph=None
        for repeat in range(a.repeats):
            for kind in (['lr','pwa'] if repeat%2==0 else ['pwa','lr']):
                print(f'NETWORK start case={case} model={kind} repeat={repeat+1}/{a.repeats}',flush=True)
                metrics,ev=run(case,kind,a,repeat==0,a.output/f'{case}_{kind}_traces.npz')
                if reference_graph is None:reference_graph=metrics['graph']
                elif metrics['graph']!=reference_graph:raise RuntimeError('Network topology differs between trials; refusing timing comparison')
                results[kind].append(metrics)
                (a.output/f'{case}_{kind}_repeat{repeat+1}.json').write_text(json.dumps(metrics,indent=2)+'\n')
                if repeat==0:traces[kind]=ev
                print(f'NETWORK done case={case} model={kind} elapsed={metrics["times_seconds"]["run"]:.3f}s firing={metrics["recorded_firing_rate_hz"]} IP3={metrics["astro_IP3_range"]}',flush=True)
                for notice in metrics['notices']:print(f'NETWORK NOTE {case} {kind}: {notice}',flush=True)
        lr=statistics.median(q['times_seconds']['run'] for q in results['lr']);pwa=statistics.median(q['times_seconds']['run'] for q in results['pwa'])
        summary['results'][case]=dict(lr=results['lr'],pwa=results['pwa'],median_lr_s=lr,median_pwa_s=pwa,speedup=lr/pwa,
            note='Same seed and checked primary/input graph; closed-loop spike histories can diverge.')
        make_plot(case,traces,a.output/f'{case}_network.png')
        (a.output/'network_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(f'NETWORK RESULT {case}: LR={lr:.3f}s PWA={pwa:.3f}s speedup={lr/pwa:.3f}',flush=True)


if __name__=='__main__':main()
