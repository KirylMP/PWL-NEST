#!/usr/bin/env python3
"""Fixed 100-neuron recurrent NEST network; scale independent astrocytes/neuron."""
import argparse,hashlib,json,os,statistics,sys,time
from pathlib import Path
import numpy as np
import nest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
BENCH=Path(__file__).resolve().parents[3]/'experiments/fm_am_minimal_zones/nest_benchmark';sys.path.insert(0,str(BENCH))
from latent_flow import Candidate

def edges(n,indegree,seed):
 rng=np.random.default_rng(seed);out=[]
 for dst in range(n):
  pre=np.r_[np.arange(dst),np.arange(dst+1,n)];out.extend((int(src),dst) for src in rng.choice(pre,indegree,replace=False))
 return sorted(out)

def run(kind,ratio,a,connection_pairs,output):
 nest.ResetKernel();nest.SetKernelStatus(dict(resolution=a.dt,local_num_threads=a.threads,rng_seed=a.seed,print_time=False));nest.Install('pwa_astrocyte_module')
 t0=time.perf_counter();N=a.neurons
 cells=nest.Create('aeif_cond_alpha_astro',N,params={'tau_syn_ex':2.,'tau_syn_in':4.})
 astro_params=dict(IP3_0=a.ip3,IP3=a.ip3,tau_IP3=20000.,delta_IP3=a.delta_ip3,Ca_astro=.7,h_IP3R=.8)
 if kind=='lr':astro_params['Km_SERCA']=.05
 astro=nest.Create('astrocyte_lr_1994' if kind=='lr' else 'astrocyte_am_pwa',N*ratio,params=astro_params)
 noise=nest.Create('poisson_generator',params={'rate':a.poisson_rate});nest.Connect(noise,cells,syn_spec={'weight':1.})
 syn={'synapse_model':'tsodyks_synapse','weight':a.weight,'tau_psc':2.,'delay':2.}
 for src in range(N):
  targets=sorted(dst for s,dst in connection_pairs if s==src)
  if targets:nest.Connect(cells[src:src+1],nest.NodeCollection([cells.tolist()[j] for j in targets]),syn_spec=syn)
  group=astro[src*ratio:(src+1)*ratio];nest.Connect(cells[src:src+1],group,syn_spec=syn)
  nest.Connect(group,cells[src:src+1],syn_spec={'synapse_model':'sic_connection','weight':a.sic_weight/ratio})
 graph={}
 for name,source,target in [('neuron_neuron',cells,cells),('neuron_astro',cells,astro),('astro_neuron',astro,cells)]:
  conn=nest.GetConnections(source=source,target=target).get(['source','target']);pairs=np.asarray([conn['source'],conn['target']],dtype=np.int64).T
  if len(pairs):pairs=pairs[np.lexsort((pairs[:,1],pairs[:,0]))]
  graph[name]=dict(count=len(pairs),sha256=hashlib.sha256(pairs.tobytes()).hexdigest())
 if graph['neuron_neuron']['count']!=len(connection_pairs) or any(graph[name]['count']!=N*ratio for name in ('neuron_astro','astro_neuron')):raise RuntimeError(f'Unexpected connectivity: {graph}')
 spikes=nest.Create('spike_recorder');nest.Connect(cells,spikes)
 meter=nest.Create('multimeter',params={'interval':a.record_interval,'record_from':['Ca_astro','IP3']});sampled=nest.NodeCollection(astro.tolist()[::max(1,(N*ratio)//min(8,N*ratio))][:8]);nest.Connect(meter,sampled)
 sic=nest.Create('multimeter',params={'interval':a.record_interval,'record_from':['I_SIC']});nest.Connect(sic,cells[:min(N,8)])
 build=time.perf_counter()-t0;t=time.perf_counter();nest.Prepare();prepare=time.perf_counter()-t
 try:
  t=time.perf_counter();nest.Run(a.duration*1000.);wall=time.perf_counter()-t
 finally:nest.Cleanup()
 ev=meter.get('events');sp=spikes.get('events');si=sic.get('events');ids=np.asarray(ev['senders']);c=np.asarray(ev['Ca_astro']);ip3=np.asarray(ev['IP3']);times=np.asarray(ev['times']);mask=ids==ids.min();trace=c[mask];peaks,_=find_peaks(trace,prominence=.02,distance=max(1,int(1000/a.record_interval)))
 if output is not None:np.savez_compressed(output,times=times,senders=ids,calcium=c,ip3=ip3,spike_times=np.asarray(sp['times']),spike_senders=np.asarray(sp['senders']),sic_times=np.asarray(si['times']),sic_senders=np.asarray(si['senders']),sic=np.asarray(si['I_SIC']))
 return dict(kind=kind,ratio=ratio,neurons=N,astrocytes=N*ratio,graph=graph,run_s=wall,build_s=build,prepare_s=prepare,rate_hz=len(sp['times'])/(N*a.duration),sampled_C_range=[float(np.min(c)),float(np.max(c))],sampled_IP3_range=[float(np.min(ip3)),float(np.max(ip3))],first_astro_peaks=int(len(peaks)),first_astro_C_range=[float(np.min(trace)),float(np.max(trace))],max_abs_sic=float(np.max(np.abs(si['I_SIC']))),sic_weight_per_astro=a.sic_weight/ratio)

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--ratios',type=int,nargs='+',default=[1,2,4,8,16]);p.add_argument('--neurons',type=int,default=100);p.add_argument('--indegree',type=int,default=10);p.add_argument('--duration',type=float,default=60);p.add_argument('--dt',type=float,default=.1);p.add_argument('--threads',type=int,default=16);p.add_argument('--repeats',type=int,default=3);p.add_argument('--seed',type=int,default=100);p.add_argument('--record-interval',type=float,default=10);p.add_argument('--ip3',type=float,default=.60);p.add_argument('--delta-ip3',type=float,default=.0002);p.add_argument('--poisson-rate',type=float,default=2000);p.add_argument('--weight',type=float,default=1);p.add_argument('--sic-weight',type=float,default=.05);a=p.parse_args()
 if min(a.ratios)<1 or len(set(a.ratios))!=len(a.ratios) or not 0<a.indegree<a.neurons or min(a.duration,a.dt,a.threads,a.repeats,a.record_interval)<=0:raise ValueError('Invalid parameters')
 if any(abs(v/a.dt-round(v/a.dt))>1e-7 for v in (a.duration*1000,a.record_interval,2.)):raise ValueError('Duration, recording interval and delay must align to dt')
 a.out.mkdir(parents=True,exist_ok=True);manifest=json.loads((BENCH/'module/generated/manifest.json').read_text());d=manifest['am'];path=Path(d['path'])
 if hashlib.sha256(path.read_bytes()).hexdigest()!=d['sha256']:raise RuntimeError('Compiled candidate changed')
 model=Candidate(path)
 if len(model.thresholds)!=2:raise ValueError('Expected two thresholds')
 pairs=edges(a.neurons,a.indegree,a.seed);graph_hash=hashlib.sha256(np.asarray(pairs,dtype=np.int64).tobytes()).hexdigest();result=dict(scope='Finite closed-loop network with 100 fixed neurons and variable independent astrocytes. SIC normalized per target; sampled oscillations not assumed.',settings=vars(a)|{'out':str(a.out)},candidate=d,neuron_edge_count=len(pairs),neuron_edge_sha256=graph_hash,ratios={})
 for ratio in a.ratios:
  measurements={'lr':[],'pwa':[]}
  for i in range(a.repeats):
   for kind in (['lr','pwa'] if i%2==0 else ['pwa','lr']):
    print(f'START ratio={ratio} kind={kind} repeat={i+1}',flush=True);row=run(kind,ratio,a,pairs,a.out/f'ratio_{ratio}_{kind}_traces.npz' if i==0 else None);measurements[kind].append(row);
    if measurements['lr'] and measurements['pwa'] and measurements['lr'][0]['graph']!=measurements['pwa'][0]['graph']:raise RuntimeError('LR/PWA connectivity differs')
    print('DONE',json.dumps(row),flush=True)
  lr=statistics.median(x['run_s'] for x in measurements['lr']);pwa=statistics.median(x['run_s'] for x in measurements['pwa']);result['ratios'][str(ratio)]=dict(lr=measurements['lr'],pwa=measurements['pwa'],lr_median_s=lr,pwa_median_s=pwa,speedup=lr/pwa)
  (a.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
  fig,axs=plt.subplots(3,1,sharex=True,figsize=(9,7),layout='constrained')
  for kind,color in (('lr','tab:purple'),('pwa','black')):
   trace=np.load(a.out/f'ratio_{ratio}_{kind}_traces.npz');mask=trace['senders']==np.min(trace['senders']);t=trace['times'][mask]/1000
   axs[0].plot(t,trace['calcium'][mask],color=color,label=kind.upper());axs[1].plot(t,trace['ip3'][mask],color=color,label=kind.upper());axs[2].scatter(trace['spike_times']/1000,trace['spike_senders'],s=.3,alpha=.3,color=color,label=kind.upper())
  for panel,label in zip(axs,('C (µM)','IP3 (µM)','Neuron ID')):panel.set_ylabel(label);panel.legend(fontsize=7)
  axs[-1].set_xlabel('Time (s)');fig.suptitle(f'Ratio {ratio}:1');fig.savefig(a.out/f'ratio_{ratio}_traces.png',dpi=120);plt.close(fig)
 x=np.array([int(r) for r in result['ratios']]);y=[result['ratios'][str(r)]['speedup'] for r in x];fig,ax=plt.subplots(figsize=(7,4));ax.plot(x,y,'o-');ax.axhline(1,color='gray',lw=.8);ax.set(xlabel='Astrocytes per neuron',ylabel='Network speedup: LR / PWA',title='Fixed recurrent network: 100 neurons');ax.grid(alpha=.25);fig.tight_layout();fig.savefig(a.out/'speedup_vs_ratio.png',dpi=150);plt.close(fig);print('SAVED',a.out/'summary.json',flush=True)
if __name__=='__main__':main()
