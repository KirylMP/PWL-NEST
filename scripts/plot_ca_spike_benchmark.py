"""Review matched-condition SIC and calcium-spike NEST benchmarks."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
meta=json.loads((ROOT/'paper/data/random_selected/manifest.json').read_text())['models']
settings={'AM':(.2,.5),'AFM':(.2,.6),'FM':(.2,.7)}
records=[];lags={}
for case,inputs in settings.items():
    datasets={}
    for mode,prefix in [('sic','two_inputs'),('events','ca_events')]:
        folder=ROOT/f'results/{prefix}_{case}';raw=(folder/'runs.jsonl').read_bytes()
        rows=[json.loads(z) for z in raw.splitlines()];manifest=json.loads((folder/'manifest.json').read_text())
        if any(r['candidate_sha256']!=meta[case]['candidate_sha256'] for r in rows):raise ValueError('candidate hash mismatch')
        datasets[mode]=(rows,manifest,hashlib.sha256(raw).hexdigest())
    for I in inputs:
        row={'case':case,'I':I,'modes':{}}
        traces={}
        for mode,(rows,manifest,digest) in datasets.items():
            samples=[r for r in rows if r['I']==I and not r['warmup']]
            diagnostics=next(r['diagnostics'] for r in rows if r['I']==I and r.get('diagnostics'))
            rec={'runs_sha256':digest,'module_sha256':manifest['module_sha256'],'models':{}}
            for model in ('LR','PWA'):
                runs=[r['result'] for r in samples if r['model']==model]
                if len(runs)!=3 or any('error' in r for r in runs):raise ValueError('missing timed run')
                times=[r['run_s'] for r in runs];d=diagnostics[model]
                if mode=='events' and any(r['ca_events_recorded']!=r['ca_events_emitted'] for r in runs):raise ValueError('count mismatch')
                rec['models'][model]={'times':times,'median':float(np.median(times)),
                    'diagnostic':d['dynamics'],'events_s':d.get('event_times_s'),
                    'recorded_events':[v['ca_events_recorded'] for v in runs] if mode=='events' else None}
                traces[(mode,model)]=d['trace']
            rec['speedup']=rec['models']['LR']['median']/rec['models']['PWA']['median']
            row['modes'][mode]=rec
        row['max_trace_difference']={m:float(np.max(np.abs(np.array(traces[('sic',m)]['C'])-np.array(traces[('events',m)]['C'])))) for m in ('LR','PWA')}
        records.append(row)
        if I==inputs[1]:
            a=row['modes']['events']['models'];lr=np.asarray(a['LR']['events_s']);pwa=np.asarray(a['PWA']['events_s']);n=min(len(lr),len(pwa));lags[case]={'lr':lr.tolist(),'pwa':pwa.tolist(),'ordinal_delta':(pwa[:n]-lr[:n]).tolist()}

fig,(a,b,c)=plt.subplots(3,1,figsize=(10.5,9),constrained_layout=True,gridspec_kw={'height_ratios':[1.2,1,1]})
x=np.arange(len(records));width=.18
for j,(mode,model,color,label) in enumerate([('sic','LR','#555555','LR, SIC'),('sic','PWA','#c98989','PWA, SIC'),('events','LR','#1f517e','LR, spikes'),('events','PWA','#c52f35','PWA, spikes')]):
    vals=[r['modes'][mode]['models'][model]['median'] for r in records];pos=x+(j-1.5)*width
    a.bar(pos,vals,width,color=color,label=label,alpha=.85)
    for i,r in enumerate(records):a.scatter(pos[i]+np.array([-.035,0,.035]),r['modes'][mode]['models'][model]['times'],s=6,c='black',zorder=4)
a.set_ylabel('Run time (s)');a.legend(ncol=4,fontsize=8,frameon=False,loc='lower center',bbox_to_anchor=(.5,1.01));a.grid(axis='y',alpha=.2)
labels=[f"{r['case']}\n{r['I']:g}" for r in records];a.set_xticks(x,labels);a.set_xlabel('Regime and IP₃ (µM)')
for j,(mode,color,label) in enumerate([('sic','#999999','SIC'),('events','#ba3038','spikes')]):
    b.bar(x+(j-.5)*.34,[r['modes'][mode]['speedup'] for r in records],.34,color=color,label=label)
b.set_xticks(x,labels);b.set_ylabel('LR / PWA');b.grid(axis='y',alpha=.2);b.legend(frameon=False,ncol=2,fontsize=8,loc='lower center',bbox_to_anchor=(.5,1.01));b.set_xlabel('Regime and IP₃ (µM)')
for case,color in [('AM','#175a91'),('AFM','#be4a22'),('FM','#7c4c91')]:
    delta=lags[case]['ordinal_delta'];c.plot(np.arange(1,len(delta)+1),delta,'o-',ms=3,label=case,color=color)
c.axhline(0,color='black',lw=.7);c.set_xlabel('Ordinal event number');c.set_ylabel('PWA − LR time (s)');c.grid(alpha=.2);c.legend(frameon=False,ncol=3,fontsize=8)
out=ROOT/'paper/figures/ca_spike_benchmark';fig.savefig(str(out)+'.pdf',bbox_inches='tight');fig.savefig(str(out)+'.png',dpi=200,bbox_inches='tight')
(ROOT/'paper/data/random_selected/ca_spike_benchmark.json').write_text(json.dumps({'records':records,'ordinal_lags':lags,'scope':'finite diagnostic trajectories; ordinal event pairing; no matched output intervention'},indent=2,allow_nan=False)+'\n')
print(out)
