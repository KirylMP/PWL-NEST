"""Network diagnostics and separate repeated timing for one added IP3 hinge."""
import csv
import hashlib
import json
import statistics
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from pwl_nest.models import ROOT
from validate_network_replacement import VARIANTS, run_one

OUT=ROOT/'results/high_ip3_hinge'


def main():
    for name in ['full_AM','AM_2C','AM_2C_2I']:
        VARIANTS[name+'_upper']='models/high_ip3_hinge/'+name+'.json'
    names=list(VARIANTS)
    report=dict(fit=json.loads((OUT/'fit.json').read_text()),models=VARIANTS.copy(),
                seed=20261029,resolution_ms=.1,threads=1,diagnostics={},timing={},
                note='One deterministic feedforward topology; no network metrics used to fit models')
    trace_dir=OUT/'traces';trace_dir.mkdir(exist_ok=True)
    csvrows=[]
    for label,delta in [('original_input',.2),('default_jump',.0002)]:
        condition=dict(duration_ms=30000.,delta_IP3=delta,record_ms=10.,allow_ip3_extrapolation=True)
        rows={};traces={}
        for name in names:
            r,data=run_one(name,condition)
            assert r['completed'],r
            traces[name]=data;rows[name]=r
            np.savez_compressed(trace_dir/f'{label}_{name}.npz',**data)
            print('diagnostic',label,name,flush=True)
        assert len({r['topology_sha256'] for r in rows.values()})==1
        truth=traces['LR'];baseline_count=rows['no_SIC']['post_spikes']
        for name,data in traces.items():
            r=rows[name]
            for key in ['pre_spikes_times','pre_spikes_senders']:
                assert np.array_equal(data[key],truth[key])
            r['identical_presynaptic_spikes']=True
            r['excess_spikes_vs_no_SIC']=r['post_spikes']-baseline_count
            r['effect_fraction_vs_LR']=(r['post_spikes']-baseline_count)/(rows['LR']['post_spikes']-baseline_count)
            cvs=[]
            for node in np.unique(data['post_spikes_senders']):
                isi=np.diff(data['post_spikes_times'][data['post_spikes_senders']==node])
                if len(isi)>1:cvs.append(float(np.std(isi)/np.mean(isi)))
            r['mean_ISI_CV']=float(np.mean(cvs))
            for field,tkey,n in [('astro_Ca_astro','astro_times',5),('astro_h_IP3R','astro_times',5),
                                 ('post_I_SIC','post_times',10)]:
                err=data[field]-truth[field]
                r[field+'_RMSE']=float(np.sqrt(np.mean(err**2)))
                r[field+'_max_abs_error']=float(np.max(np.abs(err)))
                tail=data[tkey]>=25000.
                r[field+'_tail_mean']=float(data[field][tail].mean())
            r['physical_recorded_states']=bool(np.all(np.isfinite(data['astro_Ca_astro'])) and
                 np.all((data['astro_Ca_astro']>=0)&(data['astro_Ca_astro']<=2/1.185)) and
                 np.all((data['astro_h_IP3R']>=0)&(data['astro_h_IP3R']<=1)))
            for field in ['post_I_SIC','astro_Ca_astro']:
                n=10 if field.startswith('post') else 5
                avg=data[field].reshape(-1,n).mean(axis=1)
                t=np.unique(data['post_times' if n==10 else 'astro_times'])/1000.
                r[field+'_area']=float(np.trapezoid(avg,t))
                r[field+'_peak_times']=t[find_peaks(avg,prominence=1. if n==10 else .02)[0]].tolist()
            for binsize in [100.,1000.]:
                edges=np.arange(0,30000.+binsize,binsize)
                x=np.histogram(data['post_spikes_times'],edges)[0]/(10*binsize/1000)
                y=np.histogram(truth['post_spikes_times'],edges)[0]/(10*binsize/1000)
                r[f'population_rate_RMSE_{int(binsize)}ms']=float(np.sqrt(np.mean((x-y)**2)))
            csvrows.append(dict(condition=label,model=name,**{k:v for k,v in r.items() if isinstance(v,(int,float,bool))}))
        report['diagnostics'][label]=dict(condition=condition,results=rows,identical_topology=True)
        fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
        for name,data in traces.items():
            if name not in ['LR','no_SIC','AM_2C','AM_2C_upper','full_AM_upper','AM_2C_2I_upper']:continue
            for ax,field,n in zip(axes[:2],['astro_Ca_astro','post_I_SIC'],[5,10]):
                t=np.unique(data['astro_times' if n==5 else 'post_times'])/1000.
                ax.plot(t,data[field].reshape(-1,n).mean(axis=1),label=name)
            counts=np.histogram(data['post_spikes_times'],np.arange(0,31000.,1000.))[0]/10.
            axes[2].plot(np.arange(30)+.5,counts,label=name)
        for ax,lab in zip(axes,['Ca (µM)','Delivered SIC (pA)','Rate (Hz, 1 s bins)']):
            ax.set_ylabel(lab);ax.grid(alpha=.2)
        axes[0].legend(ncol=3);axes[-1].set_xlabel('Time (s)')
        fig.tight_layout();fig.savefig(OUT/f'{label}.png',dpi=170);fig.savefig(OUT/f'{label}.pdf');plt.close(fig)
        (OUT/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    keys=sorted(set().union(*(r.keys() for r in csvrows)))
    with (OUT/'metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,keys);writer.writeheader();writer.writerows(csvrows)
    # Same 30-s network, no multimeters or spike recorders, same single thread.
    condition=report['diagnostics']['original_input']['condition']
    report['timing_protocol']=dict(condition=condition,record=False,excluded_warmups=1,
                                  repetitions=7,order_seed=20261030,
                                  scope='nest.Run only; build/Prepare/Cleanup saved separately')
    for name in names:
        r,_=run_one(name,condition,record=False)
        assert r['completed'];report['timing'][name]=dict(warmup=r,samples=[])
    rng=np.random.default_rng(20261030)
    for repeat in range(7):
        order=rng.permutation(names).tolist()
        for name in order:
            r,_=run_one(name,condition,record=False)
            assert r['completed']
            r['repeat']=repeat;r['order']=order
            report['timing'][name]['samples'].append(r)
        print('timing repetition',repeat+1,flush=True)
        (OUT/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    for name,r in report['timing'].items():
        values=[s['timing']['run_seconds'] for s in r['samples']]
        r['median_seconds']=statistics.median(values)
        r['min_seconds']=min(values);r['max_seconds']=max(values)
        r['realtime_factor']=30/r['median_seconds']
    lr=report['timing']['LR']['median_seconds']
    for r in report['timing'].values():r['speedup_vs_LR']=lr/r['median_seconds']
    report['source_sha256']={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
          ['scripts/fit_high_ip3_hinge.py','scripts/evaluate_high_ip3_hinge.py',
           'scripts/validate_network_replacement.py','native/nest_pwa_module.cpp',
           'native/pwa_step.h','native/pwa_event.cpp',
           'src/pwl_nest/calcium_coordinates.py','src/pwl_nest/nest_adapter.py']}
    (OUT/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
