"""Unmodified small-network input with explicit outer-region IP3 extrapolation."""
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from pwl_nest.models import ROOT
from pwl_nest.nest_adapter import load_candidate
from validate_network_replacement import run_one, VARIANTS


def frozen_equilibria(report,traces):
    for name,path in VARIANTS.items():
        if path is None or name not in traces: continue
        model,_=load_candidate('AM', ROOT/path)
        # Coefficient ordering: [intercept, u, v, IP3] in each row.
        q=model.geometry_coefficients().reshape(-1,2,4)[-1]
        a=q[:,1:3];ip=float(traces[name]['astro_IP3'][-1])
        eq=np.linalg.solve(a,-q[:,0]-q[:,3]*ip)
        eigen=np.linalg.eigvals(a)
        try:
            physical=model.to_physical(np.r_[eq,ip]).tolist()
        except ValueError:
            physical=None  # Python coordinate rejects h outside (0,1).
        report['results'][name]['outer_region_at_frozen_terminal_IP3']=dict(
            IP3=ip, eigenvalue_real=eigen.real.tolist(), eigenvalue_imag=eigen.imag.tolist(),
            latent_equilibrium=eq.tolist(), physical_equilibrium=physical,
            self_consistent_C_region=bool(eq[0]>model.latent.thresholds[-1]))


def main():
    out=ROOT/'results/network_extrapolation';out.mkdir(parents=True,exist_ok=True)
    condition=dict(duration_ms=30000., delta_IP3=.2, record_ms=10.,
                   allow_ip3_extrapolation=True)
    report=dict(condition=condition, seed=20261029, resolution_ms=.1,
                model_sha256={k:hashlib.sha256((ROOT/p).read_bytes()).hexdigest()
                              for k,p in VARIANTS.items() if p is not None}, results={})
    traces={}
    for name in VARIANTS:
        result,data=run_one(name,condition)
        report['results'][name]=result
        if data is not None:
            traces[name]=data
            np.savez_compressed(out/f'{name}.npz',**data)
            result['physical_recorded_states']=bool(
                np.all(np.isfinite(data['astro_Ca_astro'])) and
                np.all((data['astro_Ca_astro']>=0)&(data['astro_Ca_astro']<=2/1.185)) and
                np.all((data['astro_h_IP3R']>=0)&(data['astro_h_IP3R']<=1)))
            for field in ['astro_Ca_astro','post_I_SIC']:
                # Tail variation distinguishes a stationary level from oscillations/drift.
                mask=data['astro_times' if field.startswith('astro') else 'post_times']>=25000.
                result[field+'_tail_range']=[float(data[field][mask].min()),float(data[field][mask].max())]
        print(name,json.dumps(result),flush=True)
        (out/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    assert len({r['topology_sha256'] for r in report['results'].values()})==1
    truth=traces['LR']
    for name,data in traces.items():
        result=report['results'][name]
        assert np.array_equal(data['pre_spikes_times'],truth['pre_spikes_times'])
        assert np.array_equal(data['pre_spikes_senders'],truth['pre_spikes_senders'])
        result['identical_presynaptic_spikes']=True
        for field in ['astro_Ca_astro','post_I_SIC','population_rate']:
            result[field+'_RMSE_vs_LR']=float(np.sqrt(np.mean((data[field]-truth[field])**2)))
    report['identical_topology']=True
    frozen_equilibria(report,traces)
    (out/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axes=plt.subplots(4,1,figsize=(11,10),sharex=True)
    for name,data in traces.items():
        for ax,field,n in zip(axes[:3],['astro_IP3','astro_Ca_astro','post_I_SIC'],[5,5,10]):
            t=data['astro_times' if field.startswith('astro') else 'post_times']
            ax.plot(np.unique(t)/1000.,data[field].reshape(-1,n).mean(axis=1),label=name)
        axes[3].plot(data['rate_times'],data['population_rate'],alpha=.6,label=name)
    for ax,label in zip(axes,['IP3 (µM)','Ca (µM)','SIC (pA)','Rate (Hz)']):
        ax.set_ylabel(label);ax.grid(alpha=.2)
    axes[0].legend(ncol=3);axes[-1].set_xlabel('Time (s)')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);fig.savefig(out/'comparison.pdf')


if __name__=='__main__': main()
