#!/usr/bin/env python3
"""Check NEST latent state persistence and both directions at thresholds."""
import json
from pathlib import Path
import numpy as np
import nest
from scipy.integrate import solve_ivp
from latent_flow import Candidate
from benchmark_nest import pwa_initial

HERE=Path(__file__).resolve().parent


def init():
    nest.ResetKernel();nest.SetKernelStatus({'resolution':1.,'local_num_threads':1,'print_time':False});nest.Install('pwa_astrocyte_module')


def trajectory(case,split,model):
    init()
    if model.direct_calcium:
        # Start each phenomenological model at its own stable equilibrium.
        # Its hidden coordinate need not equal the gate of Li–Rinzel.
        base=.2 if case=='am' else .45
        calcium,hidden=pwa_initial(case,base)
    else:
        base=.4 if case=='am' else .55
        calcium,hidden=.1,.75
    node=nest.Create(f'astrocyte_{case}_pwa',params={'Ca_astro':calcium,'h_IP3R':hidden,'IP3':base,'IP3_0':base,'tau_IP3':30000.,'delta_IP3':.15})
    source=nest.Create('spike_generator',params={'spike_times':[4000.,9000.,14000.]})
    nest.Connect(source,node,syn_spec={'delay':1.,'weight':1.})
    if split:
        nest.Simulate(5000.)
        before=np.array([node.get(k) for k in ('Ca_astro','h_IP3R','IP3')],float)
        node.set({'delta_IP3':.15})
        after=np.array([node.get(k) for k in ('Ca_astro','h_IP3R','IP3')],float)
        if np.max(abs(before-after))>2e-12:raise RuntimeError('State changed while setting an unrelated parameter')
        nest.Simulate(15000.)
    else:nest.Simulate(20000.)
    return np.array([node.get(k) for k in ('Ca_astro','h_IP3R','IP3')],float)


def boundary_checks(case,path):
    model=Candidate(path);base=.4 if case=='am' else .55;reports=[]
    for cut,z in zip(model.thresholds,model.z_thresholds):
        basis=np.r_[1.,z,base,np.maximum(z-model.z_thresholds,0.)]
        n=float(basis@model.alpha)
        for direction in (-1,1):
            h=n+direction*.05
            if not 0<h<1:
                reports.append(dict(C=float(cut),direction=direction,skipped='no physical h at chosen offset'));continue
            init();node=nest.Create(f'astrocyte_{case}_pwa',params={'Ca_astro':float(cut),'h_IP3R':h,'IP3':base,'IP3_0':base,'tau_IP3':30000.})
            nest.Simulate(1.)
            actual=np.array([node.get(k) for k in ('Ca_astro','h_IP3R','IP3')],float)
            sol=solve_ivp(lambda t,y:model.rhs(t,y,base,30.),(0.,.001),[z,h,base],method='DOP853',rtol=1e-12,atol=1e-14,max_step=.00005)
            if not sol.success:raise RuntimeError(sol.message)
            expected=sol.y[:,-1].copy();expected[0]=model.inverse(expected[0])
            error=float(np.max(abs(actual-expected)));reports.append(dict(C=float(cut),direction=direction,max_abs_error=error))
            if error>2e-9:raise RuntimeError(f'{case} threshold test failed: {reports[-1]}')
    return reports


def main():
    manifest=json.loads((HERE/'module/generated/manifest.json').read_text());out={}
    for case in ('am','fm'):
        model=Candidate(manifest[case]['path'])
        continuous=trajectory(case,False,model);split=trajectory(case,True,model);error=float(np.max(abs(continuous-split)))
        if error>2e-9:raise RuntimeError(f'{case} split Simulate mismatch {error}')
        boundaries=boundary_checks(case,Path(manifest[case]['path']))
        out[case]=dict(split_simulate_max_abs_error=error,threshold_checks=boundaries)
    print('STATE CHECKS',json.dumps(out,indent=2),flush=True)

if __name__=='__main__':main()
