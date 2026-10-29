#!/usr/bin/env python3
"""Check compiled NEST PWA transition against DOP853, including free q."""
import json,os,sys
from pathlib import Path
import numpy as np
import nest
from scipy.integrate import solve_ivp
BENCH=Path(__file__).resolve().parents[3]/'experiments/fm_am_minimal_zones/nest_benchmark';sys.path.insert(0,str(BENCH))
from latent_flow import Candidate
from benchmark_nest import pwa_initial

def main():
 manifest=json.loads((BENCH/'module/generated/manifest.json').read_text());model=Candidate(manifest['am']['path']);I=float(os.environ['PAPER_BASELINE_IP3']);reports=[]
 c0,q0=pwa_initial('am',I);seeds=[(c0,q0,'equilibrium')]
 for c,z in zip(model.thresholds,model.z_thresholds):
  n=float(np.r_[1.,z,I,np.maximum(z-model.z_thresholds,0.)]@model.alpha)
  seeds.extend([(float(c),n-.05,'cut_down'),(float(c),n+.05,'cut_up')])
 for c,q,label in seeds:
  nest.ResetKernel();nest.SetKernelStatus({'resolution':1.,'local_num_threads':1,'print_time':False});nest.Install('pwa_astrocyte_module')
  node=nest.Create('astrocyte_am_pwa',params={'Ca_astro':c,'h_IP3R':q,'IP3':I,'IP3_0':I,'tau_IP3':20000.});node.set({'delta_IP3':.008});nest.Simulate(1.)
  got=np.array([node.get(k) for k in ('Ca_astro','h_IP3R','IP3')]);z=float(model.phi(c));sol=solve_ivp(lambda t,y:model.rhs(t,y,I,20.),(0,.001),[z,q,I],method='DOP853',rtol=1e-12,atol=1e-13,max_step=.00002)
  if not sol.success:raise RuntimeError(sol.message)
  ref=sol.y[:,-1].copy();ref[0]=model.inverse(ref[0]);err=float(np.max(np.abs(got-ref)/np.maximum(1.,np.abs(ref))));reports.append(dict(label=label,C_initial=c,q_initial=q,max_scaled_error=err))
  if err>1e-7:raise RuntimeError(f'NEST field check failed: {reports[-1]}')
 print('NEST_DOP853_CHECK',json.dumps(reports,indent=2),flush=True)
if __name__=='__main__':main()
