"""Audit fold itinerary and physical readout for selected optimized candidates."""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from physical_fold_routes import Route

ROOT=Path(__file__).resolve().parents[1];folder=ROOT/'results/cycle_optimization_v3'
selection=json.loads((folder/'selection.json').read_text());summary={}
for key,sel in selection.items():
 p=folder/sel['directory'];j=json.loads((p/'candidate.json').read_text());old={'physical_thresholds_C':j['cuts'],'alpha':j['n'],'beta':j['h'],'a':j['a'],'b':j['b'],'trace':j['trace']};route=Route(old,'both');t=np.array(j['times']);res,z,q,mats=route.match(t,j['I_fold'],True);samples=[];rows=[]
 for i,(A,r) in enumerate(zip(mats,route.regions)):
  y=np.r_[z[i],q[i],1.];state=np.array([expm(A*s)@y for s in np.linspace(0,t[i],501)]);lower=-np.inf if r==0 else j['cuts'][r-1];upper=np.inf if r==2 else j['cuts'][r];speed0=float((A@y)[0]);speed1=float((A@state[-1])[0]);valid=bool(min(state[:,0])>=lower-1e-8 and max(state[:,0])<=upper+1e-8);samples.extend(state[:,0]);rows.append({'region':r,'valid_sampled':valid,'speed_entry':speed0,'speed_exit':speed1,'u_min':float(min(state[:,0])),'u_max':float(max(state[:,0]))})
 u=np.array(samples);C=np.exp(u) if j['kind']=='log' else u*u;v=json.loads((p/'candidate_verification.json').read_text());summary[key]={'directory':sel['directory'],'I_fold':j['I_fold'],'matching_max':float(max(abs(res))),'logM':v['logM'],'segments':rows,'min_C_sampled':float(min(C)),'max_C_sampled':float(max(C)),'min_u_sampled':float(min(u)),'ivp_max_closure':max(x['closure'] for x in v['IVP']),'ivp_max_abs_logM':max(abs(x['logM']) for x in v['IVP']),'matching_curvature':v['matching_curvature_samples'],'matching_input_derivative':v['matching_input_derivative'],'scope':'501 samples per segment, finite difference diagnostic, independent IVP, not a proof of all crossings'}
 assert all(r['valid_sampled'] for r in rows),key
 print(key,'fold max C',max(C),'closure',summary[key]['ivp_max_closure'])
(folder/'fold_audit.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
