"""Save independent DOP853 errors for the first 220 s pulse of each NEST model."""
import json
from pathlib import Path

import numpy as np

from benchmark_nest import compare, initial_state, protocols
from pwl_nest.calcium_coordinates import simulate_coordinates
from pwl_nest.models import ROOT
from pwl_nest.nest_adapter import load_candidate
from pwl_nest.protocols import simulate

out=ROOT/'results/nest_benchmark'
report={}
for case in ['AM','FM']:
    model,_=load_candidate(case)
    protocol=protocols(case)[0];name=protocol.pop('name')
    initial=initial_state(case,protocol['baseline'])
    report[case]={}
    for kind in ['lr','pwa']:
        with np.load(out/f'{case}_{name}_{kind}.npz') as data:
            t=data['t'];x=np.column_stack([data[f] for f in ['C','h','IP3']])
        if kind=='lr':
            tr,xr,_=simulate(case,initial,**protocol,dt=.01,rtol=2e-11,atol=2e-13,max_step=.01)
        else:
            tr,xr,_=simulate_coordinates(model,initial,**protocol,dt=.01,rtol=2e-11,atol=2e-13,max_step=.01)
        ref=np.column_stack([np.interp(t,tr,xr[:,d]) for d in range(3)])
        away=np.abs(t-20.)>1e-7
        d=x-ref
        errors=dict(max_abs_C=float(np.max(np.abs(d[:,0]))),max_abs_h=float(np.max(np.abs(d[:,1]))),
                    max_abs_IP3_away_from_jump=float(np.max(np.abs(d[away,2]))),
                    rmse_C=float(np.sqrt(np.mean(d[:,0]**2))))
        assert errors['max_abs_C']<3e-5,errors
        assert errors['max_abs_h']<3e-5,errors
        assert errors['max_abs_IP3_away_from_jump']<1e-8,errors
        report[case][kind]=errors
        print(case,kind,errors,flush=True)
(out/'independent_validation.json').write_text(json.dumps(report,indent=2)+'\n')
