"""Compare NEST LR calcium with independent adaptive integration at recorded times."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from random_selected_variable_input import lr_rhs,KER

ROOT=Path(__file__).resolve().parents[1];out=[]
for case in ('AM','AFM','FM'):
    raw=(ROOT/'paper/data/random_selected'/case/'candidate.json').read_bytes();model=json.loads(raw)
    rows=[json.loads(x) for x in (ROOT/f'results/two_inputs_{case}/runs.jsonl').read_text().splitlines()]
    for I in sorted({x['I'] for x in rows}):
        row=next(x for x in rows if x['I']==I and x.get('diagnostics'))
        if row['candidate_sha256']!=hashlib.sha256(raw).hexdigest():raise ValueError('candidate hash mismatch')
        trace=row['diagnostics']['LR']['trace'];t=np.array(trace['time_s']);C=np.array(trace['C'])
        sol=solve_ivp(lambda t,y:lr_rhs(I,y,KER[case]),(0,t[-1]),model['rest_target'],method='DOP853',t_eval=t,rtol=1e-11,atol=1e-13,max_step=.03)
        if not sol.success:raise RuntimeError(sol.message)
        err=np.abs(sol.y[0]-C);item={'case':case,'I':I,'samples':len(t),'max_abs_C':float(max(err)),'rmse_C':float(np.sqrt(np.mean(err**2)))}
        print(item,flush=True);out.append(item)
(ROOT/'paper/data/random_selected/nest_lr_ivp_check.json').write_text(json.dumps({'method':'DOP853','rtol':1e-11,'atol':1e-13,'max_step_s':.03,'scope':'comparison at recorded samples, six fixed inputs; not an error bound','results':out},indent=2)+'\n')
