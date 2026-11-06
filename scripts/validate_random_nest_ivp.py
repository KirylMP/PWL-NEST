"""Compare recorded PWA NEST calcium against independent adaptive integration."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from fit_lower_fixed_fold import rhs

ROOT=Path(__file__).resolve().parents[1]
output=[]
for case in ('AM','AFM','FM'):
    path=ROOT/'paper/data/random_selected'/case/'candidate.json';raw=path.read_bytes();model=json.loads(raw)
    f=ROOT/f'results/two_inputs_{case}/runs.jsonl';rows=[json.loads(x) for x in f.read_text().splitlines()]
    for I in sorted({x['I'] for x in rows}):
        row=next(x for x in rows if x['I']==I and x.get('diagnostics'))
        if row['candidate_sha256']!=hashlib.sha256(raw).hexdigest():raise ValueError('candidate hash mismatch')
        trace=row['diagnostics']['PWA']['trace'];t=np.asarray(trace['time_s']);C=np.asarray(trace['C'])
        solution=solve_ivp(lambda t,y:rhs(model,I,y),(0,t[-1]),model['rest_target'],method='DOP853',t_eval=t,rtol=1e-10,atol=1e-12,max_step=.03)
        if not solution.success:raise RuntimeError(solution.message)
        err=np.abs(solution.y[0]-C)
        data={'case':case,'I':I,'candidate_sha256':row['candidate_sha256'],'samples':len(t),
              'max_abs_C':float(max(err)),'rmse_C':float(np.sqrt(np.mean(err**2))), 'last_abs_C':float(err[-1])}
        print(data,flush=True);output.append(data)
path=ROOT/'paper/data/random_selected/nest_ivp_check.json'
path.write_text(json.dumps({'scope':'PWA calcium only; recorded samples; finite comparison; no LR numerical accuracy claim',
    'method':'scipy.solve_ivp DOP853','rtol':1e-10,'atol':1e-12,'max_step_s':.03,'results':output},indent=2)+'\n')
