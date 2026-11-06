"""Sensitivity of AM 465 physicality to finite input transition times."""
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from fit_lower_fixed_fold import rhs

ROOT=Path(__file__).resolve().parents[1];model=json.loads((ROOT/'paper/data/random_selected/AM/candidate.json').read_text());out=[]
for width in (1.,5.,10.):
    def ip3(t):
        prev=.2
        for start,nxt in [(80.,.58),(160.,.3),(240.,.75),(320.,.2)]:
            if t<start:return prev
            if t<start+width:return prev+(nxt-prev)*(t-start)/width
            prev=nxt
        return prev
    t=np.arange(0,400.00001,.1)
    sol=solve_ivp(lambda t,y:rhs(model,ip3(t),y),(0,400),model['rest_target'],method='DOP853',rtol=1e-9,atol=1e-11,max_step=.03,t_eval=t)
    if not sol.success:raise RuntimeError(sol.message)
    mask=t>=320;C=sol.y[0,mask];tt=t[mask];j=np.argmin(C)
    item={'ramp_s':width,'post_320_sampled_min_C':float(C[j]),'min_time_s':float(tt[j]),'sample_interval_s':.1}
    out.append(item);print(item,flush=True)
(ROOT/'paper/data/random_selected/am_ramps.json').write_text(json.dumps({'scope':'AM 465, piecewise-linear input ramps, finite sampling; not general positivity proof','rtol':1e-9,'atol':1e-11,'max_step_s':.03,'results':out},indent=2)+'\n')
