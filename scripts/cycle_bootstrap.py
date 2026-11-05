"""Finite IVP initialization of return-map crossing times."""
import numpy as np
from scipy.integrate import solve_ivp
from fit_lower_fixed_fold import rhs

def bootstrap(model,I,duration=400.):
    cuts=model['physical_thresholds_C']
    def e0(t,y): return y[0]-cuts[0]
    def e1(t,y): return y[0]-cuts[1]
    sol=solve_ivp(lambda t,y:rhs(model,I,y),(0,duration),model['rest_target'],events=(e0,e1),rtol=2e-8,atol=2e-10,max_step=.2)
    if not sol.success:return {}
    events=[]
    for j in (0,1):
        for t,y in zip(sol.t_events[j],sol.y_events[j]):
            direction=1 if rhs(model,I,y)[0]>0 else -1
            events.append((float(t),(j,direction)))
    events.sort();patterns={'both':[(0,1),(1,1),(1,-1),(0,-1),(0,1)],'upper':[(1,1),(1,-1),(1,1)],'lower':[(0,1),(0,-1),(0,1)]};seeds={}
    for name,pattern in patterns.items():
        for j in range(len(events)-len(pattern),-1,-1):
            window=events[j:j+len(pattern)]
            if [v[1] for v in window]==pattern:
                seeds[name]=[np.diff([v[0] for v in window])];break
    return seeds

