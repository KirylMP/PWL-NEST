"""Finite physicality stress: reachable high-IP3 states followed by I=0.

This is not a proof of invariance. q is a hidden model coordinate, not LR h.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from fit_lower_fixed_fold import rhs

def integrate(model, I, y, duration, rtol=1e-8, atol=1e-10, max_step=.15):
    def turning(t,y): return rhs(model,I,y)[0]
    turning.direction=0
    result=solve_ivp(lambda t,y:rhs(model,I,y),(0.,duration),y,method='DOP853',
                     events=turning,dense_output=True,rtol=rtol,atol=atol,max_step=max_step)
    if not result.success or not np.all(np.isfinite(result.y)):raise ValueError('IVP failed')
    values=[float(result.y[0,0]),float(result.y[0,-1]),*[float(x[0]) for x in result.y_events[0]]]
    return result,float(min(values))

def stress(model,high_inputs=(.75,1.,1.2),holds=(20.,40.,60.,80.),
           baseline=.2,pre_duration=80.,after_duration=30.,rtol=1e-8,atol=1e-10,max_step=.15):
    pre,pre_min=integrate(model,baseline,model['rest_target'],pre_duration,rtol,atol,max_step)
    cases=[];minimum=pre_min
    for high_I in high_inputs:
        high,high_min=integrate(model,float(high_I),pre.y[:,-1],max(holds),rtol,atol,max_step)
        minimum=min(minimum,high_min)
        for hold in holds:
            initial=high.sol(hold)
            sol,low_min=integrate(model,0.,initial,after_duration,rtol,atol,max_step)
            minimum=min(minimum,low_min)
            cases.append({'I_high':float(high_I),'hold_s':float(hold),'drop_state':initial.tolist(),'min_C_after':low_min})
    return {'scope':'finite protocol, continuous minimum candidates via derivative roots; no invariance proof',
            'protocol':{'baseline':baseline,'pre_s':pre_duration,'high_inputs':list(high_inputs),'holds_s':list(holds),
                        'I_after':0.,'after_s':after_duration,'rtol':rtol,'atol':atol,'max_step_s':max_step},
            'pre_min_C':pre_min,'min_C':float(minimum),'cases':cases}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('candidate',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();raw=a.candidate.read_bytes();result=stress(json.loads(raw));result['candidate_sha256']=hashlib.sha256(raw).hexdigest()
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(a.candidate,'min_C',result['min_C'],flush=True)
if __name__=='__main__':main()
