"""Check local boundary-equilibrium hypotheses for Simpson HLB1/HLB2."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1];base=ROOT/'paper/data/random_selected';out={}
for case in ('AM','AFM','FM'):
    p=base/case/'candidate.json';raw=p.read_bytes();d=json.loads(raw)
    T=np.asarray(d['trace']);D=np.asarray(d['determinant']);delta=T*T-4*D
    beta=d['a']*d['b']*(d['beta'][2]-d['alpha'][2])
    # T/(sqrt(4D-T²)) equals real part divided by imaginary part.
    alpha=T[1]/np.sqrt(-delta[1])+T[2]/np.sqrt(-delta[2])
    slopes=beta/D
    assert T[0]<0 and D[0]>0 and delta[0]>0
    assert T[1]>0 and D[1]>0 and delta[1]<0
    assert T[2]<0 and D[2]>0 and delta[2]<0
    assert beta>0 and alpha>0 and min(slopes)>0
    out[case]={'candidate_sha256':hashlib.sha256(raw).hexdigest(),'I1_I2':d['fixed_equilibrium_threshold_I'],
               'threshold_C':d['physical_thresholds_C'],'T':T.tolist(),'D':D.tolist(),'discriminant':delta.tolist(),
               'beta_oriented':float(beta),'equilibrium_dC_dI':slopes.tolist(),
               'upper_alpha':float(alpha),'lower':'HLB2, local stable cycle for I>I1',
               'upper':'HLB1, local unstable cycle for I>I2',
               'scope':'local two-zone theorems with reflected coordinates and parameter orientation; no global completeness or basin claim'}
    print(case,'beta',beta,'alpha',alpha,'slopes',slopes,flush=True)
(base/'local_hlb_audit.json').write_text(json.dumps({'results':out},indent=2,allow_nan=False)+'\n')
