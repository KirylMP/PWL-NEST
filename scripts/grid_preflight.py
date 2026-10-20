"""Non-destructive runtime check before detaching the overnight process."""
import json,sys
import numpy as np
import scipy
from pwl_nest.models import ROOT
from pwl_nest.calcium_coordinates import CalciumTransformPWA,simulate_coordinates
from pwl_nest.rest_protocols import stable_rest,native_events

if sys.version_info<(3,11):raise RuntimeError('Python >=3.11 is required')
for case,baseline,target in [('AM',.3,[.123,.747,.3]),('FM',.45,[.0425,.911,.45])]:
    model=CalciumTransformPWA.load(ROOT/f'models/positive_coordinates/sqrt_tails/{case}.json')
    rest,_=stable_rest(model,baseline,target)
    protocol=dict(duration=2.,tau=5.,baseline=baseline,jumps=[(.5,.01)],dt=.05)
    t,x,_=native_events(model,rest,**protocol,maxdt=.001)
    tr,y,_=simulate_coordinates(model,rest,**protocol,rtol=2e-11,atol=2e-13,max_step=.03)
    np.testing.assert_allclose(t,tr,atol=1e-12)
    np.testing.assert_allclose(x,y,atol=2e-6)
print(json.dumps(dict(preflight='passed',python=sys.version.split()[0],numpy=np.__version__,scipy=scipy.__version__,project=str(ROOT))))
