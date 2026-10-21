"""Re-evaluate saved tau=50 protocols with stricter numerical tolerances."""
import json
import numpy as np
from pwl_nest.models import ROOT
from pwl_nest.strict_pwa import StrictPWA
from pwl_nest.protocols import simulate

report = {}
for case in ['AM','FM']:
    model = StrictPWA.load(ROOT/f'models/strict_pwa/{case}.npz')
    with np.load(ROOT/f'results/strict_pwa/{case}_decay_tau_50.0.npz') as z:
        initial = z['lr'][0].copy()
        checks = {}
        for label,candidate in [('lr',None),('pwa',model)]:
            t,y,meta = simulate(case,initial,180.,tau=50.,baseline=.3 if case=='AM' else .45,
                                model=candidate,rtol=2e-9,atol=2e-11,max_step=.1)
            checks[label] = dict(max_abs_C_difference=float(np.max(abs(z[label][:,0]-y[:,0]))),
                                 C_RMSE_difference=float(np.sqrt(np.mean((z[label][:,0]-y[:,0])**2))))
            if checks[label]['max_abs_C_difference'] > .002:
                raise AssertionError(checks)
        report[case] = checks
print(json.dumps(report,indent=2))
(ROOT/'results/strict_pwa/numerical_convergence.json').write_text(json.dumps(report,indent=2)+'\n')
