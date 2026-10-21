"""Verify persisted compact physical PWA structure, source labels and exact flow."""
import json
import numpy as np
from scipy.integrate import solve_ivp
from pwl_nest.models import ROOT
from pwl_nest.calcium_strips import CalciumStrips
from pwl_nest.strict_pwa import regional_flow,C_MAX

report={}
for case in ['AM','FM']:
    model=CalciumStrips.load(ROOT/f'models/calcium_strips/{case}.json')
    local=model.local_coefficients();theta=model.thresholds
    continuity=0.;flow_error=0.;ip3_error=0.
    for j,c in enumerate(theta):
        for h in [0.,.5,1.]:
            for ip3 in [0.,.6,1.3]:
                x=np.array([1,c,h,ip3])
                continuity=max(continuity,float(np.max(abs(local[j]@x-local[j+1]@x))))
    types=[]
    limits=np.r_[0.,theta,C_MAX]
    for j in range(model.region_count):
        point=np.array([(limits[j]+limits[j+1])/2,.7,.8])
        a,b,region=model.region(point,20.,.2)
        np.testing.assert_allclose(a[:,2],[model.matrix[0,3],model.matrix[1,3],-.05],atol=1e-14)
        f=a@point+b
        duration=min(1e-3,.03*(limits[j+1]-limits[j])/(1+abs(f[0])))
        target=regional_flow(a,b,point,duration)
        if model.region(target,20.,.2)[2]!=j:raise AssertionError('Regional check crossed a threshold')
        sol=solve_ivp(model.rhs(20.,.2),(0,duration),point,rtol=1e-12,atol=1e-14)
        if not sol.success:raise AssertionError(sol.message)
        flow_error=max(flow_error,float(np.max(abs(target-sol.y[:,-1]))))
        ip3_error=max(ip3_error,abs(target[2]-(.2+.6*np.exp(-duration/20))))
        tr,det=np.trace(a[:2,:2]),np.linalg.det(a[:2,:2])
        label='saddle' if det<0 else ('stable_' if tr<0 else 'unstable_')+('focus' if tr*tr<4*det else 'node')
        types.append(label)
    mu,k,q,qb=model.spectral_coefficients()
    predicted=(k[:-1]*theta-q[:-1])/qb
    anchors=[dict(C=float(c),source_I=e['I'],candidate_equilibrium_crossing_I=float(ip3))
             for c,e,ip3 in zip(theta,model.metadata['events'],predicted)]
    if model.metadata.get('enforce_types') and types!=model.metadata['target_types']:
        raise AssertionError((types,model.metadata['target_types']))
    if continuity>1e-11 or flow_error>1e-10 or ip3_error>1e-13:raise AssertionError('Structural failure')
    report[case]=dict(regions=model.region_count,free_parameters=model.metadata['free_parameters'],
                     continuity_max_abs_error=continuity,regional_flow_max_abs_error=flow_error,
                     IP3_max_abs_error=ip3_error,regional_types=types,thresholds=anchors)
(ROOT/'results/calcium_strips/structure.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
