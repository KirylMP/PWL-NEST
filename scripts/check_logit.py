"""Structural and coordinate checks on the saved logit PWA artifacts."""
import json
import numpy as np
from scipy.integrate import solve_ivp
from pwl_nest.models import ROOT
from pwl_nest.calcium_strips import CalciumStrips
from pwl_nest.logit_pwa import LogitPWA,to_latent,to_physical,transformed_lr_field
from pwl_nest.strict_pwa import regional_flow,C_MAX,lr_field,equilibrium_h

report={}
for case in ['AM','FM']:
    model=LogitPWA.load(ROOT/f'models/logit_pwa/{case}.json')
    control=LogitPWA.load(ROOT/f'models/logit_pwa/{case}_calcium_matched_control.json')
    baseline=CalciumStrips.load(ROOT/f'models/calcium_strips/{case}.json')
    latent=model.latent;local=latent.local_coefficients();theta=latent.thresholds
    continuity=0.;flow_error=0.;ip3_error=0.;source_jacobian_error=0.
    for j,c in enumerate(theta):
        for z in [-3.,0.,3.]:
            for ip3 in [0.,.6,1.3]:
                y=np.array([1,c,z,ip3])
                continuity=max(continuity,float(abs(local[j]@y-local[j+1]@y).max()))
    types=[];limits=np.r_[0.,theta,C_MAX]
    for j in range(latent.region_count):
        point=np.array([(limits[j]+limits[j+1])/2,.8,.8])
        a,b,region=latent.region(point,20.,.2)
        dt=min(1e-3,.03*(limits[j+1]-limits[j])/(1+abs((a@point+b)[0])))
        exact=regional_flow(a,b,point,dt)
        if latent.region(exact,20.,.2)[2]!=j:raise AssertionError('Check crossed a C boundary')
        sol=solve_ivp(latent.rhs(20.,.2),(0,dt),point,rtol=1e-12,atol=1e-14)
        if not sol.success:raise AssertionError(sol.message)
        flow_error=max(flow_error,float(abs(exact-sol.y[:,-1]).max()))
        ip3_error=max(ip3_error,abs(exact[2]-(.2+.6*np.exp(-dt/20.))))
        tr,det=np.trace(a[:2,:2]),np.linalg.det(a[:2,:2])
        types.append('saddle' if det<0 else ('stable_' if tr<0 else 'unstable_')+('focus' if tr*tr<4*det else 'node'))
    if types!=model.metadata['target_types']:raise AssertionError('Wrong fitted regional types')
    # Verify the exact transformed SOURCE Jacobian at all spectral equilibria.
    for e in model.metadata['events']:
        x=np.array([e['C'],e['h'],e['I']]);y=to_latent(x)
        jx=np.column_stack([lr_field(x.astype(complex)+1e-25j*np.eye(3)[d],case).imag/1e-25 for d in [0,1]])
        jy=np.column_stack([transformed_lr_field(y.astype(complex)+1e-25j*np.eye(3)[d],case).imag/1e-25 for d in [0,1]])
        t=np.diag([1.,1/(x[1]*(1-x[1]))])
        source_jacobian_error=max(source_jacobian_error,float(abs(jy-t@jx@np.linalg.inv(t)).max()))
    control_error=max(float(abs(np.asarray(a)-np.asarray(b)).max())
                      for a,b in zip(control.latent.spectral_coefficients(),baseline.spectral_coefficients()))
    for c in [0.,.1,.5,1.3]:
        for ip3 in [0.,.6,1.3]:
            x=np.array([c,.8,ip3])
            control_error=max(control_error,abs(baseline.field(x)[0]-control.latent.field(to_latent(x))[0]))
    if continuity>1e-11 or flow_error>1e-10 or ip3_error>1e-13 or source_jacobian_error>1e-10 or control_error>1e-11:
        raise AssertionError('Structural check failed')
    report[case]=dict(regions=latent.region_count,parameters=model.metadata['free_parameters'],
        continuity_max_abs_error=continuity,regional_flow_max_abs_error=flow_error,
        IP3_max_abs_error=ip3_error,source_equilibrium_Jacobian_similarity_error=source_jacobian_error,
        calcium_matched_control_invariant_error=control_error,regional_types=types,
        coordinates=['C','logit(h)','I'],physical_field_PWA=False)
(ROOT/'results/logit_pwa/structure.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
