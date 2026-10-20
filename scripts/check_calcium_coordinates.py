"""Verify actual saved coordinate variants: thresholds, continuity and exact flow."""
import argparse,json
import numpy as np
from scipy.integrate import solve_ivp
from pwl_nest.models import ROOT
from pwl_nest.strict_pwa import C_MAX,lr_field,regional_flow
from pwl_nest.calcium_coordinates import CalciumTransformPWA

parser=argparse.ArgumentParser()
parser.add_argument('--experiment',choices=['calcium_coordinates','calcium_powers','coordinate_families'],default='calcium_coordinates')
args=parser.parse_args()
OUT=ROOT/'results'/args.experiment;MODELS=ROOT/'models'/args.experiment;report={}
KINDS=['sqrt','square','cube'] if args.experiment=='calcium_powers' else ['identity','sqrt','cbrt','log']
for case in ['AM','FM']:
    report[case]={}
    if args.experiment=='coordinate_families':KINDS=json.loads((OUT/f'{case}_selection.json').read_text())['variants']
    for kind in KINDS:
        model=CalciumTransformPWA.load(MODELS/kind/f'{case}.json')
        latent=model.latent;local=latent.local_coefficients();theta=latent.thresholds;co=model.coordinate
        np.testing.assert_allclose(theta,co.forward(model.metadata['physical_thresholds_C']),atol=2e-14)
        continuity=0.;flow_error=0.;physical_flow_error=0.;ip3_error=0.;jac_error=0.;types=[]
        for j,th in enumerate(theta):
            for v in [-3.,0.,3.]:
                for ip3 in [0.,.6,1.3]:
                    y=np.array([1.,th,v,ip3])
                    continuity=max(continuity,float(abs(local[j]@y-local[j+1]@y).max()))
        limits=np.r_[co.forward(co.reference/10),theta,co.forward(C_MAX)]
        for j in range(latent.region_count):
            point=np.array([(limits[j]+limits[j+1])/2,.2,.8]);a,b,region=latent.region(point,20.,.2)
            dt=min(1e-4,.03*(limits[j+1]-limits[j])/(1+abs((a@point+b)[0])))
            exact=regional_flow(a,b,point,dt)
            if latent.region(exact,20.,.2)[2]!=j:raise AssertionError('Regional test crossed a threshold')
            numeric=solve_ivp(latent.rhs(20.,.2),(0,dt),point,rtol=1e-12,atol=1e-14)
            if not numeric.success:raise AssertionError(numeric.message)
            flow_error=max(flow_error,float(abs(exact-numeric.y[:,-1]).max()))
            physical_flow_error=max(physical_flow_error,float(abs(model.to_physical(exact)-model.to_physical(numeric.y[:,-1])).max()))
            ip3_error=max(ip3_error,abs(exact[2]-(.2+.6*np.exp(-dt/20))))
            trace,det=np.trace(a[:2,:2]),np.linalg.det(a[:2,:2])
            types.append('saddle' if det<0 else ('stable_' if trace<0 else 'unstable_')+('focus' if trace*trace<4*det else 'node'))
        if types!=model.metadata['target_types']:raise AssertionError('Wrong regional types')
        for event in model.metadata['events']:
            x=np.array([event['C'],event['h'],event['I']]);y=model.to_latent(x)
            jx=np.column_stack([lr_field(x.astype(complex)+1e-25j*np.eye(3)[d],case).imag/1e-25 for d in [0,1]])
            jy=np.column_stack([model.source_field(y.astype(complex)+1e-25j*np.eye(3)[d]).imag/1e-25 for d in [0,1]])
            t=np.diag([co.derivative(x[0]),model.gate.derivative(x[1])])
            jac_error=max(jac_error,float(abs(jy-t@jx@np.linalg.inv(t)).max()))
        if continuity>1e-11 or flow_error>1e-10 or physical_flow_error>1e-9 or ip3_error>1e-13 or jac_error>1e-9:
            raise AssertionError('Structural error')
        report[case][kind]=dict(regions=latent.region_count,parameters=model.metadata['parameter_count'],
            continuity_max_abs_error=continuity,latent_flow_error=flow_error,physical_readout_flow_error=physical_flow_error,
            IP3_error=ip3_error,source_Jacobian_similarity_error=jac_error,regional_types=types,
            calcium_reference=co.reference,coordinate_scale=co.scale,h_reference=model.h_reference,
            h_transform=model.h_kind,calcium_lower_boundary_guard=bool(np.isfinite(co.lower)),
            h_lower_boundary_guard=bool(np.isfinite(model.gate.lower)),h_upper_boundary_guard=bool(np.isfinite(model.gate.upper)))
(OUT/'structure.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
