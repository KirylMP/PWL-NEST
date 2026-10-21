"""Check saved input-slab and oblique affine fields, independently of fitting."""
import json
import numpy as np
from scipy.integrate import solve_ivp
from pwl_nest.models import ROOT
from pwl_nest.strict_pwa import C_MAX,lr_field,regional_flow
from pwl_nest.calcium_coordinates import CalciumTransformPWA

OUT=ROOT/'results/extended_pwa';MODELS=ROOT/'models/extended_pwa';report={}
for case in ['AM','FM']:
    selection=json.loads((OUT/f'{case}_selection.json').read_text());report[case]={}
    for variant in selection['variants']:
        model=CalciumTransformPWA.load(MODELS/variant/f'{case}.json')
        co=model.coordinate;theta=model.latent.thresholds;local=model.geometry_coefficients();n=model.latent.region_count
        np.testing.assert_allclose(theta,co.forward(model.metadata['physical_thresholds_C']),atol=2e-14)
        for ip3 in np.linspace(0,1.3,51):
            if np.any(np.diff(model.thresholds_at_input(ip3))<=0):raise AssertionError('Threshold planes intersect')
        continuity=0.;flow_error=0.;readout_error=0.;ip3_error=0.;reduction_error=0.;jac_error=0.;types=[]
        edges=np.r_[0.,model.input_thresholds,1.3]
        for s,(left,right) in enumerate(zip(edges[:-1],edges[1:])):
            ip3=(left+right)/2;thresholds=model.thresholds_at_input(ip3)
            for j,t in enumerate(thresholds):
                for v in [-2.,0.,2.]:
                    y=np.array([1.,t,v,ip3]);continuity=max(continuity,float(abs(local[s,j]@y-local[s,j+1]@y).max()))
            limits=np.r_[co.forward(co.reference/2),thresholds,co.forward(C_MAX)]
            labels=[]
            for j in range(n):
                point=np.array([(limits[j]+limits[j+1])/2,.2,ip3]);a,b,region=model.region(point,20.,.2)
                normal_velocity=(a@point+b)[0]-(model.threshold_slopes[max(0,j-1)] if len(theta) else 0)*(a@point+b)[2]
                dt=min(1e-4,.01*(limits[j+1]-limits[j])/(1+abs(normal_velocity)))
                exact=regional_flow(a,b,point,dt)
                if model.region(exact,20.,.2)[2]!=region:raise AssertionError('Regional check crossed a face')
                numeric=solve_ivp(model.rhs(20.,.2),(0,dt),point,rtol=1e-12,atol=1e-14)
                if not numeric.success:raise AssertionError(numeric.message)
                flow_error=max(flow_error,float(abs(exact-numeric.y[:,-1]).max()))
                readout_error=max(readout_error,float(abs(model.to_physical(exact)-model.to_physical(numeric.y[:,-1])).max()))
                ip3_error=max(ip3_error,abs(exact[2]-(.2+(ip3-.2)*np.exp(-dt/20))))
                reduction_error=max(reduction_error,float(abs(model.field(point)-model.at_input(ip3).latent.field(point)).max()))
                trace,det=np.trace(a[:2,:2]),np.linalg.det(a[:2,:2])
                labels.append('saddle' if det<0 else ('stable_' if trace<0 else 'unstable_')+('focus' if trace*trace<4*det else 'node'))
            if labels!=model.metadata['target_types']:raise AssertionError('Changed regional spectral types')
            types.append(labels)
        for s,knot in enumerate(model.input_thresholds):
            thresholds=model.thresholds_at_input(knot);limits=np.r_[co.forward(co.reference/2),thresholds,co.forward(C_MAX)]
            for j in range(n):
                y=np.array([1.,(limits[j]+limits[j+1])/2,.3,knot])
                continuity=max(continuity,float(abs(local[s,j]@y-local[s+1,j]@y).max()))
                np.testing.assert_allclose(local[s,j,:,1:3],local[s+1,j,:,1:3],atol=0)
        for e in model.metadata['events']:
            x=np.array([e['C'],e['h'],e['I']]);y=model.to_latent(x)
            jx=np.column_stack([lr_field(x.astype(complex)+1e-25j*np.eye(3)[d],case).imag/1e-25 for d in [0,1]])
            jy=np.column_stack([model.source_field(y.astype(complex)+1e-25j*np.eye(3)[d]).imag/1e-25 for d in [0,1]])
            transform=np.diag([co.derivative(x[0]),model.gate.derivative(x[1])])
            jac_error=max(jac_error,float(abs(jy-transform@jx@np.linalg.inv(transform)).max()))
        crossings=model.input_crossing_times(1.2,20.,.2,180.)
        for when,knot in crossings:
            if abs(.2+np.exp(-when/20)-knot)>2e-14:raise AssertionError('Wrong explicit input crossing')
        B,D=model.latent.matrix[:,2]
        input_slopes=(B*local[:,0,1,3]-D*local[:,0,0,3]).tolist()
        if len(model.input_thresholds) and min(input_slopes)<=0:raise AssertionError('Nonpositive restoring input slope')
        anchors=[e['I'] for e in model.metadata['events']]
        equilibrium_inputs=model.equilibrium_inputs_at_thresholds(anchors)
        equilibrium_error=0.
        for j,ip3 in enumerate(equilibrium_inputs):
            u=model.thresholds_at_input(ip3)[j]
            p=local[np.searchsorted(model.input_thresholds,ip3,side='right'),j]
            v=-(p[0,0]+p[0,1]*u+p[0,3]*ip3)/p[0,2]
            equilibrium_error=max(equilibrium_error,float(abs(model.field([u,v,ip3])).max()))
        if equilibrium_error>1e-8:raise AssertionError('Input anchors are not actual equilibrium intersections')
        if continuity>1e-10 or flow_error>1e-10 or readout_error>1e-8 or ip3_error>2e-14 or reduction_error>1e-10 or jac_error>1e-8:
            raise AssertionError('Structural tolerance exceeded')
        report[case][variant]=dict(C_regions=n,input_bands=len(edges)-1,total_regions=n*(len(edges)-1),
            active_fit_parameters=model.metadata.get('active_fit_parameters'),
            continuity_error=continuity,regional_flow_error=flow_error,physical_readout_flow_error=readout_error,
            input_exact_error=ip3_error,constant_input_reduction_error=reduction_error,source_Jacobian_similarity_error=jac_error,
            regional_types_by_input_band=types,input_thresholds=model.input_thresholds.tolist(),
            example_input_crossings=crossings,C_coordinate=co.kind,h_coordinate=model.h_kind,
            input_restoring_slopes_first_C_region=input_slopes,
            equilibrium_inputs_at_thresholds=equilibrium_inputs.tolist(),equilibrium_intersection_error=equilibrium_error,
            threshold_slopes=model.threshold_slopes.tolist(),reference_input=model.reference_input)
(OUT/'structure.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
