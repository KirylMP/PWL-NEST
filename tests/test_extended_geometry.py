import unittest,tempfile
from pathlib import Path
import numpy as np
from pwl_nest.calcium_coordinates import CalciumCoordinate,GateCoordinate,CalciumTransformPWA,NativeCoordinateDynamic,simulate_coordinates
from pwl_nest.calcium_strips import CalciumStrips
from pwl_nest.thresholds import OrderedThresholds,oblique_thresholds
from pwl_nest.strict_pwa import regional_flow,lr_field
from pwl_nest.models import ROOT
from test_calcium_coordinates import simple_model


def geometry_model():
    model=simple_model('sqrt');model.latent.matrix[:,4:]=[[.05,-.03,.02],[.01,.02,-.01]]
    return CalciumTransformPWA(model.latent,model.coordinate,model.h_reference,'logit',
                              [.35,.55],[[.008,-.01],[.02,-.015]],[.01,-.005,.02])


class ExtendedGeometryTest(unittest.TestCase):
    def test_hill_coordinate_inverse_derivative_and_chain_rule(self):
        for half in [.05,.1]:
            raw=CalciumCoordinate('hill',.05,half_saturation=half)
            co=CalciumCoordinate('hill',.05,float(raw.forward(1.6)/1.6),half)
            c=np.array([.001,.03,.05,.1,1.6])
            np.testing.assert_allclose(co.inverse(co.forward(c)),c,atol=1e-12)
            self.assertAlmostEqual(float(co.forward(1.6)),1.6,places=14)
            for x in [.01,.1,.7]:
                numeric=(co.forward(x+1e-6)-co.forward(x-1e-6))/2e-6
                np.testing.assert_allclose(numeric,co.derivative(x),rtol=2e-8)
            for boundary in [co.lower,co.upper]:
                with self.assertRaises(ValueError):co.inverse(boundary)
            model=simple_model('sqrt');physical=model.metadata['physical_thresholds_C']
            model.coordinate=co;model.latent.thresholds=co.forward(physical);model.h_kind='cube'
            x=np.array([.12,.65,.6]);f=lr_field(x,'AM');eps=1e-6
            numeric=(model.to_latent(x+eps*np.r_[f,0])-model.to_latent(x-eps*np.r_[f,0]))/(2*eps)
            np.testing.assert_allclose(numeric[:2],model.source_field(model.to_latent(x)),atol=1e-8)
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'hill.json';model.save(path);loaded=CalciumTransformPWA.load(path)
                self.assertEqual(loaded.coordinate.half_saturation,half)
                np.testing.assert_allclose(loaded.to_physical(model.to_latent(x)),x,atol=1e-14)
    def test_cube_gate_chain_rule_and_positive_branch(self):
        gate=GateCoordinate('cube',.75);h=np.array([.01,.1,.5,.75,.999])
        np.testing.assert_allclose(gate.inverse(gate.forward(h)),h,atol=1e-12)
        self.assertAlmostEqual(float(gate.derivative(.75)),1/(.75*.25),places=14)
        for v in [gate.lower,gate.upper,gate.lower-.1,gate.upper+.1]:
            with self.assertRaises(ValueError):gate.inverse(v)
        model=simple_model('sqrt');model.h_kind='cube';x=np.array([.12,.65,.6]);f=lr_field(x,'AM')
        eps=1e-6;y=model.to_latent(x)
        numeric=(model.to_latent(x+eps*np.r_[f,0])-model.to_latent(x-eps*np.r_[f,0]))/(2*eps)
        np.testing.assert_allclose(numeric[:2],model.source_field(y),atol=1e-8)
        # Similarity holds at source equilibria only; use one saved event.
        seed=CalciumTransformPWA.load(ROOT/'models/movable_thresholds/selected_AM.json');e=seed.metadata['events'][0]
        x=np.array([e['C'],e['h'],e['I']]);y=model.to_latent(x)
        jx=np.column_stack([lr_field(x.astype(complex)+1e-25j*np.eye(3)[d],'AM').imag/1e-25 for d in [0,1]])
        jy=np.column_stack([model.source_field(y.astype(complex)+1e-25j*np.eye(3)[d]).imag/1e-25 for d in [0,1]])
        transform=np.diag([model.coordinate.derivative(x[0]),model.gate.derivative(x[1])])
        np.testing.assert_allclose(jy,transform@jx@np.linalg.inv(transform),atol=2e-10)

    def test_full_field_constant_input_reduction_and_continuity(self):
        model=geometry_model();rng=np.random.default_rng(23)
        for ip3 in [0.,.35,.5,.55,1.3]:
            effective=model.at_input(ip3)
            for u,v in rng.uniform([.01,-1],[.2,1],(12,2)):
                y=np.array([u,v,ip3]);a,b,_=model.region(y,2,.2)
                np.testing.assert_allclose(model.field(y),effective.latent.field(y),atol=1e-15)
                np.testing.assert_allclose((a@y+b)[:2],model.field(y),atol=1e-15)
            local=model.geometry_coefficients();band=np.searchsorted(model.input_thresholds,ip3,side='right')
            for j,th in enumerate(model.thresholds_at_input(ip3)):
                state=np.array([1.,th,.3,ip3])
                np.testing.assert_allclose(local[band,j]@state,local[band,j+1]@state,atol=1e-15)
        for s,knot in enumerate(model.input_thresholds):
            for j,u in enumerate(np.r_[.001,(model.thresholds_at_input(knot)[:-1]+model.thresholds_at_input(knot)[1:])/2,.3]):
                state=np.array([1.,u,.3,knot])
                np.testing.assert_allclose(local[s,j]@state,local[s+1,j]@state,atol=1e-15)

    def test_analytic_regional_flow_and_native_across_input_crossings(self):
        model=geometry_model();initial=[.1,.8,.7];y0=model.to_latent(initial)
        a,b,region=model.region(y0,2.,.2)
        t,x,meta=simulate_coordinates(model,initial,8.,tau=2,baseline=.2,dt=.02)
        np.testing.assert_allclose(meta['latent'][1],regional_flow(a,b,y0,.02),atol=2e-12)
        rk=NativeCoordinateDynamic().run(model,initial,8.,2.,.2,outputs=len(t),maxdt=.001)
        self.assertIsNotNone(rk)
        np.testing.assert_allclose(rk,meta['latent'][:,:2],atol=2e-8)
        crossings=model.input_crossing_times(.7,2.,.2,8.)
        self.assertEqual(len(crossings),2)
        for when,knot in crossings:self.assertAlmostEqual(.2+.5*np.exp(-when/2),knot,places=14)
        self.assertEqual(model.input_crossing_times(.2,2.,.2,8.),[])
        self.assertEqual(model.input_crossing_times(.7,2.,.2,.1),[])

    def test_oblique_endpoints_stay_ordered_and_restore_zero(self):
        mapping=OrderedThresholds([.05,.06,.1,.3],0.,.5)
        center,slopes=oblique_thresholds(mapping,np.zeros(4))
        np.testing.assert_array_equal(center,mapping.reference);np.testing.assert_array_equal(slopes,np.zeros(4))
        rng=np.random.default_rng(1)
        for shifts in rng.uniform(-np.log(2),np.log(2),(20,4)):
            center,slopes=oblique_thresholds(mapping,shifts)
            for ip3 in np.linspace(0,1.3,11):
                values=center+slopes*(ip3-.65)
                self.assertTrue(np.all(np.diff(np.r_[mapping.lower,values,mapping.upper])>0))

    def test_equilibrium_input_anchors_are_actual_roots(self):
        seed=CalciumTransformPWA.load(ROOT/'models/movable_thresholds/selected_AM.json')
        B,D=seed.latent.matrix[:,2];qb=seed.latent.spectral_coefficients()[3]
        p=np.array([.01,-.02]);weights=np.array([p,(qb*np.array([.1,-.2])+D*p)/B])
        model=CalciumTransformPWA(seed.latent,seed.coordinate,seed.h_reference,seed.h_kind,[.35,.64],weights,np.full(3,.001))
        anchors=[e['I'] for e in seed.metadata['events']]
        for j,ip3 in enumerate(model.equilibrium_inputs_at_thresholds(anchors)):
            u=model.thresholds_at_input(ip3)[j];local=model.geometry_coefficients()[np.searchsorted(model.input_thresholds,ip3,side='right'),j]
            v=-(local[0,0]+local[0,1]*u+local[0,3]*ip3)/local[0,2]
            np.testing.assert_allclose(model.field([u,v,ip3]),[0.,0.],atol=2e-14)

    def test_geometry_jumps_and_serialization(self):
        model=geometry_model();t,x,meta=simulate_coordinates(model,[.1,.8,.3],2.,tau=3,baseline=.2,jumps=[(1.,.4)],dt=.02)
        event=meta['jumps'][0]
        np.testing.assert_array_equal(event['physical_before'][:2],event['physical_after'][:2])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.json';model.save(path);loaded=CalciumTransformPWA.load(path)
            np.testing.assert_allclose(loaded.field(meta['latent']),model.field(meta['latent']),atol=1e-15)
            np.testing.assert_allclose(loaded.threshold_slopes,model.threshold_slopes,atol=0)
            np.testing.assert_allclose(loaded.to_physical(meta['latent']),x,atol=1e-14)


if __name__=='__main__':unittest.main()
