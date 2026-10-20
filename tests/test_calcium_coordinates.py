import unittest,tempfile,json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import expit,logit
from pwl_nest.calcium_coordinates import (CalciumCoordinate,CalciumTransformPWA,NativeCoordinateDynamic,simulate_coordinates)
from pwl_nest.calcium_strips import CalciumStrips,NativeDynamic
from pwl_nest.strict_pwa import regional_flow,lr_field,spectral_events

KINDS=['identity','sqrt','cbrt','log','square','cube']


def simple_model(kind):
    co=CalciumCoordinate(kind,.05)
    matrix=np.zeros((2,7))
    matrix[0,:4]=[.05,-1.,0.,.02]
    matrix[1,:4]=[0.,.1,-1.,-.1]
    theta=np.array([.08,.15,.3])
    return CalciumTransformPWA(CalciumStrips('AM',co.forward(theta),matrix,dict(physical_thresholds_C=theta.tolist())),co,.75)


class CalciumCoordinatesTest(unittest.TestCase):
    def test_round_trip_derivative_and_normalization(self):
        for kind in KINDS:
            c=np.array([.001 if kind in ('square','cube') else 1e-6,.01,.05,.1,1.6])
            co=CalciumCoordinate(kind,.05)
            np.testing.assert_allclose(co.inverse(co.forward(c)),c,rtol=1e-11,atol=1e-14)
            self.assertAlmostEqual(float(co.forward(.05)),0.,places=14)
            self.assertAlmostEqual(float(co.derivative(.05)),1.,places=14)
            for point in [.02,.1,.7]:
                numeric=(co.forward(point+1e-6)-co.forward(point-1e-6))/2e-6
                self.assertAlmostEqual(float(numeric),float(co.derivative(point)),places=7)

    def test_invalid_root_branch_is_not_folded(self):
        for kind in KINDS:
            co=CalciumCoordinate(kind,.05)
            for c in [0.,-.1]:
                with self.assertRaises(ValueError):co.forward(c)
            if kind!='log':
                for u in [co.lower,co.lower-.1]:
                    with self.assertRaises(ValueError):co.inverse(u)
            else:
                self.assertGreater(float(co.inverse(-1.)),0.)
                with self.assertRaises(ValueError):co.inverse(-1e6)

    def test_exact_source_chain_rule(self):
        x=np.array([.12,.65,.6]);f=lr_field(x,'AM')
        for kind in KINDS:
            model=simple_model(kind);y=model.to_latent(x);fy=model.source_field(y)
            np.testing.assert_allclose(fy,[model.coordinate.derivative(x[0])*f[0],f[1]/(x[1]*(1-x[1]))],atol=1e-13)
            eps=1e-6
            numeric=(model.to_latent(x+eps*np.r_[f,0])-model.to_latent(x-eps*np.r_[f,0]))/(2*eps)
            np.testing.assert_allclose(numeric[:2],fy,atol=1e-8)

    def test_source_equilibrium_jacobian_similarity(self):
        e=spectral_events('AM')[0];x=np.array([e['C'],e['h'],e['I']])
        jx=np.column_stack([lr_field(x.astype(complex)+1e-25j*np.eye(3)[d],'AM').imag/1e-25 for d in [0,1]])
        for kind in KINDS:
            model=simple_model(kind);y=model.to_latent(x)
            jy=np.column_stack([model.source_field(y.astype(complex)+1e-25j*np.eye(3)[d]).imag/1e-25 for d in [0,1]])
            t=np.diag([model.coordinate.derivative(x[0]),1/(x[1]*(1-x[1]))])
            np.testing.assert_allclose(jy,t@jx@np.linalg.inv(t),atol=1e-9)

    def test_analytic_flow_and_domain_aware_RK(self):
        for kind in KINDS:
            model=simple_model(kind);initial=[.1,.8,.7];y0=model.to_latent(initial)
            a,b,_=model.latent.region(y0,2,.2)
            t,x,meta=simulate_coordinates(model,initial,1.,tau=2,baseline=.2,dt=.05)
            expected=np.stack([regional_flow(a,b,y0,float(time)) for time in t])
            np.testing.assert_allclose(meta['latent'],expected,atol=2e-12)
            np.testing.assert_allclose(x,model.to_physical(expected),atol=2e-12)
            rk=NativeCoordinateDynamic().run(model,initial,1.,2.,.2,outputs=len(t),maxdt=.002)
            self.assertIsNotNone(rk)
            np.testing.assert_allclose(rk,expected[:,:2],atol=2e-11)

    def test_root_exit_terminates_simulation_and_native_fit(self):
        for kind in ['sqrt','cbrt','square','cube']:
            model=simple_model(kind)
            model.latent.matrix[0]=0.;model.latent.matrix[0,0]=-1.
            with self.assertRaisesRegex(RuntimeError,'positive-root branch exit'):
                simulate_coordinates(model,[.1,.8,.2],1.,baseline=.2)
            self.assertIsNone(NativeCoordinateDynamic().run(model,[.1,.8,.2],1.,2.,.2,outputs=2,maxdt=.01))

    def test_range_normalization_and_precision_loss(self):
        from pwl_nest.strict_pwa import C_MAX
        for kind in ['sqrt','square','cube']:
            scale=float(CalciumCoordinate(kind,.05).forward(C_MAX)/C_MAX)
            co=CalciumCoordinate(kind,.05,scale)
            self.assertAlmostEqual(float(co.forward(C_MAX)),C_MAX,places=14)
            self.assertAlmostEqual(float(co.derivative(.05)),1/scale,places=13)
            with self.assertRaises(ValueError):co.inverse(co.lower)
            c=np.array([.001,.03,.1,1.3])
            np.testing.assert_allclose(co.inverse(co.forward(c)),c,rtol=1e-10)
            model=simple_model(kind);model.coordinate=co
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'scaled.json';model.save(path)
                restored=CalciumTransformPWA.load(path)
                self.assertEqual(restored.coordinate.scale,scale)
                np.testing.assert_allclose(restored.coordinate.forward(c),co.forward(c))
        model=simple_model('sqrt')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'legacy.json';model.save(path)
            record=json.loads(path.read_text());record['calcium_coordinate'].pop('scale')
            path.write_text(json.dumps(record))
            self.assertEqual(CalciumTransformPWA.load(path).coordinate.scale,1.)
        with self.assertRaisesRegex(ValueError,'floating-point precision'):
            CalciumCoordinate('cube',.05).forward(1e-12)

    def test_jump_preserves_physical_C_h_and_recorded_latent_basis(self):
        for kind in KINDS:
            model=simple_model(kind)
            t,x,meta=simulate_coordinates(model,[.1,.8,.2],2.,tau=3,baseline=.2,jumps=[(1.,.1)],dt=.02)
            log=meta['jumps'][0]
            np.testing.assert_array_equal(log['latent_before'][:2],log['latent_after'][:2])
            np.testing.assert_array_equal(log['physical_before'][:2],log['physical_after'][:2])
            self.assertAlmostEqual(x[-1,2],.2+.1*np.exp(-1/3),places=13)
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'model.json';model.save(path);restored=CalciumTransformPWA.load(path)
                self.assertEqual(json.loads(path.read_text())['basis'][:4],['1','u','v','I'])
                np.testing.assert_allclose(restored.to_physical(meta['latent']),x,atol=1e-14)

if __name__=='__main__':unittest.main()
