import unittest,tempfile,json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import expit,logit
from pwl_nest.calcium_coordinates import (CalciumCoordinate,GateCoordinate,CalciumTransformPWA,NativeCoordinateDynamic,simulate_coordinates)
from pwl_nest.calcium_strips import CalciumStrips,NativeDynamic
from pwl_nest.strict_pwa import regional_flow,lr_field,spectral_events

KINDS=['identity','sqrt','cbrt','log','square','cube','exp','inverse','inverse_sqrt']


def simple_model(kind):
    co=CalciumCoordinate(kind,.05)
    matrix=np.zeros((2,7))
    matrix[0,:4]=[.05,-1.,0.,.02]
    if kind=='inverse':matrix[0,0]=.01
    matrix[1,:4]=[0.,.1,-1.,-.1]
    theta=np.array([.08,.15,.3])
    return CalciumTransformPWA(CalciumStrips('AM',co.forward(theta),matrix,dict(physical_thresholds_C=theta.tolist())),co,.75)


class CalciumCoordinatesTest(unittest.TestCase):
    def test_bounded_tails_preserve_core_and_join_derivatives(self):
        from pwl_nest.strict_pwa import C_MAX
        core=CalciumCoordinate('sqrt',.05,1.7);co=CalciumCoordinate('sqrt_tails',.05,1.7)
        middle=np.linspace(.005,1.3,100)
        np.testing.assert_array_equal(co.forward(middle),core.forward(middle))
        np.testing.assert_array_equal(co.derivative(middle),core.derivative(middle))
        c=np.r_[1e-8,.001,middle,C_MAX-1e-7]
        np.testing.assert_allclose(co.inverse(co.forward(c)),c,rtol=1e-12,atol=1e-15)
        self.assertAlmostEqual(float(co.inverse(co.forward(1e-200)))/1e-200,1.,places=12)
        for join in [.005,1.3]:
            for delta in [-1e-9,1e-9]:
                np.testing.assert_allclose(co.derivative(join+delta),core.derivative(join),rtol=1e-6)
        for point in [.001,.1,1.6]:
            numeric=(co.forward(point+1e-7)-co.forward(point-1e-7))/2e-7
            np.testing.assert_allclose(numeric,co.derivative(point),rtol=1e-8)
        self.assertGreater(float(co.inverse(core.lower-.1)),0.)
        self.assertLess(float(co.inverse(core.forward(2.))),C_MAX)
        for invalid in [0.,C_MAX]:
            with self.assertRaises(ValueError):co.forward(invalid)
        with self.assertRaises(ValueError):co.inverse(-1e6)
        with self.assertRaises(ValueError):co.inverse(1e6)
        model=simple_model('sqrt_tails')
        x=np.array([1.5,.8,.7]);f=lr_field(x,'AM');eps=1e-6
        numeric=(model.to_latent(x+eps*np.r_[f,0])-model.to_latent(x-eps*np.r_[f,0]))/(2*eps)
        np.testing.assert_allclose(model.source_field(model.to_latent(x)),numeric[:2],atol=1e-8)
        model.latent.matrix[:]=0.;model.latent.matrix[0,0]=.2
        y0=model.to_latent([1.25,.8,.7]);a,b,_=model.region(y0,2.,.2)
        t,x,meta=simulate_coordinates(model,[1.25,.8,.7],2.,tau=2.,baseline=.2,dt=.02)
        exact=np.array([regional_flow(a,b,y0,float(s)) for s in t])
        np.testing.assert_allclose(meta['latent'],exact,atol=2e-12)
        np.testing.assert_allclose(x,model.to_physical(exact),atol=2e-12)
        self.assertGreater(x[-1,0],1.3)
        self.assertLess(x[-1,0],C_MAX)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'tails.json';model.save(path)
            restored=CalciumTransformPWA.load(path)
            np.testing.assert_allclose(restored.to_physical(exact),x)
            record=json.loads(path.read_text());record['calcium_coordinate']['tail_joins_C']=[.01,1.4]
            path.write_text(json.dumps(record))
            with self.assertRaises(ValueError):CalciumTransformPWA.load(path)

    def test_bounded_calcium_bijection_and_source_chain_rule(self):
        from pwl_nest.strict_pwa import C_MAX
        co=CalciumCoordinate('bounded_logit',.05)
        c=np.array([1e-5,.03,.1,1.,C_MAX-1e-5])
        np.testing.assert_allclose(co.inverse(co.forward(c)),c,rtol=1e-12)
        for invalid in [0.,-1.,C_MAX,C_MAX+.01]:
            with self.assertRaises(ValueError):co.forward(invalid)
        with self.assertRaises(ValueError):co.inverse(1e6)
        with self.assertRaises(ValueError):co.inverse(-1e6)
        self.assertTrue(np.isinf(co.guard_upper))
        model=simple_model('bounded_logit');x=np.array([.12,.65,.6]);f=lr_field(x,'AM')
        y=model.to_latent(x);eps=1e-6
        numeric=(model.to_latent(x+eps*np.r_[f,0])-model.to_latent(x-eps*np.r_[f,0]))/(2*eps)
        np.testing.assert_allclose(model.source_field(y),numeric[:2],atol=1e-8)
        _,physical,_=simulate_coordinates(model,[.1,.8,.7],1.,tau=2.,baseline=.2)
        self.assertTrue(np.all((physical[:,0]>0)&(physical[:,0]<C_MAX)))

    def test_sixth_power_branches_and_analytic_flow(self):
        for kind in ['sixth','sixth_root']:
            co=CalciumCoordinate(kind,.05)
            c=np.array([.02,.05,.1,.7,1.6])
            np.testing.assert_allclose(co.inverse(co.forward(c)),c,rtol=2e-12)
            for point in [.03,.08,.3]:
                numeric=(co.forward(point+1e-7)-co.forward(point-1e-7))/2e-7
                np.testing.assert_allclose(numeric,co.derivative(point),rtol=1e-8)
            with self.assertRaises(ValueError):co.inverse(co.lower-.01)
            model=simple_model(kind);model.h_kind='sixth'
            y=model.to_latent([.1,.8,.7]);a,b,_=model.region(y,2.,.2)
            t,x,meta=simulate_coordinates(model,[.1,.8,.7],.1,tau=2.,baseline=.2,dt=.01)
            expected=np.array([regional_flow(a,b,y,float(s)) for s in t])
            np.testing.assert_allclose(meta['latent'],expected,rtol=1e-10,atol=1e-12)
            f=lr_field([.1,.8,.7],'AM')
            np.testing.assert_allclose(model.source_field(y),[co.derivative(.1)*f[0],model.gate.derivative(.8)*f[1]])
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'sixth.json';model.save(path)
                restored=CalciumTransformPWA.load(path)
                np.testing.assert_allclose(restored.to_physical(meta['latent']),x)
        gate=GateCoordinate('sixth',.75)
        h=np.array([.1,.5,.75,.99])
        np.testing.assert_allclose(gate.inverse(gate.forward(h)),h,rtol=1e-11)
        for invalid in [gate.lower,gate.lower-.1,gate.upper]:
            with self.assertRaises(ValueError):gate.inverse(invalid)
        # A sixth root readout is only invertible on its chosen positive branch.
        co=CalciumCoordinate('sixth',.05)
        with self.assertRaisesRegex(ValueError,'floating-point precision'):co.forward(1e-12)

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
                if np.isfinite(co.upper):
                    for u in [co.upper,co.upper+.1]:
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

    def test_gate_transforms_chain_rule_flow_jumps_and_serialization(self):
        x=np.array([.12,.65,.6]);f=lr_field(x,'AM')
        for kind in ['logit','log','exp']:
            gate=GateCoordinate(kind,.75);h=np.array([.001,.1,.65,.75,.999])
            np.testing.assert_allclose(gate.inverse(gate.forward(h)),h,atol=2e-15)
            self.assertAlmostEqual(float(gate.forward(.75)),0.,places=14)
            self.assertAlmostEqual(float(gate.derivative(.75)),1/(.75*.25),places=14)
            for value in [gate.lower,gate.upper]:
                with self.assertRaises(ValueError):gate.inverse(value)
            model=simple_model('sqrt');model.h_kind=kind;y=model.to_latent(x)
            np.testing.assert_allclose(model.source_field(y),[model.coordinate.derivative(x[0])*f[0],gate.derivative(x[1])*f[1]],atol=1e-13)
            eps=1e-6
            numeric=(model.to_latent(x+eps*np.r_[f,0])-model.to_latent(x-eps*np.r_[f,0]))/(2*eps)
            np.testing.assert_allclose(numeric[:2],model.source_field(y),atol=1e-8)
            initial=[.1,.8,.2];a,b,_=model.latent.region(model.to_latent(initial),3,.2)
            t,physical,meta=simulate_coordinates(model,initial,2.,tau=3,baseline=.2,jumps=[(1.,.1)],dt=.02)
            at_jump=meta['jumps'][0]
            np.testing.assert_array_equal(at_jump['physical_before'][:2],at_jump['physical_after'][:2])
            np.testing.assert_allclose(meta['latent'][t==1.][0],regional_flow(a,b,model.to_latent(initial),1.),atol=2e-12)
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'gate.json';model.save(path);restored=CalciumTransformPWA.load(path)
                self.assertEqual(restored.h_kind,kind)
                np.testing.assert_allclose(restored.to_physical(meta['latent']),physical,atol=1e-14)

    def test_gate_exit_guard_on_internal_steps(self):
        for kind in ['log','exp','cube']:
            model=simple_model('sqrt');model.h_kind=kind
            model.latent.matrix[1]=0;model.latent.matrix[1,0]=3.
            with self.assertRaisesRegex(RuntimeError,'gate domain exit h=1'):
                simulate_coordinates(model,[.1,.8,.2],1.,baseline=.2)
            self.assertIsNone(NativeCoordinateDynamic().run(model,[.1,.8,.2],1.,2.,.2,outputs=2,maxdt=.01))
        model.h_kind='exp';model.latent.matrix[1,0]=-5.
        with self.assertRaisesRegex(RuntimeError,'gate domain exit h=0'):
            simulate_coordinates(model,[.1,.8,.2],1.,baseline=.2)
        self.assertIsNone(NativeCoordinateDynamic().run(model,[.1,.8,.2],1.,2.,.2,outputs=2,maxdt=.01))
        model.h_kind='cube'
        with self.assertRaisesRegex(RuntimeError,'gate domain exit h=0'):
            simulate_coordinates(model,[.1,.8,.2],1.,baseline=.2)
        self.assertIsNone(NativeCoordinateDynamic().run(model,[.1,.8,.2],1.,2.,.2,outputs=2,maxdt=.01))

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
