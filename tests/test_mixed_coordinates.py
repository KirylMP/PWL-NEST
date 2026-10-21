"""Mixed coordinates keep a PWA latent field and use physical-domain guards."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from pwl_nest.calcium_coordinates import (CalciumTransformPWA, NativeCoordinateDynamic,
                                         simulate_coordinates)
from pwl_nest.rest_protocols import native_events
from pwl_nest.strict_pwa import lr_field, regional_flow
from test_calcium_coordinates import simple_model


class MixedCoordinatesTest(unittest.TestCase):
    def model(self, a=.025, b=.004):
        m=simple_model('sqrt')
        m.mix_linear,m.mix_quadratic,m.mix_center=a,b,.1
        m.__post_init__()
        return m

    def test_roundtrip_chain_rule_and_thresholds(self):
        m=self.model()
        x=np.array([[.03,.2,.6],[.12,.65,.6],[.9,.95,.7]])
        y=m.to_latent(x)
        np.testing.assert_allclose(m.to_physical(y),x,atol=3e-15)
        f=np.column_stack([lr_field(x,m.case),np.zeros(len(x))])
        eps=1e-7
        numeric=(m.to_latent(x+eps*f)-m.to_latent(x-eps*f))/(2*eps)
        np.testing.assert_allclose(numeric[:,:2],m.source_field(y),atol=2e-8)
        v=np.linspace(-.4,.4,7)
        for theta in m.latent.thresholds:
            physical=m.to_physical(np.column_stack([np.full(len(v),theta),v,np.full(len(v),.6)]))
            np.testing.assert_allclose(m.to_latent(physical)[:,0],theta,atol=1e-15)
            self.assertGreater(np.ptp(physical[:,0]),.001)

    def test_exact_regional_motion_and_input_jump(self):
        m=self.model();initial=[.1,.8,.2];y0=m.to_latent(initial)
        a,b,_=m.region(y0,3.,.2)
        truth=solve_ivp(lambda t,y:a@y+b,[0,1],y0,rtol=1e-12,atol=1e-14,method='DOP853').y[:,-1]
        np.testing.assert_allclose(regional_flow(a,b,y0,1.),truth,atol=1e-13)
        t,x,meta=simulate_coordinates(m,initial,2.,tau=3.,baseline=.2,jumps=[(1.,.1)],dt=.02)
        jump=meta['jumps'][0]
        np.testing.assert_array_equal(jump['physical_before'][:2],jump['physical_after'][:2])
        np.testing.assert_allclose(meta['latent'][t==1.][0],regional_flow(a,b,y0,1.),atol=2e-12)
        tn,xn,_=native_events(m,initial,2.,3.,.2,[(1.,.1)],dt=.02,maxdt=.001)
        np.testing.assert_allclose(tn,t,atol=1e-14)
        np.testing.assert_allclose(xn,x,atol=1e-10)

    def test_guards_follow_physical_calcium_not_U(self):
        # U is below phi's lower boundary, while U+g(V) is valid.
        m=self.model(a=2.,b=0.);m.latent.matrix[:]=0.
        initial=[.01,.9,.2]
        self.assertLess(m.to_latent(initial)[0],m.coordinate.lower)
        _,x,_=simulate_coordinates(m,initial,.1,baseline=.2)
        np.testing.assert_allclose(x[:,0],.01,atol=1e-15)
        self.assertIsNotNone(NativeCoordinateDynamic().run(m,initial,.1,2.,.2,outputs=2))
        # V decreases at fixed U: calcium exits despite U itself being constant.
        m.latent.matrix[1,0]=-10.
        with self.assertRaisesRegex(RuntimeError,'positive-root branch exit'):
            simulate_coordinates(m,initial,1.,baseline=.2)
        self.assertIsNone(NativeCoordinateDynamic().run(m,initial,1.,2.,.2,outputs=2,maxdt=.001))

    def test_save_load_and_constant_input_preserve_mixing(self):
        m=self.model();m.input_thresholds=np.array([.4]);m.input_weights=np.array([[.01],[.02]])
        m.__post_init__()
        y=m.to_latent([.12,.65,.6])
        np.testing.assert_allclose(m.at_input(.6).to_physical(y),m.to_physical(y))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'mixed.json';m.save(path)
            restored=CalciumTransformPWA.load(path)
        self.assertEqual(restored.mix_quadratic,m.mix_quadratic)
        np.testing.assert_allclose(restored.to_physical(y),m.to_physical(y))
        np.testing.assert_allclose(restored.field(y),m.field(y))


if __name__=='__main__':unittest.main()
