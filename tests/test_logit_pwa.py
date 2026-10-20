import unittest
import tempfile
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import expit
from pwl_nest.calcium_strips import decode
from pwl_nest.logit_pwa import (LogitPWA,to_latent,to_physical,transformed_lr_field,
                                calcium_matched_seed,simulate_logit)
from pwl_nest.strict_pwa import lr_field,equilibrium_h,regional_flow


class LogitPWATest(unittest.TestCase):
    def setUp(self):
        theta=np.array([.1,.2,.4]);mu=np.full(4,-.3);k=np.full(4,.2)
        z=np.r_[np.log(abs(mu)),np.log(k),.4,np.log(.1),np.log(.5),np.log(.3),.1,-.2]
        self.physical=decode('AM',theta,z,np.sign(mu),np.ones(4))
        self.model=calcium_matched_seed(self.physical,.2,anchor_h=.8)

    def test_round_trip_and_no_clipping(self):
        x=np.array([[.1,1e-6,.2],[.2,.5,.4],[.3,1-1e-6,.7]])
        np.testing.assert_allclose(to_physical(to_latent(x)),x,atol=2e-15)
        for bad in [0.,1.,-.1,1.1]:
            with self.assertRaises(ValueError):to_latent([.1,bad,.2])

    def test_saved_model_declares_latent_basis(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.json'
            self.model.save(path)
            data=json.loads(path.read_text())
            self.assertEqual(data['basis'][:4],['1','C','z','I'])
            restored=LogitPWA.load(path)
            np.testing.assert_array_equal(restored.latent.matrix,self.model.latent.matrix)
            self.assertEqual(restored.metadata['coordinates'],['C','z','I'])

    def test_source_chain_rule(self):
        x=np.array([.12,.65,.6]);y=to_latent(x)
        physical=lr_field(x,'AM');transformed=transformed_lr_field(y,'AM')
        np.testing.assert_allclose(transformed,[physical[0],physical[1]/(x[1]*(1-x[1]))],atol=1e-14)
        epsilon=1e-6
        tangent=(to_latent(x+epsilon*np.r_[physical,0])-to_latent(x-epsilon*np.r_[physical,0]))/(2*epsilon)
        np.testing.assert_allclose(tangent[:2],transformed,atol=1e-9)

    def test_source_equilibrium_jacobian_similarity(self):
        # h equilibrium suffices: the nonlinear coordinate's derivative depends
        # only on h, so extra Jacobian term vanishes when hdot=0.
        for case in ['AM','FM']:
            x=np.array([.15,equilibrium_h(.15,.6),.6]);y=to_latent(x)
            jx=np.column_stack([lr_field(x.astype(complex)+1e-25j*np.eye(3)[d],case).imag/1e-25 for d in [0,1]])
            jy=np.column_stack([transformed_lr_field(y.astype(complex)+1e-25j*np.eye(3)[d],case).imag/1e-25 for d in [0,1]])
            transform=np.diag([1,1/(x[1]*(1-x[1]))])
            np.testing.assert_allclose(jy,transform@jx@np.linalg.inv(transform),atol=2e-13)

    def test_control_preserves_calcium_closed_system(self):
        for left,right in zip(self.physical.spectral_coefficients(),self.model.latent.spectral_coefficients()):
            np.testing.assert_allclose(left,right,atol=2e-14)
        x=[.15,.8,.7]
        for tau,base in [(20,.7),(2,.2)]:
            expected=solve_ivp(self.physical.rhs(tau,base),(0,2),x,t_eval=np.linspace(0,2,41),rtol=1e-11,atol=1e-13,max_step=.01)
            t,actual,meta=simulate_logit(self.model,x,2,tau=tau,baseline=base,dt=.05,rtol=1e-11,atol=1e-13,max_step=.01)
            np.testing.assert_allclose(actual[:,0],expected.y[0],atol=2e-12)

    def test_latent_exponential_and_physical_readout(self):
        initial=[.15,.8,.7];latent=to_latent(initial)
        a,b,_=self.model.latent.region(latent,2,.2)
        end=regional_flow(a,b,latent,1e-4)
        _,physical,meta=simulate_logit(self.model,initial,1e-4,tau=2,baseline=.2,dt=1e-4)
        np.testing.assert_allclose(meta['latent'][-1],end,atol=2e-13)
        np.testing.assert_allclose(physical[-1],to_physical(end),atol=2e-13)

    def test_jump_updates_only_I_and_internal_state_is_latent(self):
        t,x,meta=simulate_logit(self.model,[.15,.8,.2],2,tau=3,baseline=.2,jumps=[(1,.1)],dt=.02)
        jump=meta['jumps'][0]
        np.testing.assert_array_equal(jump['latent_before'][:2],jump['latent_after'][:2])
        np.testing.assert_array_equal(jump['physical_before'][:2],jump['physical_after'][:2])
        self.assertAlmostEqual(x[-1,2],.2+.1*np.exp(-1/3),places=13)
        np.testing.assert_allclose(x[:,1],expit(meta['latent'][:,1]),atol=1e-15)
        self.assertGreater(np.max(abs(x[:,1]-meta['latent'][:,1])),.1)

if __name__=='__main__':unittest.main()
