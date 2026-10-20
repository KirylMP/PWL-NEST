import unittest
import numpy as np
from pwl_nest.models import ROOT
from pwl_nest.calcium_coordinates import CalciumTransformPWA,simulate_coordinates
from pwl_nest.rest_protocols import stable_rest,native_events,event_residual


class RestProtocolsTest(unittest.TestCase):
    def test_algebraic_rest_and_native_impulses_match_independent_solver(self):
        model=CalciumTransformPWA.load(ROOT/'models/positive_coordinates/sqrt_tails/FM.json')
        rest,count=stable_rest(model,.45,[.0425,.911,.45])
        self.assertEqual(count,1)
        np.testing.assert_allclose(model.field(model.to_latent(rest)),0.,atol=1e-12)
        kwargs=dict(duration=8.,tau=5.,baseline=.45,jumps=[(2.,.3),(5.,.15)],dt=.05)
        t,x,_=native_events(model,rest,**kwargs,maxdt=.0005)
        tr,truth,_=simulate_coordinates(model,rest,**kwargs,rtol=2e-11,atol=2e-13,max_step=.03)
        np.testing.assert_allclose(t,tr,atol=1e-12)
        np.testing.assert_allclose(x,truth,atol=2e-6)
        np.testing.assert_allclose(x[t<2],np.broadcast_to(rest,(sum(t<2),3)),atol=1e-12)

    def test_event_score_preserves_height_and_latency_information(self):
        t=np.linspace(0,80,801);c=.05+np.exp(-((t-30)/2)**2)+np.exp(-((t-55)/2)**2)
        truth=np.column_stack([c,np.full(len(t),.8),np.full(len(t),.5)])
        np.testing.assert_allclose(event_residual(t,truth,t,truth),0.,atol=1e-12)
        delayed=truth.copy();delayed[:,0]=np.interp(t-2,t,c)
        self.assertGreater(np.linalg.norm(event_residual(t,delayed,t,truth)),1.)
        taller=truth.copy();taller[:,0]*=1.2
        self.assertGreater(np.linalg.norm(event_residual(t,taller,t,truth)),1.)


if __name__=='__main__':unittest.main()
