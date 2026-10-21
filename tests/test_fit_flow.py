"""The fitting backend must reproduce physical trajectories, not just loss values."""
import unittest
import numpy as np
from pwl_nest.models import ROOT
from pwl_nest.calcium_coordinates import CalciumTransformPWA,simulate_coordinates
from pwl_nest.fit_flow import FlowDynamic
from pwl_nest.rest_protocols import stable_rest,native_events


@unittest.skipUnless((ROOT/'.build/fit_flow.so').exists(),'Build the fitting flow backend')
class FitFlowTests(unittest.TestCase):
    def test_periodic_inputs_and_input_regions_match_dop853(self):
        for gate in ['cube','log']:
            with self.subTest(gate=gate):
                model=CalciumTransformPWA.load(ROOT/f'models/server_candidates/AM_sqrt_tails__{gate}.json')
                rest,_=stable_rest(model,.3,[.1,.8,.3])
                protocol=dict(duration=150.,tau=20.,baseline=.3,jumps=[[t,.16] for t in [20.,40.,60.,80.,100.]])
                t,x,_=native_events(model,rest,**protocol,dt=.25,maxdt=.02,native=FlowDynamic())
                tr,xr,_=simulate_coordinates(model,rest,**protocol,dt=.25,rtol=2e-11,atol=2e-13,max_step=.02)
                np.testing.assert_allclose(t,tr,rtol=0,atol=1e-12)
                np.testing.assert_allclose(x,xr,rtol=0,atol=3e-6)

    def test_full_fixed_duration_trajectory_matches_dop853(self):
        model=CalciumTransformPWA.load(ROOT/'models/server_candidates/AM_sqrt_tails__cube.json')
        initial=[.6,.8,.638]
        y=FlowDynamic().run(model,initial,100.,20.,.638,outputs=501,maxdt=.02)
        t=np.linspace(0,100,501)
        actual=model.to_physical(np.column_stack([y,np.full(len(t),.638)]))
        tr,expected,_=simulate_coordinates(model,initial,100.,20.,.638,dt=.2,
            rtol=2e-11,atol=2e-13,max_step=.02)
        np.testing.assert_allclose(t,tr,rtol=0,atol=1e-12)
        np.testing.assert_allclose(actual,expected,rtol=0,atol=3e-6)


if __name__=='__main__':unittest.main()
