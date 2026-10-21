import unittest

import numpy as np

from pwl_nest.calcium_coordinates import CalciumTransformPWA, simulate_coordinates
from pwl_nest.event_solver import EventSolver


class EventSolverTests(unittest.TestCase):
    def test_exact_region_flow_and_localized_thresholds_match_dop853(self):
        model=CalciumTransformPWA.load('models/phase_shape_candidates/FM_width_soft_candidate.json')
        initial=np.asarray(model.metadata['exact_rest_constraint'],dtype=float)
        protocol=dict(duration=220.,tau=20.,baseline=.45,jumps=[[20.,.7]])
        t,x,counters=EventSolver().run(model,initial,**protocol,output_dt=.05,scan_dt=.02)
        td,xd,_=simulate_coordinates(model,initial,**protocol,dt=.005,rtol=2e-11,atol=2e-13,max_step=.02)
        reference=np.column_stack([np.interp(t,td,xd[:,d]) for d in range(3)])
        np.testing.assert_allclose(x[:,0],reference[:,0],rtol=0.,atol=5e-6)
        np.testing.assert_allclose(x[:,1],reference[:,1],rtol=0.,atol=1e-6)
        self.assertGreater(counters['calcium_region_crossings'],0)
        self.assertEqual(counters['ip3_jumps'],1)
        self.assertEqual(counters['affine_flows']>0,True)


if __name__=='__main__':
    unittest.main()
