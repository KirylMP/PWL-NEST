import itertools
import unittest
import numpy as np
from scipy.integrate import solve_ivp
from pwl_nest.strict_pwa import (StrictPWA,vertices,lr_field,regional_flow,
                                  interpolation_rows,C_MAX)
from pwl_nest.protocols import simulate


class StrictPWATest(unittest.TestCase):
    def setUp(self):
        self.axes = (np.array([0.,.2,C_MAX]),np.array([0.,.4,1.]),np.array([0.,.5,1.3]))
        nodes = vertices(self.axes)
        self.model = StrictPWA('AM',self.axes,lr_field(nodes,'AM'),{})

    def test_all_six_tetrahedra_are_affine(self):
        base = np.array([.0,.0,.0])
        width = np.array([.2,.4,.5])
        rhs = self.model.rhs(7.,.1)
        for perm in itertools.permutations(range(3)):
            u = np.zeros(3)
            u[list(perm)] = [.8,.5,.2]
            point = base+width*u
            a,b,_ = self.model.region(point,7.,.1)
            for scale in [-.01,0.,.01]:
                test = point+scale*width
                np.testing.assert_allclose(a@test+b,rhs(0,test),atol=2e-14)
            self.assertEqual(a[2,0],0.)
            self.assertEqual(a[2,1],0.)
            self.assertEqual(a[2,2],-1/7.)

    def test_continuity_across_cell_and_diagonal_faces(self):
        # Shared faces: sample arbitrarily close from each adjacent side.
        for point,direction in [(np.array([.2,.27,.3]),[1,0,0]),
                                 (np.array([.12,.24,.1]),[1,-1,0]),
                                 (np.array([.1,.1,.25]),[1,0,-1])]:
            direction = np.array(direction)*1e-11
            left_a,left_b,_ = self.model.region(point-direction,10.,.2)
            right_a,right_b,_ = self.model.region(point+direction,10.,.2)
            np.testing.assert_allclose(left_a@point+left_b,right_a@point+right_b,atol=2e-12)

    def test_affine_exponential_and_exact_ip3(self):
        initial = np.array([.12,.2,.1])
        a,b,tetra = self.model.region(initial,7.,.05)
        duration = 1e-4
        exact = regional_flow(a,b,initial,duration)
        numeric = solve_ivp(self.model.rhs(7.,.05),(0,duration),initial,rtol=1e-12,atol=1e-14)
        np.testing.assert_allclose(exact,numeric.y[:,-1],atol=1e-12)
        self.assertAlmostEqual(exact[2],.05+(.1-.05)*np.exp(-duration/7),places=14)

    def test_inward_box_faces(self):
        nodes = vertices(self.axes)
        field = self.model.values
        self.assertTrue((field[nodes[:,0]==0,0]>=0).all())
        self.assertTrue((field[nodes[:,0]==C_MAX,0]<=0).all())
        self.assertTrue((field[nodes[:,1]==0,1]>=0).all())
        self.assertTrue((field[nodes[:,1]==1,1]<=0).all())

    def test_jumps_preserve_C_h(self):
        t,y,meta = simulate('AM',[.08,.8,.2],2.,tau=3.,baseline=.2,jumps=[(1.,.1)],model=self.model,dt=.02)
        before,after = (np.array(meta['jumps'][0][k]) for k in ['before','after'])
        np.testing.assert_array_equal(before[:2],after[:2])
        self.assertAlmostEqual(after[2]-before[2],.1)
        self.assertLess(meta['ip3_max_abs_error'],1e-12)

    def test_no_domain_extrapolation(self):
        for point in [[-.1,.5,.5],[.1,1.1,.5],[.1,.5,1.4]]:
            with self.assertRaises(ValueError):self.model.field(point)

if __name__=='__main__':unittest.main()
