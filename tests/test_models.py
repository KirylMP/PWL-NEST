import unittest
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from pwl_nest.models import load_model, coefficients, pwa_rhs, lr_rhs
from pwl_nest.native import observables

class ModelsTest(unittest.TestCase):
    def test_am_boundary_continuity(self):
        model = load_model('AM')
        for ip3 in [0., .35, .5, .67, .8, 1.3]:
            theta, mu, k, q, g = coefficients(model, ip3)
            np.testing.assert_allclose(mu[:-1]*theta+g[:-1],mu[1:]*theta+g[1:],atol=1e-13)
            np.testing.assert_allclose(q[:-1]-k[:-1]*theta,q[1:]-k[1:]*theta,atol=1e-13)

    def test_fm_discontinuity_is_explicit(self):
        theta, mu, k, q, _ = coefficients(load_model('FM'), .8)
        self.assertGreater(np.max(abs((q[:-1]-k[:-1]*theta)-(q[1:]-k[1:]*theta))),1e-4)

    def test_regional_affine_flow(self):
        for case in ['AM','FM']:
            model = load_model(case)
            theta, mu, k, q, g = coefficients(model,.5)
            c = (theta[-2]+theta[-1])/2
            j = len(theta)-1
            if case=='AM':
                a = np.array([[mu[j],1,g[j]],[-k[j],0,q[j]],[0,0,0]])
                y = np.array([c,-mu[j]*c-g[j]])
            else:
                a = np.array([[0,1,0],[-k[j],mu[j],q[j]],[0,0,0]])
                y = np.array([c,0.])
            sol = solve_ivp(pwa_rhs(model,.5),(0,.001),y,rtol=1e-12,atol=1e-14)
            np.testing.assert_allclose(sol.y[:,-1],(expm(a*.001)@np.r_[y,1])[:2],atol=1e-11)

    def test_native_cycles_and_rest(self):
        for case, cycle, rest in [('AM',.5,.8),('FM',.8,1.2)]:
            x = observables(case,cycle)
            self.assertTrue(np.isfinite(x['period']))
            self.assertGreater(x['Cmax']-x['Cmin'],.01)
            y = observables(case,rest)
            self.assertLess(y['Cmax']-y['Cmin'],1e-4)

    def test_domain(self):
        for bad in [-.1,1.31,float('nan')]:
            with self.assertRaises(ValueError):coefficients(load_model('AM'),bad)

if __name__=='__main__':unittest.main()
