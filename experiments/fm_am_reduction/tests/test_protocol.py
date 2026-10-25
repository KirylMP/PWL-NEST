"""Independent mathematical regression checks for the FM audit corrections."""
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from fit_stage import phi, model_roots, quadrature_weights
from prepare_reference_data import lr_field, jacobian, equilibrium_h, roots_for_ip3
from scipy.optimize import root
from analyze_candidate import orbit, stability
from plan import PACKAGE, read_json

class ProtocolChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cases=read_json(PACKAGE/'config'/'cases.json')
        cls.p={**cases['shared_parameters'],'K_ER':.05}

    def test_signed_cube_root_reproduces_nullcline(self):
        c=.004; i=.6; p=self.p
        rad=(p['v_ER']*c*c/((c*c+p['K_ER']**2)*(p['C_0']-(1+p['c_1'])*c))-p['r_L'])/(p['r_C']*(i/(i+p['d_1']))**3*(c/(c+p['d_5']))**3)
        self.assertLess(rad,0)
        self.assertAlmostEqual(float(lr_field(c,np.cbrt(rad),i,p)[0]),0,places=12)

    def test_reference_jacobian_against_field_difference(self):
        for c,h,i in [(.05,.7,.4),(.3,.3,.8),(.9,.4,1.2)]:
            x=np.array([c,h]); step=1e-6
            numeric=np.column_stack([(np.array(lr_field(*(x+step*np.eye(2)[j]),i,self.p))-np.array(lr_field(*(x-step*np.eye(2)[j]),i,self.p)))/(2*step) for j in range(2)])
            np.testing.assert_allclose(jacobian(c,h,i,self.p),numeric,rtol=1e-7,atol=1e-8)

    def test_tangent_equilibrium_at_fold_is_retained(self):
        p=self.p
        def equations(v):
            c,i=v;h=equilibrium_h(c,i,p);q=p['d_2']*(i+p['d_1'])/(i+p['d_3'])
            j=jacobian(c,h,i,p)
            return [lr_field(c,h,i,p)[0],j[0,0]-j[0,1]*h/(q+c)]
        answer=root(equations,[.08,.5],tol=1e-11)
        self.assertTrue(answer.success)
        found=roots_for_ip3(float(answer.x[1]),p,p['C_0']/(1+p['c_1']),1600)
        self.assertTrue(any(abs(q['C']-answer.x[0])<1e-7 and abs(q['determinant'])<1e-7 for q in found))

    def test_transform_normalization_and_monotonicity(self):
        for eta in [np.zeros(4),np.array([3.,-2.,.8,1.2]),np.array([-7.,-7.,0.,0.])]:
            self.assertAlmostEqual(float(phi(np.array([.5]),eta,.5)[0]),0)
            slope=(phi(np.array([.500001]),eta,.5)-phi(np.array([.499999]),eta,.5))[0]/.000002
            self.assertAlmostEqual(slope,2,places=7)
            self.assertTrue(np.all(np.diff(phi(np.linspace(.01,1.6,100),eta,.5))>0))

    def test_exponential_coordinate_derivative_at_zero_k(self):
        eta=np.zeros(4);step=1e-8;delta=np.array([0.,0.,0.,step])
        numeric=(phi(np.array([1.]),eta+delta,.5)-phi(np.array([1.]),eta-delta,.5))/(2*step)
        self.assertAlmostEqual(float(numeric[0]),1/6,places=7)

    def test_nonuniform_quadrature_integrates_linear(self):
        x=np.array([.1,.15,.4,1.2]); w=quadrature_weights(x)
        self.assertAlmostEqual(float(w@x), (x[-1]**2-x[0]**2)/2)

    def test_actual_model_root_in_physical_coordinates(self):
        theta=np.r_[[0,1,0,0],[.3,0,0,0],0,0,25,-25,0,0]
        roots=model_roots(theta,np.array([.3]),.6,.5,.001,1.6)
        self.assertEqual(len(roots),1)
        np.testing.assert_allclose(roots[0],[.65,.3],atol=1e-9)

    def test_zero_trace_saddle_is_hyperbolic(self):
        self.assertEqual(stability(dict(trace=0.,determinant=-1.)),"saddle")

    def test_orbit_statuses(self):
        periodic=orbit(lambda t,y: [-(y[1]-.5),y[0]-.5],[.6,.5],120,[.001,.999])
        self.assertEqual(periodic['status'],'observed_periodic_orbit')
        self.assertAlmostEqual(periodic['period'],2*np.pi,delta=.05)
        rest=orbit(lambda t,y: [.5-y[0],.5-y[1]],[.6,.6],80,[.001,.999])
        self.assertEqual(rest['status'],'rest')

if __name__=='__main__':unittest.main()
