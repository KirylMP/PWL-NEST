import unittest
import numpy as np
from scipy.integrate import solve_ivp
from pwl_nest.calcium_strips import decode,NativeDynamic,NativeStatistics
from pwl_nest.strict_pwa import regional_flow


class CalciumStripsTest(unittest.TestCase):
    def setUp(self):
        self.theta=np.array([.1,.2,.4])
        mu=np.array([-.3,.2,.3,-.4]);k=np.array([.2,.1,-.1,.3])
        z=np.r_[np.log(abs(mu)),np.log(abs(k)),.2,np.log(.1),np.log(.5),np.log(.3),.1,-.2]
        self.model=decode('AM',self.theta,z,np.sign(mu),np.sign(k))

    def test_continuity_on_whole_planes(self):
        for c in self.theta:
            for h,i in [(.2,0),(.8,.4),(1,1.3)]:
                x=np.array([c,h,i]);eps=np.array([1e-10,0,0])
                a,b,_=self.model.region(x-eps,20,.2)
                ap,bp,_=self.model.region(x+eps,20,.2)
                np.testing.assert_allclose(a@x+b,ap@x+bp,atol=1e-14)

    def test_shared_columns_and_exact_spectrum(self):
        local=self.model.local_coefficients()
        for p in local:
            np.testing.assert_array_equal(p[:,2:],local[0,:,2:])
        mu,k,q,qb=self.model.spectral_coefficients()
        np.testing.assert_allclose(mu,[-.3,.2,.3,-.4],atol=1e-14)
        np.testing.assert_allclose(k,[.2,.1,-.1,.3],atol=1e-14)
        for j,p in enumerate(local):
            c=.05 if j==0 else self.theta[j-1]+.01
            x=[c,.6,.7]
            f=self.model.field(x)
            direct=p[:,1:3]@f+p[:,3]*0
            self.assertAlmostEqual(direct[0],mu[j]*f[0]-k[j]*c+q[j]+qb*.7,places=13)

    def test_exact_regional_dynamic_I(self):
        x=np.array([.15,.6,.7])
        a,b,_=self.model.region(x,2,.2)
        exact=regional_flow(a,b,x,1e-3)
        sol=solve_ivp(self.model.rhs(2,.2),(0,1e-3),x,rtol=1e-12,atol=1e-14)
        np.testing.assert_allclose(exact,sol.y[:,-1],atol=1e-13)
        self.assertAlmostEqual(exact[2],.2+.5*np.exp(-1e-3/2),places=14)

    def test_native_RK_matches_numerical(self):
        initial=[.15,.6,.7]
        y=NativeDynamic().run(self.model,initial,2.,2.,.2,outputs=41,maxdt=.002)
        self.assertIsNotNone(y)
        sol=solve_ivp(self.model.rhs(2,.2),(0,2),initial,t_eval=np.linspace(0,2,41),
                      rtol=1e-11,atol=1e-13,max_step=.01)
        np.testing.assert_allclose(y,sol.y[:2].T,atol=2e-7)

    def test_hard_subtypes(self):
        mu=np.array([-.5,-.2,.4,-.6])
        z=np.r_[np.log(abs(mu)),[-1.,-3.,-2.,.5],.2,np.log(.1),np.log(.5),np.log(.3),.1,-.2]
        model=decode('AM',self.theta,z,np.sign(mu),np.ones(4),
                     dict(enforce_types=True,target_types=['stable_node','stable_focus','unstable_focus','stable_node']))
        tr,det,_,_=model.spectral_coefficients()
        self.assertTrue(((tr*tr-4*det)*np.array([1,-1,-1,1])>0).all())

    def test_native_statistics_after_crossings(self):
        mu=np.full(4,-.3);k=np.full(4,.2)
        z=np.r_[np.log(abs(mu)),np.log(k),.4,np.log(.1),np.log(.5),np.log(.3),.1,-.2]
        model=decode('AM',self.theta,z,np.sign(mu),np.ones(4))
        obs,diagnostics=NativeStatistics().run(model,.7,[.15,.6,.7],duration=1000.,diagnostics=True)
        equilibrium=np.linalg.solve(model.local_coefficients()[-1,:,1:3],
                                    -model.local_coefficients()[-1,:,0]-model.local_coefficients()[-1,:,3]*.7)
        self.assertAlmostEqual(obs[0],equilibrium[0],places=9)
        self.assertAlmostEqual(obs[1],equilibrium[0],places=9)
        self.assertTrue(np.isnan(obs[2]))
        self.assertGreaterEqual(diagnostics['max_h'],.6)

if __name__=='__main__':unittest.main()
