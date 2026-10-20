import unittest
import numpy as np
from scipy.integrate import quad
from pwl_nest.impulsive_ip3 import exponential_ip3,regular_ip3_velocity
from pwl_nest.strict_pwa import regional_flow
from pwl_nest.calcium_coordinates import CalciumTransformPWA,simulate_coordinates
from pwl_nest.models import ROOT


class ImpulsiveIP3Test(unittest.TestCase):
    def test_superposition_matches_jumps_and_regular_derivative(self):
        events=[(1.,.15),(3.,.12),(5.,-.05)];tau=2.;baseline=.2
        model=CalciumTransformPWA.load(ROOT/'models/calcium_powers/selected_AM.json')
        t,x,meta=simulate_coordinates(model,[.08,.8,.4],8.,tau=tau,baseline=baseline,jumps=events,dt=.02)
        exact=exponential_ip3(t,.4,tau,baseline,events,side='left')
        np.testing.assert_allclose(x[:,2],exact,atol=2e-13)
        for event in meta['jumps']:
            before=exponential_ip3(event['time'],.4,tau,baseline,events,side='left')
            after=exponential_ip3(event['time'],.4,tau,baseline,events,side='right')
            self.assertAlmostEqual(float(before),event['physical_before'][2],places=13)
            self.assertAlmostEqual(float(after),event['physical_after'][2],places=13)
        for time in [.5,2.,6.]:
            eps=1e-5
            numerical=(exponential_ip3(time+eps,.4,tau,baseline,events)-exponential_ip3(time-eps,.4,tau,baseline,events))/(2*eps)
            self.assertAlmostEqual(float(numerical),float(regular_ip3_velocity(exponential_ip3(time,.4,tau,baseline,events),tau,baseline)),places=10)

    def test_delta_identity_in_weak_form(self):
        events=[(1.,.15),(3.,.12),(5.,-.05)];tau=2.;baseline=.2;duration=8.
        def integrand(time):
            value=float(exponential_ip3(time,.4,tau,baseline,events))
            phi=time*(duration-time);derivative=duration-2*time
            return -value*derivative+(value-baseline)/tau*phi
        measured=quad(integrand,0,duration,points=[e[0] for e in events],epsabs=1e-12)[0]
        expected=sum(amplitude*time*(duration-time) for time,amplitude in events)
        self.assertAlmostEqual(measured,expected,places=12)

    def test_regular_velocity_coordinate_keeps_affine_regional_flow(self):
        model=CalciumTransformPWA.load(ROOT/'models/calcium_powers/selected_AM.json')
        y=model.to_latent([.2,.75,.6]);tau=7.;baseline=.2
        a,b,_=model.latent.region(y,tau,baseline)
        transform=np.diag([1.,1.,-1/tau]);shift=np.array([0.,0.,baseline/tau])
        anew=transform@a@np.linalg.inv(transform);bnew=transform@b-anew@shift
        for duration in [.01,.1,1.]:
            original=regional_flow(a,b,y,duration)
            changed=regional_flow(anew,bnew,transform@y+shift,duration)
            np.testing.assert_allclose(changed,transform@original+shift,atol=2e-15)
        jump=np.array([0.,0.,.15])
        np.testing.assert_allclose(transform@jump,[0.,0.,-.15/tau],atol=1e-16)

    def test_invalid_and_simultaneous_impulses(self):
        self.assertAlmostEqual(float(exponential_ip3(1.,.2,2.,.2,[(1.,.1),(1.,.2)])),.5)
        for kwargs in [dict(tau=0),dict(tau=2,impulses=[(-1.,.2)]),dict(tau=2,side='middle')]:
            with self.assertRaises(ValueError):exponential_ip3([0.,1.],initial=.2,baseline=.2,**kwargs)


if __name__=='__main__':unittest.main()
