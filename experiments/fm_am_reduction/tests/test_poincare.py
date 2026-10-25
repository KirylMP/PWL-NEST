import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from poincare import Section,first_return,smooth_counts,isolate_cycles,PWASystem

class Radial:
    nparams=0
    bounds=np.array([[-1.2,1.2],[-1.2,1.2]])
    def coordinate(self,c):return c
    def coordinate_gradient(self,c):return np.empty(0)
    def physical(self,z):return z
    def field(self,y):
        x,h=y;s=x*x+h*h;q=(s-.09)*(.49-s)
        return np.array([-h+x*q,x+h*q])
    def jac(self,y):
        x,h=y;s=x*x+h*h;q=(s-.09)*(.49-s);dq=.58-2*s
        return np.array([[q+2*x*x*dq,-1+2*x*h*dq],[1+2*x*h*dq,q+2*h*h*dq]])
    def partials(self,y):return np.empty((2,0))

class ReturnChecks(unittest.TestCase):
    def test_first_return_is_full_period_and_multiplier(self):
        rec=first_return(Radial(),Section(0,0,-1),.3,max_time=20,max_step=.15)
        self.assertEqual(rec['status'],'valid');self.assertAlmostEqual(rec['period'],2*np.pi,places=6)
        self.assertAlmostEqual(rec['P'],.3,places=6)
        expected=np.exp(2*.09*(.49-.09)*2*np.pi)
        self.assertAlmostEqual(rec['multiplier'],expected,places=5)

    def test_repelling_and_attracting_cycles_enumerated(self):
        records=isolate_cycles(Radial(),Section(0,0,-1),np.linspace(.1,.95,19),dict(max_time=30,max_step=.2))
        self.assertEqual(sorted(q['kind'] for q in records['cycles']),['stable','unstable'])
        np.testing.assert_allclose(sorted(q['s'] for q in records['cycles']),[.3,.7],atol=1e-6)
        counts,_,_=smooth_counts(records['records'],.001)
        np.testing.assert_allclose(counts,[1,1],atol=.02)

    def test_invalid_nodes_do_not_form_edges(self):
        records=[dict(status='valid',G=2),dict(status='timeout'),dict(status='valid',G=-2)]
        counts,_,edges=smooth_counts(records,.02)
        self.assertEqual(edges,0);np.testing.assert_array_equal(counts,[0,0])
        # Callers must inspect edges=0 and mark the domain unresolved.

    def test_independent_integrators_agree_on_pwa_orbit(self):
        theta=np.r_[[1.5,2,0,-2.5,2.5],[.5,-1,0,0,0],0,np.log(.1),2,-2,0,0]
        system=PWASystem(theta,np.array([.3,.7]),.7,.5,1.68)
        a=first_return(system,Section(.55,.5),.4,max_time=100,rtol=2e-10,atol=2e-12)
        b=first_return(system,Section(.55,.5),.4,max_time=100,rtol=2e-10,atol=2e-12,method='RK45')
        self.assertEqual(a['status'],'valid');self.assertEqual(b['status'],'valid')
        np.testing.assert_allclose([a['P'],a['period'],a['Cmin'],a['Cmax']],
                                   [b['P'],b['period'],b['Cmin'],b['Cmax']],rtol=2e-6,atol=2e-7)

    def test_counter_derivative(self):
        x=.13;nodes=np.linspace(-1,1,17)
        def records(v):return [dict(status='valid',G=t-v,dG=[-1.]) for t in nodes]
        count,jac,_=smooth_counts(records(x),.2,nparams=1)
        step=1e-6
        numeric=(smooth_counts(records(x+step),.2)[0]-smooth_counts(records(x-step),.2)[0])/(2*step)
        np.testing.assert_allclose(jac[:,0],numeric,atol=1e-8)

    def test_pwa_moving_section_sensitivities(self):
        theta=np.r_[[1.5,2,0,-2.5,2.5],[.5,-1,0,0,0],0,np.log(.1),2,-2,0,0]
        thresholds=np.array([.3,.7]);section=Section(.55,.5)
        def system(v):return PWASystem(v,thresholds,.7,.5,1.68)
        opts=dict(max_time=100,max_step=.1,rtol=2e-9,atol=2e-11)
        result=first_return(system(theta),section,.4,True,**opts)
        self.assertEqual(result['status'],'valid')
        for j in range(len(theta)):
            d=np.eye(len(theta))[j]*2e-5
            plus=first_return(system(theta+d),section,.4,False,**opts)
            minus=first_return(system(theta-d),section,.4,False,**opts)
            self.assertEqual(plus['status'],'valid');self.assertEqual(minus['status'],'valid')
            numeric=(plus['G']-minus['G'])/(4e-5)
            np.testing.assert_allclose(result['dG'][j],numeric,rtol=.01,atol=3e-4,err_msg=f'parameter {j}')

if __name__=='__main__':unittest.main()
