import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from full_cycle_fit import FullCycleObjective
from poincare import smooth_counts
from fit_stage import model_roots,NonisolatedEquilibria

class Geometry:
    def residual(self,x):return np.array([x[0]-.1])
    def parts(self,x):return dict(nullclines=float((x[0]-.1)**2),precycle_total=float((x[0]-.1)**2))

class FullLossChecks(unittest.TestCase):
    def fixture(self):
        obj=FullCycleObjective.__new__(FullCycleObjective)
        obj.geometry=Geometry();obj.run={'parameters':{'C_0':2.,'c_1':.185}}
        obj.thresholds=np.array([.3,.7]);obj.c_ref=.5
        obj.nodes=[.1,.4,.8];obj.epsilon=.15;obj.last_x=None
        obj.spec={'cycle_smoothing':{'xi':1e-7,'softplus_tau':.02},'loss_weights':{'cycles':30.,'extra_cycles':30.}}
        obj.cfg={'min_regular_edge_fraction':.5}
        obj.rows=[dict(I=i,section={'C':.5,'h0':.4},records=[dict(status='valid',G=g) for g in [.3,-.05,.1]]) for i in [.6,.8]]
        theta=np.r_[[.12,2,0,-2.5,2.5],[.5,-1,0,0,0],0,np.log(.1),2,-2,0,0]
        def table(system,*args,**kwargs):
            result=[]
            for g,d in zip([.3,-.05,.1],[-1.,.3,.5]):
                grad=np.zeros(len(theta));grad[0]=d
                result.append(dict(status='valid',G=g+system.theta[0]*d,dG=grad.tolist()))
            return result
        return obj,theta,table

    def test_five_term_normalization_and_gradient(self):
        obj,x,table=self.fixture()
        with patch('full_cycle_fit.return_table',table):
            r=obj.residual(x).copy();j=obj.jac(x).copy()
            step=1e-6;d=np.eye(len(x))[0]*step
            numeric=(obj.residual(x+d)-obj.residual(x-d))/(2*step)
            np.testing.assert_allclose(j[:,0],numeric,rtol=1e-5,atol=1e-7)
            counts,_,_=smooth_counts(table(type('S',(),{'theta':x})()),.15,nparams=len(x))
            truth,_,_=smooth_counts(obj.rows[0]['records'],.15)
            expected=(x[0]-.1)**2+30*np.sum((counts-truth)**2)
            sp=lambda v:.02*np.logaddexp(0,v/.02)
            expected+=30*len(obj.rows)*(sp(counts.sum()-2)-sp(truth.sum()-2))**2
            self.assertAlmostEqual(float(r@r),expected,places=9)
            self.assertTrue(obj.parts(x)['return_domain_admissible'])

    def test_missing_returns_are_inadmissible_not_zero_cycles(self):
        obj,x,_=self.fixture()
        with patch('full_cycle_fit.return_table',return_value=[dict(status='timeout')]*3):
            obj.residual(x)
            self.assertFalse(obj.parts(x)['return_domain_admissible'])
            self.assertIsNone(obj.report[0]['counts'])

    def test_nonphysical_equilibrium_retained(self):
        theta=np.r_[[0,1,0,0],[3.,0,0,0],0,0,25,-25,0,0]
        roots=model_roots(theta,np.array([.3]),.6,.5,.001,1.6)
        self.assertEqual(len(roots),1);self.assertGreater(roots[0][0],1.6)

    def test_coincident_graphs_flagged(self):
        theta=np.r_[[0,1,0,0],[0,1,0,0],0,0,0,0,0,0]
        with self.assertRaises(NonisolatedEquilibria):model_roots(theta,np.array([.3]),.6,.5,.001,1.6)

if __name__=='__main__':unittest.main()
