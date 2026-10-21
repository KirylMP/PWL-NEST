import unittest
import numpy as np
from pwl_nest.models import ROOT
from pwl_nest.calcium_coordinates import CalciumTransformPWA
from pwl_nest.reduced_thresholds import ReducedFamily
from pwl_nest.protocols import simulate
from pwl_nest.strict_pwa import C_MAX,equilibrium_h,lr_field
from scipy.optimize import brentq


class ReducedThresholdTests(unittest.TestCase):
    def test_two_calcium_thresholds_with_both_input_geometries(self):
        source=CalciumTransformPWA.load(ROOT/'models/am_baseline_refit/AM_sqrt_tails__cube.json')
        c=brentq(lambda c:lr_field([c,equilibrium_h(c,.3),.3],'AM')[0],1e-5,1.)
        rest=np.array([c,equilibrium_h(c,.3),.3])
        _,points,_=simulate('AM',rest,50.,tau=20.,baseline=.3,jumps=[[20.,.35]],dt=.2)
        for keep in [False,True]:
            f=ReducedFamily(source,np.array(source.metadata['physical_thresholds_C'])[[1,2]],rest,points,
                            .05563833512124773,.51941408164821,keep_input=keep)
            model=f.make(np.zeros(f.size))
            self.assertEqual(model.latent.region_count,3)
            self.assertEqual(len(model.input_thresholds),2 if keep else 0)
            self.assertEqual(f.size,16 if keep else 12)
            np.testing.assert_allclose(model.field(model.to_latent(rest)),0.,atol=1e-12)
            y=model.to_latent(rest)
            a,_,_=model.region(y,20.,.3)
            self.assertLess(np.linalg.eigvals(a[:2,:2]).real.max(),0.)
            # Region matrices must agree on every C face for arbitrary h and I.
            local=model.geometry_coefficients()
            for band in range(len(model.input_thresholds)+1):
                for j,t in enumerate(model.latent.thresholds):
                    for v in [-1.,0.,1.]:
                        q=np.array([1.,t,v,.6])
                        np.testing.assert_allclose(local[band,j]@q,local[band,j+1]@q,atol=1e-11)
            # Threshold changes remain ordered and preserve the exact prescribed rest.
            z=np.zeros(f.size);z[-2:]=[.05,-.05]
            changed=f.make(z,check_rest=False)
            self.assertTrue(np.all(np.diff(changed.latent.thresholds)>0))
            np.testing.assert_allclose(changed.field(changed.to_latent(rest)),0.,atol=1e-12)


if __name__=='__main__':unittest.main()
