import unittest
import numpy as np
from pwl_nest.thresholds import OrderedThresholds
from pwl_nest.models import ROOT
from pwl_nest.calcium_coordinates import CalciumTransformPWA,decode_coordinates


class OrderedThresholdsTest(unittest.TestCase):
    def test_reference_roundtrip_and_order_at_box_corners(self):
        ref=np.array([.0486,.0521,.0548,.0613,.1357,.2518,.3926])
        mapping=OrderedThresholds(ref,.028,.55)
        np.testing.assert_array_equal(mapping.decode(np.zeros(7)),ref)
        rng=np.random.default_rng(24)
        for shifts in rng.choice([-np.log(2),np.log(2)],size=(128,7)):
            theta=mapping.decode(shifts)
            self.assertTrue(np.all(np.diff(np.r_[mapping.lower,theta,mapping.upper])>0))
            np.testing.assert_allclose(mapping.encode(theta),shifts,atol=2e-14)
        np.testing.assert_array_equal(ref,mapping.reference)
        with self.assertRaises(ValueError):mapping.reference[0]=.03

    def test_invalid_order_and_collapsed_regions(self):
        for reference in [[],[.2,.1],[.1,.1],[0.,.1]]:
            with self.assertRaises(ValueError):OrderedThresholds(reference,0.,1.)
        mapping=OrderedThresholds([.1,.2],0.,1.)
        with self.assertRaises(ValueError):mapping.encode([.3,.2])
        with self.assertRaises(ValueError):mapping.decode([1.])
        with self.assertRaises(ValueError):mapping.decode([1e6,-1e6])

    def test_moved_thresholds_preserve_continuity_and_regional_spectra(self):
        seed=CalciumTransformPWA.load(ROOT/'models/coordinate_families/selected_AM.json')
        ref=np.array(seed.metadata['physical_thresholds_C']);mapping=OrderedThresholds(ref,.055,.55)
        theta=seed.coordinate.forward(mapping.decode([.4,-.4,.2]));m=seed.metadata
        moved=decode_coordinates(seed.case,theta,m['parameters'],m['mu_sign'],m['k_sign'],m.copy(),seed.coordinate,seed.h_reference,seed.h_kind)
        local=moved.latent.local_coefficients()
        for j,t in enumerate(theta):
            for v,ip3 in [(-2.,0.),(.2,.6),(2.,1.3)]:
                state=np.array([1.,t,v,ip3])
                np.testing.assert_allclose(local[j]@state,local[j+1]@state,atol=2e-14)
        for current,original in zip(moved.latent.spectral_coefficients()[:2],seed.latent.spectral_coefficients()[:2]):
            np.testing.assert_allclose(current,original,atol=2e-14)


if __name__=='__main__':unittest.main()
