import unittest
import json
import sys
from unittest.mock import patch
import numpy as np
from pwl_nest.equilibrium_constraints import anchor_equilibrium
from pwl_nest.rest_protocols import stable_rest
from pwl_nest.calcium_coordinates import CalciumTransformPWA,simulate_coordinates
from pwl_nest.models import ROOT
sys.path.insert(0,str(ROOT/'scripts'))
from fit_anchored_shapes import parameterization,translated_start,ShapeObjective,shape_checks
from fit_positive_coordinates import fit as legacy_fit
from fit_mixed_coordinates import parameterization as legacy_parameterization


class EquilibriumConstraintsTest(unittest.TestCase):
    def test_exact_rest_preserves_all_regional_matrices_and_source(self):
        model=CalciumTransformPWA.load(ROOT/'models/server_candidates/FM_identity__identity.json')
        model.mix_linear=-.03;model.mix_quadratic=-.005
        target=np.array([.04252319858338,.911256065186791,.45])
        model.mix_center=float(model.gate.forward(target[1]))
        original=model.latent.matrix.copy()
        anchored=anchor_equilibrium(model,target)
        np.testing.assert_array_equal(model.latent.matrix,original)
        np.testing.assert_allclose(anchored.field(anchored.to_latent(target)),0,atol=5e-16)
        np.testing.assert_array_equal(anchored.latent.local_coefficients()[:,:,1:],model.latent.local_coefficients()[:,:,1:])
        np.testing.assert_array_equal(anchored.latent.thresholds,model.latent.thresholds)
        rest,_=stable_rest(anchored,.45,target)
        np.testing.assert_allclose(rest,target,atol=2e-14)
        _,x,_=simulate_coordinates(anchored,target,20.,baseline=.45,dt=1.)
        np.testing.assert_allclose(x,np.broadcast_to(target,x.shape),atol=1e-13)

    def test_fitted_parameters_keep_rest_stable_with_two_constants_removed(self):
        seed=CalciumTransformPWA.load(ROOT/'models/server_candidates/FM_identity__identity.json')
        spec=json.loads((ROOT/'results/fm_mixed_coordinates/cache/FM/data.json').read_text())
        rng=np.random.default_rng(73)
        for family,expected in [('control',27),('linear',28),('quadratic',29)]:
            z,lo,hi,active,m,make=parameterization(seed,spec,family)
            self.assertEqual(len(z),expected)
            self.assertNotIn(16,active);self.assertNotIn(21,active)
            for _ in range(3):
                p=np.clip(z+rng.normal(0,.01,len(z)),lo+1e-8,hi-1e-8)
                model=make(p);latent=model.to_latent(spec['rest'])
                self.assertLess(latent[0],model.latent.thresholds[0])
                np.testing.assert_allclose(model.field(latent),0,atol=5e-15)
                rest,_=stable_rest(model,.45,spec['rest'])
                np.testing.assert_allclose(rest,spec['rest'],atol=5e-13)

    def test_translated_partition_preserves_original_latent_orbits(self):
        seed=CalciumTransformPWA.load(ROOT/'models/server_candidates/FM_identity__identity.json')
        spec=json.loads((ROOT/'results/fm_mixed_coordinates/cache/FM/data.json').read_text())
        z,lo,hi,active,m,make=parameterization(seed,spec,'control')
        p=translated_start(seed,spec,z,active,m,make);model=make(p)
        rest,_=stable_rest(seed,.45,spec['rest'])
        shift=seed.to_latent(rest)-model.to_latent(spec['rest'])
        y=seed.to_latent(np.array([[.05,.9,.5],[.2,.7,.8],[1.,.55,1.1]]))
        np.testing.assert_allclose(model.latent.thresholds,seed.latent.thresholds-shift[0],atol=1e-15)
        np.testing.assert_allclose(model.field(y-shift),seed.field(y),atol=2e-14)

    def test_failed_simulations_keep_optimizer_residual_dimension(self):
        seed=CalciumTransformPWA.load(ROOT/'models/server_candidates/FM_identity__identity.json')
        objective=ShapeObjective(ROOT/'results/fm_mixed_coordinates','FM',seed)
        valid=objective(seed)
        with patch.object(objective.dynamic,'run',return_value=None):failed=objective(seed)
        self.assertEqual(valid.shape,failed.shape)
        self.assertTrue(np.isfinite(failed).all())

    def test_phase_weight_changes_only_absolute_timing_terms(self):
        seed=CalciumTransformPWA.load(ROOT/'models/continued_shape_candidates/FM_control.json')
        objective=ShapeObjective(ROOT/'results/fm_shift_continuation','FM',seed)
        original,parts=objective(seed,parts=True)
        objective.timing_weight=3.
        weighted,new_parts=objective(seed,parts=True)
        self.assertEqual(original.shape,weighted.shape)
        for key,value in parts.items():
            self.assertAlmostEqual(new_parts[key],value*(9. if key.endswith('_timing') else 1.),places=7)
        checks_before=shape_checks(objective,seed)
        objective.timing_weight=1.;objective(seed)
        self.assertEqual(checks_before,shape_checks(objective,seed))

    def test_legacy_fitters_cannot_silently_unfix_rest(self):
        seed=CalciumTransformPWA.load(ROOT/'models/server_candidates/FM_identity__identity.json')
        anchored=anchor_equilibrium(seed,[.04252319858338,.911256065186791,.45])
        with self.assertRaisesRegex(ValueError,'preserve the exact rest'):
            legacy_fit('FM','control',anchored,None,None,1)
        with self.assertRaisesRegex(ValueError,'preserve the exact rest'):
            legacy_parameterization(anchored,None,'control')
