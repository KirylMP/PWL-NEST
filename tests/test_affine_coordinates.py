import unittest
from pathlib import Path
import numpy as np
from pwl_nest.affine_coordinates import recenter_identity
from pwl_nest.calcium_coordinates import CalciumTransformPWA,decode_coordinates

ROOT=Path(__file__).resolve().parents[1]


class AffineCoordinatesTest(unittest.TestCase):
    def test_recenter_preserves_physical_field_surfaces_and_spectrum(self):
        model=CalciumTransformPWA.load(ROOT/'models/anchored_shape_candidates/FM_control.json')
        target=[.04252319858338,.911256065186791,.45]
        shifted=recenter_identity(model,*target[:2],calcium_scale=.5)
        rng=np.random.default_rng(42)
        x=np.column_stack([rng.uniform(.001,1.5,1000),rng.uniform(.01,.99,1000),rng.uniform(.1,1.3,1000)])
        old=model.field(model.to_latent(x));new=shifted.field(shifted.to_latent(x))
        old/=np.column_stack([model.coordinate.derivative(x[:,0]),model.gate.derivative(x[:,1])])
        new/=np.column_stack([shifted.coordinate.derivative(x[:,0]),shifted.gate.derivative(x[:,1])])
        np.testing.assert_allclose(old,new,atol=3e-13,rtol=3e-13)
        np.testing.assert_allclose(shifted.to_latent(target)[:2],0,atol=1e-15)
        np.testing.assert_allclose(model.coordinate.inverse(model.latent.thresholds),
                                   shifted.coordinate.inverse(shifted.latent.thresholds),atol=3e-16)
        for a,b in zip(model.latent.spectral_coefficients()[:2],shifted.latent.spectral_coefficients()[:2]):
            np.testing.assert_allclose(a,b,atol=5e-15)
        # Rebuilding from serialized fit parameters must preserve the new field.
        rebuilt=decode_coordinates(shifted.case,shifted.latent.thresholds,
            np.array(shifted.metadata['parameters']),shifted.metadata['mu_sign'],shifted.metadata['k_sign'],
            shifted.metadata,shifted.coordinate,shifted.h_reference,shifted.h_kind)
        np.testing.assert_allclose(rebuilt.latent.matrix,shifted.latent.matrix,atol=5e-14)

    def test_unsupported_nonlinear_mixing_is_rejected(self):
        model=CalciumTransformPWA.load(ROOT/'models/anchored_shape_candidates/FM_control.json')
        model.mix_quadratic=.01
        with self.assertRaisesRegex(ValueError,'unmixed identity'):
            recenter_identity(model,.04,.91)

    def test_invalid_gate_origins_are_rejected_before_scaling(self):
        model=CalciumTransformPWA.load(ROOT/'models/anchored_shape_candidates/FM_control.json')
        for reference in [0.,1.,float('nan')]:
            with self.assertRaises(ValueError):recenter_identity(model,.04,reference)
