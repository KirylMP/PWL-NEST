import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from fit_lower_fixed_fold import make_model, rhs


class LowerZoneTest(unittest.TestCase):
    def test_field_correction_and_spectral_metadata(self):
        base = {'physical_thresholds_C': [.2, .7], 'alpha': [1., 2., 3., 4., 5.],
                'beta': [.2, .4, .6, .8, 1.], 'a': 1.3, 'b': .7}
        cut, da, db = .1, .3, -.8
        model = make_model(base, cut, da, db)
        for c in [-.1, 0., cut, .15, .5, .9]:
            for q, i in [(0., 0.), (.5, .3), (1., 1.)]:
                actual = np.asarray(rhs(model, i, [c, q])) - rhs(base, i, [c, q])
                expected = np.array([-base['a'] * da, base['b'] * db]) * max(cut-c, 0.)
                np.testing.assert_allclose(actual, expected, atol=2e-15)
        self.assertEqual(len(model['trace']), 4)
        self.assertEqual(len(model['determinant']), 4)


if __name__ == '__main__':
    unittest.main()
