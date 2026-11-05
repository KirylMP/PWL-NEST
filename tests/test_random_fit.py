import copy
import json
import unittest
from pathlib import Path

import numpy as np
from check_focus_feasibility import constraints
from find_random_fold_seed import draw_start

ROOT = Path(__file__).resolve().parents[1]


class RandomStarts(unittest.TestCase):
    def test_reproducible_and_independent_of_fitted_cycle(self):
        for case in ('AM', 'AFM', 'FM'):
            base = json.loads((ROOT / 'paper/data/physical_models/fixed_fold' / case / 'parameters.json').read_text())
            x, t, draws = draw_start(base, 0, case, 5000, 40., 1e-4)
            altered = copy.deepcopy(base)
            altered['fold_constraint']['crossing_times'] = [999, 999, 999, 999]
            x2, t2, draws2 = draw_start(altered, 0, case, 5000, 40., 1e-4)
            np.testing.assert_array_equal(x, x2)
            np.testing.assert_array_equal(t, t2)
            self.assertEqual(draws, draws2)
            self.assertGreaterEqual(min(constraints(base, x, 1e-4)), 0)
            self.assertTrue(np.all(np.abs(x) <= 40))
            self.assertTrue(np.all((t >= .05) & (t <= 50)))
            x3, _, _ = draw_start(base, 1, case, 5000, 40., 1e-4)
            self.assertFalse(np.array_equal(x, x3))


if __name__ == '__main__':
    unittest.main()
