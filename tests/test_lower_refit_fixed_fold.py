import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from fit_lower_refit_fixed_fold import fold_solution, PIVOTS
from continue_physical_fixed import Orbit


class FoldRefitTest(unittest.TestCase):
    def test_baselines_and_perturbed_field_satisfy_fold_equations(self):
        for case in ('AM', 'AFM', 'FM'):
            with self.subTest(case=case):
                d = json.loads((ROOT/'paper/data/physical_models/fixed_fold'/case/'parameters.json').read_text())
                f = d['fold_constraint']; cut = d['physical_thresholds_C'][0]*.2
                for change in ({}, {('beta', 2): 1e-5}):
                    changed, orbit = fold_solution(d, change, PIVOTS[case], cut)
                    self.assertLess(orbit['matching_max'], 2e-7)
                    self.assertLess(abs(orbit['logM']), 2e-7)
                    self.assertGreater(orbit['Cmin_sampled'], cut)
                    self.assertAlmostEqual(orbit['I'], f['I'])
                    independent = Orbit(changed).point(np.array(orbit['times']), orbit['I'], 1e-9, require_physical=False)
                    self.assertLess(independent['closure'], 1e-5)


if __name__ == '__main__': unittest.main()
