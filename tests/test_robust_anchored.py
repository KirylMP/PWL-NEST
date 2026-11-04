import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from anchored_lower_fold import FREE
from continue_physical_fixed import Orbit
from fit_robust_anchored import PIVOTS, REFERENCE_SEEDS
from robust_anchored_fold import FoldSolver


class RobustAllTest(unittest.TestCase):
    def test_reference_and_neighbors_for_all_cases(self):
        audit=json.loads((ROOT/'paper/data/physical_models/fixed_fold/fold_audit.json').read_text())['results']
        for case in ('AM','AFM','FM'):
            with self.subTest(case=case):
                base=json.loads((ROOT/'paper/data/physical_models/fixed_fold'/case/'parameters.json').read_text())
                path=ROOT/'results/anchored_three_main'/case/f'seed_{REFERENCE_SEEDS[case]}'/'attempt_001/candidate.json'
                fit=json.loads(path.read_text())['anchored_fit'];x0=np.array(fit['best_x']);pivot=PIVOTS[case]
                free=[v for v in FREE if v!=pivot];scale=np.array([.05*max(abs(base[k][j]),.3) for k,j in free])
                cut=.25*min(base['rest_target'][0],base['physical_thresholds_C'][0],audit[case.lower()]['neutral']['Cmin'])
                solver=FoldSolver(base,pivot,cut,x0,fit['fold'])
                for j in (None,0,1,3):
                    x=x0.copy()
                    if j is not None:x[j]-=.01
                    changes={k:float(v*s) for k,v,s in zip(free,x,scale)}
                    three,_,checks,fold=solver.solve(x,changes)
                    self.assertLess(max(map(abs,checks['anchor_residuals'])),1e-9)
                    self.assertLess(fold['matching_max'],2e-7);self.assertLess(abs(fold['logM']),2e-7)
                    independent=Orbit(three).point(np.array(fold['times']),fold['I'],1e-9,require_physical=False)
                    self.assertLess(independent['closure'],1e-5)
                self.assertEqual(solver.failures,0)


if __name__=='__main__':unittest.main()
