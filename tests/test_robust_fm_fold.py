import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from anchored_lower_fold import FREE
from continue_physical_fixed import Orbit
from robust_anchored_fold import FoldSolver


class RobustFoldTest(unittest.TestCase):
    def test_neighboring_fold_solutions_and_anchors(self):
        base=json.loads((ROOT/'paper/data/physical_models/fixed_fold/FM/parameters.json').read_text())
        audit=json.loads((ROOT/'paper/data/physical_models/fixed_fold/fold_audit.json').read_text())['results']['fm']
        model=json.loads((ROOT/'results/anchored_three_main/FM/seed_8/attempt_001/candidate.json').read_text())
        ref=model['anchored_fit'];x0=np.array(ref['best_x']);pivot=('alpha',4);free=[v for v in FREE if v!=pivot]
        scale=np.array([.05*max(abs(base[k][j]),.3) for k,j in free])
        cut=.25*min(base['rest_target'][0],base['physical_thresholds_C'][0],audit['neutral']['Cmin'])
        solver=FoldSolver(base,pivot,cut,x0,ref['fold'])
        for j in (0,1,3):
            with self.subTest(variable=j):
                x=x0.copy();x[j]-=.01
                changes={k:float(v*s) for k,v,s in zip(free,x,scale)}
                three,_,checks,fold=solver.solve(x,changes)
                self.assertLess(max(map(abs,checks['anchor_residuals'])),1e-9)
                self.assertLess(fold['matching_max'],2e-7)
                self.assertLess(abs(fold['logM']),2e-7)
                independent=Orbit(three).point(np.array(fold['times']),fold['I'],1e-9,require_physical=False)
                self.assertLess(independent['closure'],1e-5)
        self.assertEqual(solver.failures,0)


if __name__=='__main__':unittest.main()
