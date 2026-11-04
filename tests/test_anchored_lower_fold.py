import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from anchored_lower_fold import anchor_model,anchored_fold
from continue_physical_fixed import Orbit
from fit_lower_fixed_fold import rhs


class AnchoredLowerTest(unittest.TestCase):
    def test_exact_anchors_and_field_reduction(self):
        for case,pivot in [('AM',('alpha',1)),('AFM',('alpha',1)),('FM',('alpha',4))]:
            with self.subTest(case=case):
                d=json.loads((ROOT/'paper/data/physical_models/fixed_fold'/case/'parameters.json').read_text())
                cut=.5*d['physical_thresholds_C'][0]
                three,model,checks,fold=anchored_fold(d,{('beta',1):.001},pivot,cut,.01,-.01)
                np.testing.assert_allclose(rhs(model,0.,d['rest_target']),[0,0],atol=1e-9)
                for c,I in zip(d['physical_thresholds_C'],d['fixed_equilibrium_threshold_I']):
                    n=np.array(model['alpha']);v=np.r_[1,c,I,np.maximum(c-np.array(model['physical_thresholds_C']),0)]
                    np.testing.assert_allclose(rhs(model,I,[c,n@v]),[0,0],atol=1e-9)
                for c in np.linspace(cut,1.,9):
                    np.testing.assert_allclose(rhs(three,.5,[c,.7]),rhs(model,.5,[c,.7]),atol=1e-12)
                self.assertLess(fold['matching_max'],2e-7);self.assertLess(abs(fold['logM']),2e-7)
                self.assertTrue(all(x>0 for x in checks['region_determinant']))
                independent=Orbit(three).point(np.array(fold['times']),fold['I'],1e-9,require_physical=False)
                self.assertLess(independent['closure'],1e-5)


if __name__=='__main__':unittest.main()
