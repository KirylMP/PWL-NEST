import json,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from anchored_lower_fold import FREE,anchor_model
from check_focus_feasibility import anchored,constraints

class FocusScreenTest(unittest.TestCase):
    def test_equilibrium_elimination_matches_existing_solver(self):
        for case in ('AM','AFM','FM'):
            base=json.loads((ROOT/'paper/data/physical_models/fixed_fold'/case/'parameters.json').read_text())
            for scale in (0.,.01,-.01):
                x=np.full(len(FREE),scale)
                coeff=[.05*max(abs(base[k][j]),.3) for k,j in FREE]
                change={key:float(v*s) for key,v,s in zip(FREE,x,coeff)}
                a,b,T,D,err=anchored(base,x)
                cut=.1*min(base['rest_target'][0],base['physical_thresholds_C'][0])
                model,_,_=anchor_model(base,change,cut,0.,0.)
                np.testing.assert_allclose(a,model['alpha'],atol=1e-12)
                np.testing.assert_allclose(b,model['beta'],atol=1e-12)
                np.testing.assert_allclose(T,model['trace'],atol=1e-12)
                np.testing.assert_allclose(D,model['determinant'],atol=1e-12)
                self.assertLess(max(abs(err)),1e-12)

    def test_saved_feasible_solutions(self):
        path=ROOT/'results/focus_screen/margin_001.json'
        if not path.exists():self.skipTest('optional screening output absent')
        data=json.loads(path.read_text())
        for case,item in data['results'].items():
            base=json.loads((ROOT/'paper/data/physical_models/fixed_fold'/case/'parameters.json').read_text())
            self.assertGreater(min(constraints(base,item['best']['x'],.01)),-1e-7)

if __name__=='__main__':unittest.main()
