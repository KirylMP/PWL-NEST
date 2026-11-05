import sys
import unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from benchmark_accuracy_speed import error_metrics, lr_rhs, reference
from benchmark_constant_ip3 import load_case
from paper_figure1 import field

class AccuracySpeedTest(unittest.TestCase):
    def test_lr_field_matches_independent_existing_implementation(self):
        for case in ('AM','AFM','FM'):
            for c,h,I in ((.05,.8,.2),(.4,.6,.6),(1.,.3,1.)):
                np.testing.assert_allclose(lr_rhs(case,I,[c,h]),field(case,[c,h],I),rtol=1e-13,atol=1e-13)

    def test_reference_refinement_and_error_calculation(self):
        times=np.linspace(.05,2.,40)
        for case in ('AM','AFM','FM'):
            d,_=load_case(case); initial=d['rest_target']
            for model in ('LR','PWA'):
                a=reference(case,model,d,.6,initial,times,fine=False)
                b=reference(case,model,d,.6,initial,times,fine=True)
                self.assertLess(error_metrics(a,b)['C_max_abs'],1e-6)
                self.assertEqual(error_metrics(a,a)['C_rmse'],0.)

if __name__=='__main__':unittest.main()
