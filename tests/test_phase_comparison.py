import unittest
import numpy as np
from pwl_nest.phase_comparison import single_shift_comparison


class PhaseComparisonTest(unittest.TestCase):
    def test_known_delay_generalizes_to_later_window(self):
        t=np.linspace(0,100,10001)
        f=lambda x:np.sin(x*.5)+.15*np.sin(x*1.1)
        result=single_shift_comparison(t,f(t),f(t-1.7),(5,45),(50,95))
        self.assertAlmostEqual(result['shift_s'],1.7,places=4)
        self.assertLess(result['aligned_eval_rmse'],1e-5)

    def test_frequency_error_survives_one_phase_shift(self):
        t=np.linspace(0,100,10001);reference=np.sin(.5*t);candidate=np.sin(.53*t)
        result=single_shift_comparison(t,reference,candidate,(5,40),(60,95))
        self.assertGreater(result['aligned_eval_rmse'],.5)
        self.assertGreater(result['aligned_eval_rmse'],result['aligned_fit_rmse'])

    def test_translation_does_not_remove_amplitude_error(self):
        t=np.linspace(0,100,10001);reference=np.sin(.5*t)
        result=single_shift_comparison(t,reference,2*np.sin(.5*(t-1)),(5,95))
        self.assertGreater(result['aligned_eval_rmse'],.6)

    def test_variable_overlap_is_rejected(self):
        t=np.linspace(0,100,1001)
        with self.assertRaises(ValueError):single_shift_comparison(t,np.sin(t),np.sin(t),(0,100))


if __name__=='__main__':unittest.main()
