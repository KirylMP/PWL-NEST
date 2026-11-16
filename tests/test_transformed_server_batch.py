import copy,json,unittest
from pathlib import Path
import numpy as np
from run_transformed_server_batch import check_fold,duplicate,signature

ROOT=Path(__file__).resolve().parents[1]

class TestFoldValidation(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.candidate=json.loads((ROOT/'results/fold_anchors/AM/square/seed_3.json').read_text())
 def test_selected_neutral_orbit_and_anchor_checks(self):
  result=check_fold(self.candidate)
  self.assertLess(result['anchor_max_residual'],1e-8)
  self.assertLess(result['ivp_closure'],1e-6)
 def test_changed_fold_input_rejected(self):
  j=copy.deepcopy(self.candidate);j['I_fold']+=.01
  with self.assertRaisesRegex(ValueError,'fixed fold input'):check_fold(j)
 def test_duplicate_detection(self):
  v=signature(self.candidate)
  self.assertTrue(duplicate(v,[v.copy()],1e-5))
  altered=v.copy();altered[0]+=1.
  self.assertFalse(duplicate(altered,[v],1e-5))
if __name__=='__main__':unittest.main()
