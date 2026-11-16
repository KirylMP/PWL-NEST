import copy,json,unittest
from pathlib import Path
import numpy as np
from run_transformed_server_batch import check_fold,duplicate,signature
from optimize_fixed_fold_cycles import Field
from sample_transformed_fold import PIVOT_NAMES,pivot_diagnostics

ROOT=Path(__file__).resolve().parents[1]

class TestFoldValidation(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.candidate=json.loads((ROOT/'results/fold_anchors/AM/log/seed_6.json').read_text())
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
 def test_each_field_coefficient_is_checked_as_a_fold_pivot(self):
  F=Field('AM','log',self.candidate);rows=pivot_diagnostics(F,F.p0,F.fold0)
  self.assertEqual([r['name'] for r in rows],list(PIVOT_NAMES))
  self.assertTrue(all(np.isfinite(r['relative_sigma_min']) for r in rows))
  self.assertTrue(any(r['locally_solvable'] for r in rows))
if __name__=='__main__':unittest.main()
