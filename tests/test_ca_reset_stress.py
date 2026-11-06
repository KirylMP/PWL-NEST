import json
from pathlib import Path
import unittest
from ca_reset_stress import stress

ROOT=Path(__file__).resolve().parents[1]
class ResetStressTest(unittest.TestCase):
    def test_reference_candidate_rejects_large_reset(self):
        path=ROOT/'results/random_64/AM/seed_465/fit/candidate.json'
        if not path.exists():self.skipTest('AM 465 source candidate not present')
        model=json.loads(path.read_text())
        r=stress(model,high_inputs=(1.2,),holds=(80.,))
        self.assertEqual(len(r['cases']),1)
        self.assertLess(r['min_C'],0.)
        self.assertAlmostEqual(r['cases'][0]['min_C_after'],-.23865,delta=5e-4)
        self.assertEqual(r['protocol']['I_after'],0.)

if __name__=='__main__':unittest.main()
