import json,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from fit_focus_fold_return import normalized_shape
from cycle_bootstrap import bootstrap
from return_cycle_metrics import solve_cycle

class FocusProtocolTest(unittest.TestCase):
    def test_phase_normalization_removes_amplitude_and_offset(self):
        x=np.cos(np.arange(64)*2*np.pi/64)
        np.testing.assert_allclose(normalized_shape(4*x+3),normalized_shape(x),atol=1e-14)
        with self.assertRaises(ValueError):normalized_shape(np.ones(64))

    def test_return_shape_peak_and_reference(self):
        path=ROOT/'results/focus_server_smoke_v2/shared_seed_AM'
        if not (path/'candidate.json').exists():self.skipTest('AM seed has not been materialized')
        model=json.loads((path/'candidate.json').read_text());rep=json.loads((path/'report.json').read_text());base=json.loads((ROOT/'paper/data/physical_models/fixed_fold/AM/parameters.json').read_text());row=base['training_cycles'][0]
        seeds=bootstrap(model,row['I']);out=solve_cycle(model,row['I'],rep['fold']['times'],extra_seeds=seeds)
        self.assertTrue(out['candidates']);cycle=out['candidates'][0];shape=cycle['shape']
        self.assertEqual(len(shape),64);self.assertAlmostEqual(shape[0],cycle['Cmax'],places=7)
        self.assertTrue(cycle['physical_sampled']);self.assertLess(cycle['logM'],0)

if __name__=='__main__':unittest.main()
