"""Regression: an unreadable held-out file must not affect field initialization."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from pwl_nest.models import ROOT
sys.path.insert(0, str(ROOT / 'scripts'))
from refine_server_candidates import rebuild_training_field


class TrainingSeparationTest(unittest.TestCase):
    def test_field_rebuild_does_not_open_heldout_trajectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            spec = dict(fixed=[dict(name='fixed')], train=[dict(name='train')],
                        test=[dict(name='test')])
            (cache / 'data.json').write_text(json.dumps(spec))
            fixed = np.array([[.05, .8, .5], [.06, .7, .5]])
            train = np.array([[.1, .6, 1.], [.2, .5, .9]])
            for name, x in [('fixed', fixed), ('train', train)]:
                np.savez_compressed(cache / f'{name}.npz', t=[0., 1.], x=x)
            (cache / 'test.npz').write_text('Held-out data must never be opened here')
            np.savez_compressed(cache / 'field.npz', points=np.full((1, 3), np.nan))
            rebuild_training_field(cache)
            with np.load(cache / 'field.npz') as d:
                np.testing.assert_array_equal(d['points'], np.vstack([fixed, train]))
            actual = json.loads((cache / 'data.json').read_text())
            self.assertFalse(actual['heldout_used_for_field_initialization'])


if __name__ == '__main__':
    unittest.main()
