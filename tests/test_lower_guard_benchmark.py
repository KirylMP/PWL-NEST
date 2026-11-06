import json
from pathlib import Path
import unittest
import numpy as np
from lower_calcium_guard import add_guard
from benchmark_lower_guard import field,encode

ROOT=Path(__file__).resolve().parents[1]
class LowerGuardTest(unittest.TestCase):
    def test_continuity_old_field_and_encoder(self):
        path=ROOT/'results/random_64/AM/seed_465/fit/candidate.json'
        if not path.exists():self.skipTest('AM 465 missing')
        base=json.loads(path.read_text());guard=add_guard(base,.025,100.)
        self.assertEqual(len(guard['physical_thresholds_C']),3)
        for C in (0.,.024999,.025,.025001,.08,.3,.6):
            for I in (0.,.5,1.2):
                q=.5;x=np.asarray(field(base,I,[C,q]));y=np.asarray(field(guard,I,[C,q]))
                np.testing.assert_allclose(y-x,[100*max(0,.025-C),0.],rtol=1e-11,atol=1e-11)
                for d in (base,guard):
                    p=encode(d,I,C,q,.2);j=np.searchsorted(p['thresholds'],p['u'],side='right')
                    row=np.asarray(p['coefficients']).reshape(len(p['thresholds'])+1,8)[j];u,v=p['u'],p['v']
                    encoded=[row[0]+row[1]*u+row[2]*v+row[3]*I,.25*(row[4]+row[5]*u+row[6]*v+row[7]*I)]
                    np.testing.assert_allclose(encoded,field(d,I,[C,q]),rtol=1e-11,atol=1e-11)
    def test_bad_cut_rejected(self):
        d={'physical_thresholds_C':[.1,.3]}
        with self.assertRaises(ValueError):add_guard(d,.1,100.)
if __name__=='__main__':unittest.main()
