import unittest
import numpy as np
from benchmark_constant_ip3 import load_case, parameters
from validate_selected_nest import rhs


class SelectedEncodingTest(unittest.TestCase):
    def test_selected_fields_in_each_zone_and_thresholds(self):
        for case in ('AM', 'AFM', 'FM'):
            d, _ = load_case(case)
            for I in (.15, .55, 1.2):
                for C in (0.05, *d['physical_thresholds_C'], .8):
                    q = .6
                    p = parameters(d, I, C, q)
                    j = np.searchsorted(p['thresholds'], p['u'], side='right')
                    r = np.asarray(p['coefficients']).reshape(3, 8)[j]
                    u, v = p['u'], p['v']
                    encoded = [r[0]+r[1]*u+r[2]*v+r[3]*I,
                               .25*(r[4]+r[5]*u+r[6]*v+r[7]*I)]
                    np.testing.assert_allclose(encoded, rhs(d, I, [C, q]), rtol=1e-12, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
