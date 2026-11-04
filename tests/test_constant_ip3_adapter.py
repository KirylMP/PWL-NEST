"""Check that NEST encoding preserves each frozen physical candidate's field."""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from benchmark_constant_ip3 import parameters


class PhysicalAdapterTest(unittest.TestCase):
    def test_field_and_thresholds(self):
        for case in ('AM','AFM','FM'):
            data=json.loads((ROOT/'paper/data/physical_models/no_fold'/case/'parameters.json').read_text())
            p=parameters(data,.6,.12,.7);coeff=np.asarray(p['coefficients']).reshape(3,2,4)
            cuts=np.asarray(data['physical_thresholds_C'])
            for C in (.01,*cuts,.8):
                for q in (-.2,.5,1.2):
                    I=.6;u=C-p['c_reference'];v=(q-p['h_reference'])/.25
                    j=np.searchsorted(p['thresholds'],u,side='right')
                    actual=coeff[j]@np.array([1,u,v,I]);basis=np.array([1,C,I,*np.maximum(C-cuts,0)])
                    expected=[data['a']*(q-np.dot(data['alpha'],basis)),data['b']*(np.dot(data['beta'],basis)-q)]
                    np.testing.assert_allclose([actual[0],.25*actual[1]],expected,atol=1e-12)


if __name__=='__main__':unittest.main()
