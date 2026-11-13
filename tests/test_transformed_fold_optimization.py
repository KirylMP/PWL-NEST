import json
import unittest
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from optimize_fixed_fold_cycles import Field,flow
from physical_fold_routes import Route

ROOT=Path(__file__).resolve().parents[1]

class TestTransformedFlow(unittest.TestCase):
 def test_stiff_and_oscillatory_flow_matches_matrix_exponential(self):
  cases=[(np.array([[-32.,4.],[.001,-.01]]),np.array([.4,-.2]),np.array([-.4,.8]),45.),
         (np.array([[.2,4.],[-.5,.1]]),np.array([-.1,.3]),np.array([.4,.8]),7.)]
  for A,b,x,t in cases:
   B=np.zeros((3,3));B[:2,:2]=A;B[:2,2]=b
   np.testing.assert_allclose(flow(A,b,x,t),(expm(B*t)@np.r_[x,1])[:2],rtol=1e-10,atol=1e-10)
 def test_fixed_fold_matches_independent_matching_equations(self):
  for c,k,s in [('AM','log',6),('AM','square',3),('AFM','square',3),('FM','log',7)]:
   j=json.loads((ROOT/'results/fold_anchors'/c/k/f'seed_{s}.json').read_text());F=Field(c,k,j)
   other=Route({'physical_thresholds_C':j['cuts'],'alpha':j['n'],'beta':j['h'],'a':j['a'],'b':j['b'],'trace':j['trace']},'both')
   logs=np.log(j['times']); np.testing.assert_allclose(F.route(F.p0,j['I_fold'],logs,'both')[0],other.match(np.exp(logs),j['I_fold']),atol=1e-9,rtol=1e-8)
if __name__=='__main__':unittest.main()
