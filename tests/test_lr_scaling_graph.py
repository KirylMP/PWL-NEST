import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from benchmark_lr_scaling import graph


class GraphTest(unittest.TestCase):
 def test_one_astrocyte_per_distinct_edge_and_reproducibility(self):
  edges,small=graph(100,1000,101);_,large=graph(100,5000,101);_,again=graph(100,1000,101)
  self.assertEqual(len(edges),9900);self.assertEqual(len(set(map(tuple,small))),1000)
  self.assertTrue(np.all(small[:,0]!=small[:,1]));np.testing.assert_array_equal(small,again)
  np.testing.assert_array_equal(small,large[:len(small)])
  with self.assertRaises(ValueError):graph(100,10000,101)


if __name__=='__main__':unittest.main()
