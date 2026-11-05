import json,sys,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from check_focus_feasibility import constraints

class LowerStableTest(unittest.TestCase):
    def test_lower_stable_node_is_allowed(self):
        n=np.zeros(5);h=np.array([0,0,1,0,0]);T=np.array([-2.,.5,-.5]);D=np.array([.2,1.,1.])
        self.assertGreater(T[0]**2-4*D[0],0)
        with patch('check_focus_feasibility.anchored',return_value=(n,h,T,D,np.zeros(4))):
            self.assertGreater(min(constraints({},[],.01)),0)
    def test_lower_saddle_is_rejected(self):
        n=np.zeros(5);h=np.array([0,0,1,0,0]);T=np.array([-2.,.5,-.5]);D=np.array([-.2,1.,1.])
        with patch('check_focus_feasibility.anchored',return_value=(n,h,T,D,np.zeros(4))):
            self.assertLess(min(constraints({},[],.01)),0)
    def test_middle_node_is_rejected(self):
        n=np.zeros(5);h=np.array([0,0,1,0,0]);T=np.array([-2.,.5,-.5]);D=np.array([.2,.01,1.])
        with patch('check_focus_feasibility.anchored',return_value=(n,h,T,D,np.zeros(4))):
            self.assertLess(min(constraints({},[],.01)),0)

if __name__=='__main__':unittest.main()
