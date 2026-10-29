"""Monotone rational calcium maps with analytic inverse and fixed LR events."""
import os,sys
from pathlib import Path
import numpy as np
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'astrocyte_lab/experiments/mixed_three_fits'))
from mixed_model import MixedModel as Parent
def phi(c,shape):return np.asarray(c,dtype=float)
def inverse(z,shape):
 z=np.asarray(z,dtype=float)
 if np.any(z<=1e-12) or np.any(z>=2/1.185):raise ValueError('Outside physical calcium range')
 return z

class PhysicalModel(Parent):
 def __init__(self,K):
  super().__init__(K);self.active=np.arange(8)
  self.lo[9:11]=[-4.,-5.];self.hi_bounds[9:11]=[3.,5.]
 def phi(self,c,shape=None):return phi(c,shape)
 def unpack(self,raw):
  x=self.constrain_events(raw);a,b=np.exp(x[:2]);shape=x[9:13];th=phi(self.cuts,shape);rest=float(phi(self.rest[0],shape))
  T1=np.exp(x[3]);T2=-np.exp(x[4]);D1=T1*T1/4+np.exp(x[5]);D2=T2*T2/4+np.exp(x[6]);D0=self.events[0]*D1*(th[1]-th[0])/((self.events[1]-self.events[0])*(th[0]-rest))
  T=np.array([-2*np.sqrt(D0)*np.exp(x[2]),T1,T2]);D=np.array([D0,D1,D2]);ns=-(T+b)/a;hs=ns-D/(a*b);q0=self.rest[1];s=(ns[0]-hs[0])*(th[0]-rest)/self.events[0]
  N=np.r_[q0-ns[0]*rest,ns[0],x[7],np.diff(ns)];H=np.r_[q0-hs[0]*rest,hs[0],x[7]+s,np.diff(hs)]
  co=np.zeros((2,11));idx=[0,1,3,4,5];co[0,idx]=-a*N;co[0,2]=a;co[1,idx]=b*H;co[1,2]=-b
  return dict(a=a,b=b,T=T,D=D,N=N,H=H,gamma=1.,shape=shape,th=th,tc=self.cuts.copy(),co=co,padded=np.r_[th,np.full(5,1e100)])
 def basis(self,c,I,m):
  z,I=np.broadcast_arrays(phi(c,m['shape']),I);return np.stack([np.ones_like(z),z,I,*[np.maximum(z-t,0) for t in m['th']]],axis=-1)
