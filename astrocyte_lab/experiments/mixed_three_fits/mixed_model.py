"""Two-threshold PWA in mixed monotone calcium coordinate."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'astrocyte_lab/experiments/high_k_family'))
from fit_high_k import configure,FlexibleModel

def phi(c,shape):
 p,e,n,k=shape; t=np.asarray(c,dtype=float)/.5
 w=np.exp(np.array([p,e,0.])-max(p,e,0.));w/=w.sum()
 return w[0]*np.expm1(n*np.log(t))/n+w[1]*(np.expm1(np.clip(k*(t-1),-700,700))/k if abs(k)>1e-7 else t-1)+w[2]*np.log(t)

def inverse(z,shape):
 z=np.asarray(z,dtype=float);lo=np.full_like(z,np.log(1e-12));hi=np.full_like(z,np.log(2/1.185))
 if np.any(z<=phi(1e-12,shape)) or np.any(z>=phi(2/1.185,shape)):raise ValueError('Latent coordinate outside physical calcium range')
 for _ in range(55):
  mid=(lo+hi)/2;mask=phi(np.exp(mid),shape)<z;lo=np.where(mask,mid,lo);hi=np.where(mask,hi,mid)
 return np.exp((lo+hi)/2)

class MixedModel(FlexibleModel):
 def __init__(self,K):
  parent,fixed,_=configure(K)
  if not fixed:raise ValueError('Both LR crossings are required')
  self.__dict__.update(parent.__dict__);self.size=13;self.active=np.r_[np.arange(8),np.arange(9,13)];self.lo=np.r_[parent.lo,[-4.,-6.,.15,-4.]];self.hi_bounds=np.r_[parent.hi_bounds,[5.,3.,2.5,4.]];self.f=self
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
