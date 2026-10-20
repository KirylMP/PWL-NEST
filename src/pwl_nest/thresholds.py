"""Ordered physical thresholds parameterized by relative interval widths."""
from dataclasses import dataclass
import numpy as np
from scipy.special import softmax


@dataclass(frozen=True)
class OrderedThresholds:
    reference: np.ndarray
    lower: float
    upper: float

    def __post_init__(self):
        ref=np.array(self.reference,dtype=float,copy=True)
        if (ref.ndim!=1 or not len(ref) or not np.isfinite(np.r_[ref,self.lower,self.upper]).all()
                or np.any(np.diff(np.r_[self.lower,ref,self.upper])<=0)):
            raise ValueError('Require ordered thresholds strictly between finite bounds')
        ref.flags.writeable=False;object.__setattr__(self,'reference',ref)

    @property
    def gaps(self):return np.diff(np.r_[self.lower,self.reference,self.upper])

    def decode(self,shifts):
        shifts=np.asarray(shifts,dtype=float)
        if shifts.shape!=self.reference.shape or not np.isfinite(shifts).all():raise ValueError('Invalid threshold shifts')
        if np.all(shifts==0):return self.reference.copy()
        # Last interval is the reference; m independent shifts for m thresholds.
        gaps=(self.upper-self.lower)*softmax(np.log(self.gaps)+np.r_[shifts,0.])
        thresholds=self.lower+np.cumsum(gaps)[:-1]
        if np.any(np.diff(np.r_[self.lower,thresholds,self.upper])<=0):
            raise ValueError('Threshold intervals collapsed at floating-point precision')
        return thresholds

    def encode(self,thresholds):
        gaps=np.diff(np.r_[self.lower,np.asarray(thresholds,dtype=float),self.upper])
        if gaps.shape!=self.gaps.shape or not np.isfinite(gaps).all() or np.any(gaps<=0):
            raise ValueError('Invalid physical thresholds')
        ratio=np.log(gaps/self.gaps)
        return ratio[:-1]-ratio[-1]
