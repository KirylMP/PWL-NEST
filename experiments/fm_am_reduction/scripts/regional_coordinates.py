"""Invertible numerical coordinates; the physical model family is unchanged."""
import numpy as np
from fit_stage import phi


class RegionalCoordinates:
    """Graph values at C_ref,I_ref, I slopes, and independent regional slopes."""
    def __init__(self,thresholds,c_ref=.5,i_ref=.8):
        self.thresholds=np.asarray(thresholds);self.c_ref=c_ref;self.i_ref=i_ref
        self.ncoef=len(thresholds)+3;self.size=2*self.ncoef+6

    def from_raw(self,x):
        n=self.ncoef;zt=phi(self.thresholds,x[-4:],self.c_ref);out=np.array(x,copy=True)
        for offset in (0,n):
            co=x[offset:offset+n];slopes=co[1]+np.r_[0.,np.cumsum(co[3:])]
            value=co[0]+co[2]*self.i_ref+co[3:]@np.maximum(-zt,0)
            out[offset:offset+n]=np.r_[value,co[2],slopes]
        return out

    def to_raw(self,y):
        n=self.ncoef;zt=phi(self.thresholds,y[-4:],self.c_ref);out=np.array(y,copy=True)
        for offset in (0,n):
            block=y[offset:offset+n];changes=np.diff(block[2:])
            intercept=block[0]-block[1]*self.i_ref-changes@np.maximum(-zt,0)
            out[offset:offset+n]=np.r_[intercept,block[2],block[1],changes]
        return out

    def jacobian(self,y):
        # Only this cheap coordinate conversion is differenced. Trajectory
        # derivatives continue to use the variational equations.
        J=np.empty((len(y),len(y)))
        for j in range(len(y)):
            h=2e-6*max(1.,abs(y[j]));d=np.eye(len(y))[j]*h
            J[:,j]=(self.to_raw(y+d)-self.to_raw(y-d))/(2*h)
        return J

    def bounds(self):
        # Bounds belong to numerical coordinates, not to the scientific model.
        n=self.ncoef
        return (np.r_[np.full(2*n,-25.),-8.,-8.,-12.,-12.,-5.,-10.],
                np.r_[np.full(2*n,25.),5.,3.,12.,12.,4.,10.])
