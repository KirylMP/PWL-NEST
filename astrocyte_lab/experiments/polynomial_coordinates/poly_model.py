"""Fixed polynomial approximations of a mixed calcium map; free PWA dynamics."""
import math
import sys
from pathlib import Path
import numpy as np
from numpy.polynomial import Polynomial, Chebyshev
from scipy.special import binom

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'astrocyte_lab/experiments/mixed_three_fits'))
from mixed_model import MixedModel, phi as mixed_phi

CENTER=.5
SCALE=.5
CMIN=0.001
CMAX=2/1.185
MAX_DEGREE=8


def build_map(seed, family, degree):
    """A true Taylor truncation at C=.5 or a Chebyshev least-squares fit."""
    if not 2<=degree<=MAX_DEGREE: raise ValueError('degree must be 2..8')
    t=seed['transform']; p,e,n,k=(float(t[x]) for x in ('p','e','n','k'))
    weights=np.exp(np.array([p,e,0.])-max(p,e,0.)); weights/=weights.sum()
    if family=='taylor':
        # u=(C-.5)/.5, so C/.5=1+u. The original map is zero at u=0.
        coeff=[0.]
        for j in range(1,degree+1):
            power=weights[0]*binom(n,j)/n
            exponential=weights[1]*(k**(j-1))/math.factorial(j)
            logarithm=weights[2]*((-1.)**(j+1))/j
            coeff.append(float(power+exponential+logarithm))
        poly=Polynomial(coeff)
    elif family=='chebyshev':
        nodes=np.cos(np.pi*(np.arange(2048)+.5)/2048)
        cs=(CMIN+CMAX)/2+(CMAX-CMIN)/2*nodes
        values=mixed_phi(cs,np.array([p,e,n,k]))
        cpoly=Chebyshev.fit(cs,values,degree,domain=[CMIN,CMAX]).convert(kind=Polynomial)
        poly=cpoly(Polynomial([CENTER,SCALE]))
    else: raise ValueError('unknown map family')
    lo=(CMIN-CENTER)/SCALE; hi=(CMAX-CENTER)/SCALE
    extrema=poly.deriv(2).roots()
    critical=[float(x.real) for x in extrema if abs(x.imag)<1e-9 and lo<x.real<hi]
    min_derivative=float(np.min(poly.deriv()([lo,hi,*critical]))/SCALE)
    if not np.isfinite(min_derivative) or min_derivative<=1e-8:
        raise ValueError(f'nonmonotone map on [{CMIN},{CMAX}]: min derivative={min_derivative:.8g}')
    coeff=np.zeros(MAX_DEGREE+1);coeff[:len(poly.coef)]=poly.coef
    grid=np.linspace(CMIN,CMAX,4000)
    errors=poly((grid-CENTER)/SCALE)-mixed_phi(grid,np.array([p,e,n,k]))
    return dict(family=family,degree=degree,center=CENTER,scale=SCALE,Cmin=CMIN,Cmax=CMAX,
                coefficients=coeff.tolist(),min_derivative=min_derivative,
                max_abs_z_error=float(np.max(np.abs(errors))),rms_z_error=float(np.sqrt(np.mean(errors**2))),
                source_transform=t)


def phi(c, spec):
    return np.polynomial.polynomial.polyval((np.asarray(c,dtype=float)-CENTER)/SCALE,spec['coefficients'])


def inverse(z,spec):
    z=np.asarray(z,dtype=float); zlo=float(phi(CMIN,spec)); zhi=float(phi(CMAX,spec))
    if np.any(z<zlo-1e-12) or np.any(z>zhi+1e-12): raise ValueError('Outside polynomial calcium domain')
    lo=np.full_like(z,CMIN); hi=np.full_like(z,CMAX)
    for _ in range(52):
        mid=(lo+hi)/2; mask=phi(mid,spec)<z
        lo=np.where(mask,mid,lo);hi=np.where(mask,hi,mid)
    return (lo+hi)/2


class PolyModel(MixedModel):
    def __init__(self,K,spec):
        super().__init__(K);self.spec=spec;self.active=np.arange(8)
        if not (CMIN<self.rest[0]<self.cuts[0]<self.cuts[1]<CMAX):raise ValueError('Rest or thresholds outside polynomial domain')
    def phi(self,c,shape=None):return phi(c,self.spec)
    def unpack(self,raw):
        x=self.constrain_events(raw);a,b=np.exp(x[:2]);th=phi(self.cuts,self.spec);rest=float(phi(self.rest[0],self.spec))
        T1=np.exp(x[3]);T2=-np.exp(x[4]);D1=T1*T1/4+np.exp(x[5]);D2=T2*T2/4+np.exp(x[6])
        D0=self.events[0]*D1*(th[1]-th[0])/((self.events[1]-self.events[0])*(th[0]-rest))
        T=np.array([-2*np.sqrt(D0)*np.exp(x[2]),T1,T2]);D=np.array([D0,D1,D2]);ns=-(T+b)/a;hs=ns-D/(a*b)
        q0=self.rest[1];s=(ns[0]-hs[0])*(th[0]-rest)/self.events[0]
        N=np.r_[q0-ns[0]*rest,ns[0],x[7],np.diff(ns)];H=np.r_[q0-hs[0]*rest,hs[0],x[7]+s,np.diff(hs)]
        co=np.zeros((2,11));idx=[0,1,3,4,5];co[0,idx]=-a*N;co[0,2]=a;co[1,idx]=b*H;co[1,2]=-b
        return dict(a=a,b=b,T=T,D=D,N=N,H=H,gamma=1.,shape=np.asarray(self.spec['coefficients']),th=th,tc=self.cuts.copy(),co=co,padded=np.r_[th,np.full(5,1e100)])
    def basis(self,c,I,m):
        z,I=np.broadcast_arrays(phi(c,self.spec),I)
        return np.stack([np.ones_like(z),z,I,*[np.maximum(z-t,0) for t in m['th']]],axis=-1)
