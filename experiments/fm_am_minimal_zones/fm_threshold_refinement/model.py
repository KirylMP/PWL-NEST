"""Continuous FM PWA models for fixed-threshold refinement experiments.

The first and last baseline thresholds are the two fixed equilibrium events.
Extra thresholds retain the type of their parent region, so no saddle is added.
"""
from pathlib import Path
import sys
import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'fm_four_zone_fit'))
import fit_four as reference

BASE = np.array([reference.TC[0], reference.TC[2]])
EVENT_I = reference.FIXED_EVENT_I.copy()
# The outer domain ends at CMAX. Midpoints are a reproducible starting design,
# not a claim that these extra cuts are optimal.
CUTS = {'base': [], 'lower': [BASE[0] / 2],
        'middle': [(BASE[0] + BASE[1]) / 2],
        'all': [BASE[0] / 2, (BASE[0] + BASE[1]) / 2,
                (BASE[1] + reference.CMAX) / 2]}

class Model:
    def __init__(self, variant):
        if variant not in CUTS:
            raise ValueError(f'Unknown variant {variant!r}; choose from {list(CUTS)}')
        self.variant = variant
        self.cuts = np.sort(np.r_[BASE, CUTS[variant]])
        self.n = len(self.cuts) + 1
        self.fixed = [int(np.flatnonzero(self.cuts == t)[0]) for t in BASE]
        midpoint = (np.r_[0., self.cuts] + np.r_[self.cuts, reference.CMAX]) / 2
        self.kind = np.searchsorted(BASE, midpoint).tolist()
        # x = log rates, regional T and D controls, N/H at (z=0,I=.8),
        # N/H input coefficients, four coordinate parameters.
        n = self.n
        self.n0, self.h0, self.ni, self.hi, self.shape = 2+2*n, 3+2*n, 4+2*n, 5+2*n, 6+2*n
        self.size = 10+2*n
        self.derived = [self.h0, self.hi]
        self.active = np.array([i for i in range(self.size) if i not in self.derived])
        self.lo = np.r_[np.log([.2,.03]), np.full(n,-5.), np.full(n,-8.), [-2,-2,-2,-2], [-4,-6,.15,-4]]
        self.hi_bounds = np.r_[np.log([40,1.2]), np.full(n,2.), np.full(n,8.), [3,3,2,2], [5,3,2.5,4]]

    def unpack(self, x):
        x=np.asarray(x,dtype=float); n=self.n
        if x.shape != (self.size,): raise ValueError('Incorrect parameter vector size')
        a,b=np.exp(x[:2]); u=x[2:2+n]; v=x[2+n:2+2*n]
        T=np.where(np.asarray(self.kind)==1,np.exp(u),-np.exp(u))
        D=np.where(np.asarray(self.kind)==0,T*T/4*expit(v),T*T/4+np.exp(v))
        ns=-(T+b)/a; hs=ns-D/(a*b)
        shape=x[self.shape:]; th=reference.phi(self.cuts,shape)
        N=np.r_[0.,ns[0],x[self.ni],np.diff(ns)]
        H=np.r_[0.,hs[0],x[self.hi],np.diff(hs)]
        at=np.r_[1.,0.,.8,np.maximum(-th,0.)]
        N[0]=x[self.n0]-at@N; H[0]=x[self.h0]-at@H
        co=np.zeros((2,11)); idx=[0,1,3,*range(4,4+len(th))]
        co[0,idx]=-a*N;co[0,2]=a;co[1,idx]=b*H;co[1,2]=-b
        padded=np.r_[th, np.full(7-len(th),1e100)]
        return dict(a=a,b=b,T=T,D=D,N=N,H=H,shape=shape,th=th,co=co,padded=padded)

    def basis(self,c,I,m):
        z=reference.phi(np.asarray(c),m['shape']);z,I=np.broadcast_arrays(z,I)
        return np.stack([np.ones_like(z),z,I,*[np.maximum(z-t,0) for t in m['th']]],axis=-1)

    def constrain_events(self,x):
        x=np.asarray(x,dtype=float).copy();m=self.unpack(x)
        d=self.basis(BASE,.8,m)@(m['N']-m['H'])-(x[self.n0]-x[self.h0])
        slope=(d[0]-d[1])/(EVENT_I[1]-EVENT_I[0])
        if not np.isfinite(slope) or abs(slope)<1e-12: raise ValueError('Degenerate fixed-event constraints')
        x[self.hi]=x[self.ni]-slope
        x[self.h0]=x[self.n0]+d[0]+slope*(EVENT_I[0]-.8)
        return x

    def event_heights(self,x):
        m=self.unpack(x)
        return np.array([self.basis(c,i,m)@m['N'] for c,i in zip(BASE,EVENT_I)])

    def physically_valid_events(self,x):
        h=self.event_heights(x)
        return bool(np.all(np.isfinite(h)) and np.all((0<h)&(h<1)))

    def equilibrium(self,I,m):
        n=m['N'][1]+np.r_[0,np.cumsum(m['N'][3:])]
        h=m['H'][1]+np.r_[0,np.cumsum(m['H'][3:])]
        nc=m['N'][0]-np.r_[0,np.cumsum(m['N'][3:]*m['th'])]
        hc=m['H'][0]-np.r_[0,np.cumsum(m['H'][3:]*m['th'])]
        z=(hc-nc+(m['H'][2]-m['N'][2])*I)/(n-h)
        y=nc+n*z+m['N'][2]*I
        valid=(z>np.r_[-np.inf,m['th']])&(z<=np.r_[m['th'],np.inf])&(y>0)&(y<1)
        out=[]
        for r in np.flatnonzero(valid):
            c=float(reference.inverse(z[r],m))
            if 0<c<reference.CMAX:out.append(dict(z=[float(z[r]),float(y[r])],C=c,region=int(r),stable=self.kind[r]!=1))
        return out

    def initial(self):
        old=reference.initialize(); om=reference.unpack(old)
        # Copy regional spectra from the closest matching original region.
        source=[0 if k==0 else 2 if k==1 else 3 for k in self.kind]
        T=om['T'][source];D=om['D'][source]
        u=np.log(abs(T)); p=np.empty(self.n)
        for j,k in enumerate(self.kind):
            if k==0:
                frac=np.clip(4*D[j]/T[j]**2,1e-8,1-1e-8);p[j]=np.log(frac/(1-frac))
            else:p[j]=np.log(max(D[j]-T[j]**2/4,1e-8))
        x=np.r_[old[:2],u,p,old[10:14],old[14:]]
        return self.constrain_events(np.clip(x,self.lo+1e-6,self.hi_bounds-1e-6))

    def lift_base(self, base_x):
        """Embed a two-threshold parameter vector exactly into this refinement."""
        base=Model('base'); old=np.asarray(base_x,dtype=float)
        if old.shape!=(base.size,):raise ValueError('Invalid base candidate')
        # The new subregions initially have exactly the parent matrices.
        u=old[2:2+base.n];v=old[2+base.n:2+2*base.n]
        idx=np.asarray(self.kind,dtype=int)
        x=np.r_[old[:2],u[idx],v[idx],old[base.n0:base.hi+1],old[base.shape:]]
        return self.constrain_events(x)
