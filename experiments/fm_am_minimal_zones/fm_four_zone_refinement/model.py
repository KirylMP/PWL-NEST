"""Continuous FM PWA models for fixed-threshold refinement experiments.

The first and last baseline thresholds are the two fixed equilibrium events.
Extra thresholds retain the type of their parent region, including the saddle.
"""
from pathlib import Path
import sys
import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'fm_four_zone_fit'))
import fit_four as reference

BASE = np.array(reference.TC, dtype=float)
EVENT_C = BASE[[0, 2]]
EVENT_I = reference.FIXED_EVENT_I.copy()
# Fixed midpoint additions within each parent region of the four-zone model.
MIDPOINTS = [(0 + BASE[0]) / 2, (BASE[0] + BASE[1]) / 2,
             (BASE[1] + BASE[2]) / 2, (BASE[2] + reference.CMAX) / 2]
CUTS = {'base': [], 'region1': [MIDPOINTS[0]],
        'region2': [MIDPOINTS[1]], 'region3': [MIDPOINTS[2]],
        'all': MIDPOINTS}

class Model:
    def __init__(self, variant):
        if variant not in CUTS:
            raise ValueError(f'Unknown variant {variant!r}; choose from {list(CUTS)}')
        self.variant = variant
        self.cuts = np.sort(np.r_[BASE, CUTS[variant]])
        self.n = len(self.cuts) + 1
        self.fixed = [int(np.flatnonzero(self.cuts == t)[0]) for t in EVENT_C]
        midpoint = (np.r_[0., self.cuts] + np.r_[self.cuts, reference.CMAX]) / 2
        self.kind = np.searchsorted(BASE, midpoint).tolist()
        # x = log rates, regional T and D controls, N/H at (z=0,I=.8),
        # N/H input coefficients, four coordinate parameters.
        n = self.n
        self.n0, self.h0, self.ni, self.hi, self.shape = 2+2*n, 3+2*n, 4+2*n, 5+2*n, 6+2*n
        self.size = 10+2*n
        self.derived = [self.h0, self.hi]
        self.active = np.array([i for i in range(self.size) if i not in self.derived])
        self.lo = np.r_[np.log([.2,.03]),np.array([-5.,-6.,-5.,-5.])[self.kind],np.array([-8.,-9.,-9.,-9.])[self.kind],[-2,-2,-2,-2],[-4,-6,.15,-4]]
        self.hi_bounds = np.r_[np.log([40,1.2]),np.array([3.,6.,2.,2.])[self.kind],np.array([8.,1.,1.,1.])[self.kind],[3,3,2,2],[5,3,2.5,4]]

    def unpack(self, x):
        x=np.asarray(x,dtype=float); n=self.n
        if x.shape != (self.size,): raise ValueError('Incorrect parameter vector size')
        a,b=np.exp(x[:2]); u=x[2:2+n]; v=x[2+n:2+2*n]
        kind=np.asarray(self.kind)
        T=np.where(kind==1,u,np.where(kind==2,np.exp(u),-np.exp(u)))
        D=np.where(kind==0,T*T/4*expit(v),np.where(kind==1,-np.exp(v),T*T/4+np.exp(v)))
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
        d=self.basis(EVENT_C,.8,m)@(m['N']-m['H'])-(x[self.n0]-x[self.h0])
        slope=(d[0]-d[1])/(EVENT_I[1]-EVENT_I[0])
        if not np.isfinite(slope) or abs(slope)<1e-12: raise ValueError('Degenerate fixed-event constraints')
        x[self.hi]=x[self.ni]-slope
        x[self.h0]=x[self.n0]+d[0]+slope*(EVENT_I[0]-.8)
        return x

    def event_heights(self,x):
        m=self.unpack(x)
        return np.array([self.basis(c,i,m)@m['N'] for c,i in zip(EVENT_C,EVENT_I)])

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
            if 0<c<reference.CMAX:out.append(dict(z=[float(z[r]),float(y[r])],C=c,region=int(r),stable=self.kind[r] in (0,3)))
        return out

    def initial(self):
        return self.lift_original(reference.initialize())

    def lift_original(self, original_x):
        """Embed the original four-region parameter vector exactly."""
        old=np.asarray(original_x,dtype=float)
        if old.shape!=(18,):raise ValueError('Invalid four-region parameter vector')
        idx=np.asarray(self.kind,dtype=int)
        x=np.r_[old[:2],old[2:6][idx],old[6:10][idx],old[10:14],old[14:18]]
        return self.constrain_events(x)

    def lift_base(self, base_x):
        """Embed a four-region parameter vector exactly into this refinement."""
        base=Model('base'); old=np.asarray(base_x,dtype=float)
        if old.shape!=(base.size,):raise ValueError('Invalid base candidate')
        # The new subregions initially have exactly the parent matrices.
        u=old[2:2+base.n];v=old[2+base.n:2+2*base.n]
        idx=np.asarray(self.kind,dtype=int)
        x=np.r_[old[:2],u[idx],v[idx],old[base.n0:base.hi+1],old[base.shape:]]
        return self.constrain_events(x)
