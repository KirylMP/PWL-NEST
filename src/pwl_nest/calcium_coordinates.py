"""PWA in (u=phi(C),v=psi(h),I).

Coordinates include powers, logarithms and pump saturation. Optional input
hinges and ordered oblique calcium planes preserve constant regional matrices.
"""
from dataclasses import dataclass
from pathlib import Path
import json,ctypes as ct
import numpy as np
from scipy.special import expit,logit
from scipy.integrate import solve_ivp
from .calcium_strips import CalciumStrips,decode
from .strict_pwa import lr_field,C_MAX
from .models import ROOT

CALCIUM_TAIL_JOINS=(.005,1.3)

@dataclass(frozen=True)
class CalciumCoordinate:
    kind: str
    reference: float
    scale: float = 1.
    half_saturation: float | None = None

    def __post_init__(self):
        if (self.kind not in ('identity','sqrt','sqrt_tails','cbrt','log','bounded_logit','square','cube','sixth','sixth_root','exp','inverse','inverse_sqrt','hill')
                or not np.isfinite(self.reference) or self.reference<=0
                or not np.isfinite(self.scale) or self.scale<=0):
            raise ValueError('Invalid calcium coordinate')
        if self.kind=='hill' and (self.half_saturation is None or not np.isfinite(self.half_saturation) or self.half_saturation<=0):
            raise ValueError('Hill coordinate requires a positive pump half-saturation')
        if self.kind=='bounded_logit' and self.reference>=C_MAX:raise ValueError('Require reference<C_MAX')
        if self.kind=='sqrt_tails' and not CALCIUM_TAIL_JOINS[0]<self.reference<CALCIUM_TAIL_JOINS[1]:raise ValueError('Reference outside tail joins')

    def saturation(self,c):return np.asarray(c)**2/(self.half_saturation**2+np.asarray(c)**2)

    def saturation_derivative(self,c):
        return 2*self.half_saturation**2*np.asarray(c)/(self.half_saturation**2+np.asarray(c)**2)**2

    @property
    def power(self):return {'identity':1.,'sqrt':.5,'cbrt':1/3,'log':0.,'square':2.,'cube':3.,'sixth':6.,'sixth_root':1/6,'inverse':-1.,'inverse_sqrt':-.5}[self.kind]

    @property
    def lower(self):
        if self.kind in ('bounded_logit','sqrt_tails'):return -np.inf
        if self.kind=='exp':return float(np.expm1(-self.reference)/self.scale)
        if self.kind=='hill':return float(-self.saturation(self.reference)/(self.saturation_derivative(self.reference)*self.scale))
        return -np.inf if self.power<=0 else -self.reference/(self.power*self.scale)

    @property
    def upper(self):
        if self.kind in ('bounded_logit','sqrt_tails'):return np.inf
        if self.kind=='hill':return float((1-self.saturation(self.reference))/(self.saturation_derivative(self.reference)*self.scale))
        return -self.reference/(self.power*self.scale) if self.kind!='exp' and self.power<0 else np.inf

    def forward(self,c):
        c=np.asarray(c)
        if not np.isfinite(c).all() or np.any(c.real<=0):raise ValueError('Require C>0; no clipping')
        if self.kind=='sqrt_tails':
            if np.any(c.real>=C_MAX):raise ValueError('Require C<C_MAX; no clipping')
            base=CalciumCoordinate('sqrt',self.reference,self.scale);lo,hi=CALCIUM_TAIL_JOINS
            low=c.real<lo;high=c.real>hi;middle=~(low|high)
            u=np.empty_like(c,dtype=np.result_type(c,float))
            u[low]=base.forward(lo)+lo*base.derivative(lo)*np.log(c[low]/lo)
            u[high]=base.forward(hi)-(C_MAX-hi)*base.derivative(hi)*np.log((C_MAX-c[high])/(C_MAX-hi))
            u[middle]=base.forward(c[middle])
            return u
        if self.kind=='bounded_logit':
            if np.any(c.real>=C_MAX):raise ValueError('Require C<C_MAX; no clipping')
            return self.reference*(1-self.reference/C_MAX)*(np.log(c/(C_MAX-c))-logit(self.reference/C_MAX))/self.scale
        if self.kind=='log':return self.reference*np.log(c/self.reference)/self.scale
        if self.kind=='hill':
            u=(self.saturation(c)-self.saturation(self.reference))/(self.saturation_derivative(self.reference)*self.scale)
            if not np.iscomplexobj(c) and (np.any(u<=self.lower) or np.any(u>=self.upper)):
                raise ValueError('Hill coordinate loses positive branch at floating-point precision')
            return u
        if self.kind=='exp':
            u=np.expm1(c-self.reference)/self.scale
            if not np.isfinite(u).all() or (not np.iscomplexobj(c) and np.any(u<=self.lower)):
                raise ValueError('Exponential coordinate loses positive branch or overflows')
            return u
        p=self.power
        difference=np.expm1(p*np.log(c/self.reference)) if self.kind in ('sixth','sixth_root') else (c/self.reference)**p-1
        u=self.reference/(p*self.scale)*difference
        if not np.iscomplexobj(c) and (np.any(u<=self.lower) or np.any(u>=self.upper)):
            raise ValueError('Coordinate loses positive branch at floating-point precision')
        return u

    def derivative(self,c):
        c=np.asarray(c)
        if np.any(c.real<=0):raise ValueError('Require C>0')
        if self.kind=='sqrt_tails':
            if np.any(c.real>=C_MAX):raise ValueError('Require C<C_MAX')
            base=CalciumCoordinate('sqrt',self.reference,self.scale);lo,hi=CALCIUM_TAIL_JOINS
            return np.where(c.real<lo,lo*base.derivative(lo)/c,
                np.where(c.real>hi,(C_MAX-hi)*base.derivative(hi)/(C_MAX-c),base.derivative(c)))
        if self.kind=='bounded_logit':
            if np.any(c.real>=C_MAX):raise ValueError('Require C<C_MAX')
            return self.reference*(C_MAX-self.reference)/(c*(C_MAX-c)*self.scale)
        if self.kind=='exp':return np.exp(c-self.reference)/self.scale
        if self.kind=='hill':return self.saturation_derivative(c)/(self.saturation_derivative(self.reference)*self.scale)
        return (self.reference/c if self.kind=='log' else (c/self.reference)**(self.power-1))/self.scale

    def inverse(self,u):
        u=np.asarray(u)
        if not np.isfinite(u).all():raise ValueError('Nonfinite calcium coordinate')
        with np.errstate(over='raise',invalid='raise',under='ignore'):
            if self.kind=='sqrt_tails':
                base=CalciumCoordinate('sqrt',self.reference,self.scale);lo,hi=CALCIUM_TAIL_JOINS
                left,right=base.forward(lo),base.forward(hi)
                low=u.real<left;high=u.real>right;middle=~(low|high)
                c=np.empty_like(u,dtype=np.result_type(u,float))
                c[low]=lo*np.exp((u[low]-left)/(lo*base.derivative(lo)))
                c[high]=C_MAX-(C_MAX-hi)*np.exp(-(u[high]-right)/((C_MAX-hi)*base.derivative(hi)))
                c[middle]=base.inverse(u[middle])
                if np.any(c.real>=C_MAX):raise ValueError('Bounded inverse saturated at C_MAX')
            elif self.kind=='bounded_logit':
                arg=logit(self.reference/C_MAX)+u*self.scale/(self.reference*(1-self.reference/C_MAX))
                c=C_MAX/(1+np.exp(-arg)) if np.iscomplexobj(arg) else C_MAX*expit(arg)
                if np.any(c.real>=C_MAX):raise ValueError('Bounded inverse saturated at C_MAX')
            elif self.kind=='log':c=self.reference*np.exp(u*self.scale/self.reference)
            elif self.kind=='hill':
                value=self.saturation(self.reference)+u*self.scale*self.saturation_derivative(self.reference)
                if np.any(u.real<=self.lower) or np.any(u.real>=self.upper) or np.any(value.real<=0) or np.any(value.real>=1):
                    raise ValueError('Hill inverse left 0<saturation<1')
                c=self.half_saturation*(value/(1-value))**.5
            elif self.kind=='exp':
                if np.any(u.real<=self.lower):raise ValueError('Exponential coordinate left C>0')
                c=self.reference+np.log1p(u*self.scale)
            else:
                if np.any(u.real<=self.lower) or np.any(u.real>=self.upper):raise ValueError('Root coordinate left the positive branch')
                base=1+self.power*u*self.scale/self.reference
                if np.any(base.real<=0):raise ValueError('Root coordinate left the positive branch')
                c=self.reference*base**(1/self.power)
        if not np.isfinite(c).all() or np.any(c.real<=0):raise ValueError('Calcium inverse overflow/underflow or nonpositive C')
        return c

    @property
    def guard_upper(self):
        """Runaway guard, distinct from the physical acceptance boundary."""
        return np.inf if self.kind in ('bounded_logit','sqrt_tails') else float(self.forward(20.))


@dataclass(frozen=True)
class GateCoordinate:
    """Centered h transforms with the same derivative as logit at h_ref."""
    kind: str
    reference: float

    def __post_init__(self):
        if self.kind not in ('identity','logit','log','exp','cube','sixth') or not 0<self.reference<1:
            raise ValueError('Invalid gate coordinate')

    @property
    def lower(self):
        if self.kind=='identity':return -1/(1-self.reference)
        if self.kind in ('cube','sixth'):return -1/(self.power*(1-self.reference))
        return float(np.expm1(-self.reference)/(self.reference*(1-self.reference))) if self.kind=='exp' else -np.inf

    @property
    def upper(self):
        if self.kind=='identity':return 1/self.reference
        if self.kind=='logit':return np.inf
        if self.kind=='log':return float(-np.log(self.reference)/(1-self.reference))
        if self.kind in ('cube','sixth'):return (1-self.reference**self.power)/(self.power*self.reference**self.power*(1-self.reference))
        return float(np.expm1(1-self.reference)/(self.reference*(1-self.reference)))

    def forward(self,h):
        h=np.asarray(h)
        if not np.isfinite(h).all() or np.any(h.real<=0) or np.any(h.real>=1):raise ValueError('Require 0<h<1')
        if self.kind=='identity':return (h-self.reference)/(self.reference*(1-self.reference))
        if self.kind=='logit':return np.log(h/(1-h))-logit(self.reference)
        if self.kind=='log':return np.log(h/self.reference)/(1-self.reference)
        if self.kind=='cube':return (h**3-self.reference**3)/(3*self.reference**3*(1-self.reference))
        if self.kind=='sixth':return np.expm1(6*np.log(h/self.reference))/(6*(1-self.reference))
        return np.expm1(h-self.reference)/(self.reference*(1-self.reference))

    def derivative(self,h):
        h=np.asarray(h)
        if np.any(h.real<=0) or np.any(h.real>=1):raise ValueError('Require 0<h<1')
        if self.kind=='identity':return np.ones_like(h)/(self.reference*(1-self.reference))
        if self.kind=='logit':return 1/(h*(1-h))
        if self.kind=='log':return 1/(h*(1-self.reference))
        if self.kind=='cube':return h**2/(self.reference**3*(1-self.reference))
        if self.kind=='sixth':return (h/self.reference)**5/(self.reference*(1-self.reference))
        return np.exp(h-self.reference)/(self.reference*(1-self.reference))

    def inverse(self,v):
        v=np.asarray(v)
        if not np.isfinite(v).all() or np.any(v.real<=self.lower) or np.any(v.real>=self.upper):
            raise ValueError('Gate coordinate left 0<h<1')
        if self.kind=='logit':
            return 1/(1+np.exp(-v-logit(self.reference))) if np.iscomplexobj(v) else expit(v+logit(self.reference))
        if self.kind=='identity':h=self.reference+self.reference*(1-self.reference)*v
        elif self.kind=='log':h=self.reference*np.exp((1-self.reference)*v)
        elif self.kind=='cube':h=(self.reference**3+3*self.reference**3*(1-self.reference)*v)**(1/3)
        elif self.kind=='sixth':h=self.reference*(1+6*(1-self.reference)*v)**(1/6)
        else:h=self.reference+np.log1p(self.reference*(1-self.reference)*v)
        if not np.isfinite(h).all() or np.any(h.real<=0) or np.any(h.real>=1):raise ValueError('Gate inverse lost 0<h<1')
        return h

    @property
    def power(self):return 6 if self.kind=='sixth' else 3


@dataclass
class CalciumTransformPWA:
    latent: CalciumStrips
    coordinate: CalciumCoordinate
    h_reference: float
    h_kind: str = 'logit'
    input_thresholds: np.ndarray | None = None
    input_weights: np.ndarray | None = None
    threshold_slopes: np.ndarray | None = None
    reference_input: float = .65

    def __post_init__(self):
        self.input_thresholds=np.array([] if self.input_thresholds is None else self.input_thresholds,dtype=float)
        self.input_weights=np.zeros((2,len(self.input_thresholds))) if self.input_weights is None else np.array(self.input_weights,dtype=float)
        self.threshold_slopes=np.zeros(len(self.latent.thresholds)) if self.threshold_slopes is None else np.array(self.threshold_slopes,dtype=float)
        if (self.input_weights.shape!=(2,len(self.input_thresholds)) or self.threshold_slopes.shape!=self.latent.thresholds.shape
                or not np.isfinite(np.r_[self.input_thresholds,self.input_weights.ravel(),self.threshold_slopes,self.reference_input]).all()
                or np.any(np.diff(self.input_thresholds)<=0) or np.any(self.input_thresholds<=0) or np.any(self.input_thresholds>=1.3)):
            raise ValueError('Invalid input thresholds or oblique geometry')
        for ip3 in [0.,1.3]:
            theta=self.thresholds_at_input(ip3)
            if np.any(np.diff(theta)<=0) or np.any(theta<=self.coordinate.lower):raise ValueError('Crossed thresholds or invalid calcium domain')

    @property
    def has_geometry(self):return bool(len(self.input_thresholds) or np.any(self.threshold_slopes!=0))

    def thresholds_at_input(self,ip3):
        return self.latent.thresholds+self.threshold_slopes*(ip3-self.reference_input)

    def field(self,y):
        if not self.has_geometry:return self.latent.field(y)
        y=np.asarray(y);theta=self.latent.thresholds
        hinges=np.maximum(0,y[...,0,None]-theta-self.threshold_slopes*(y[...,2,None]-self.reference_input))
        basis=np.concatenate([np.ones(y.shape[:-1]+(1,)),y,hinges],axis=-1)
        return basis@self.latent.matrix.T+np.maximum(0,y[...,2,None]-self.input_thresholds)@self.input_weights.T

    def geometry_coefficients(self):
        local=self.latent.local_coefficients().copy()
        tilt=self.latent.matrix[:,4:]*self.threshold_slopes
        accumulated=np.column_stack([np.zeros(2),np.cumsum(tilt,axis=1)]).T
        local[:,:,0]+=self.reference_input*accumulated
        local[:,:,3]-=accumulated
        result=np.repeat(local[None],len(self.input_thresholds)+1,axis=0)
        for j,knot in enumerate(self.input_thresholds):
            result[j+1:,:,:,0]-=self.input_weights[:,j]*knot
            result[j+1:,:,:,3]+=self.input_weights[:,j]
        return result

    def rhs(self,tau,baseline):
        if not self.has_geometry:return self.latent.rhs(tau,baseline)
        if not np.isfinite(tau) or tau<=0 or not 0<=baseline<=1.3:raise ValueError('Invalid IP3 dynamics')
        from bisect import bisect_right
        local=self.geometry_coefficients();knots=self.input_thresholds.tolist()
        theta=self.latent.thresholds;slopes=self.threshold_slopes;iref=self.reference_input
        def evaluate(t,y):
            j=np.searchsorted(theta+slopes*(y[2]-iref),y[0],side='right');s=bisect_right(knots,y[2]);p=local[s,j]
            return [p[0,0]+p[0,1]*y[0]+p[0,2]*y[1]+p[0,3]*y[2],
                    p[1,0]+p[1,1]*y[0]+p[1,2]*y[1]+p[1,3]*y[2],-(y[2]-baseline)/tau]
        return evaluate

    def region(self,y,tau,baseline):
        if not self.has_geometry:return self.latent.region(y,tau,baseline)
        j=np.searchsorted(self.thresholds_at_input(y[2]),y[0],side='right')
        s=np.searchsorted(self.input_thresholds,y[2],side='right');local=self.geometry_coefficients()[s,j]
        a,b=np.zeros((3,3)),np.zeros(3);a[:2],b[:2]=local[:,1:],local[:,0]
        a[2,2],b[2]=-1/tau,baseline/tau
        return a,b,int(s*self.latent.region_count+j)

    def at_input(self,ip3):
        """Exact constant-I planar model for existing autonomous statistics."""
        if not self.has_geometry:return self
        active=self.input_thresholds<ip3;matrix=self.latent.matrix.copy()
        matrix[:,0]-=self.input_weights[:,active]@self.input_thresholds[active]
        matrix[:,3]+=self.input_weights[:,active].sum(axis=1)
        latent=CalciumStrips(self.case,self.thresholds_at_input(ip3),matrix,self.metadata)
        return CalciumTransformPWA(latent,self.coordinate,self.h_reference,self.h_kind)

    def equilibrium_inputs_at_thresholds(self,anchors=None):
        mu,k,q,qb=self.latent.spectral_coefficients();theta=self.latent.thresholds
        if not self.has_geometry:return (k[:-1]*theta-q[:-1])/qb
        local=self.geometry_coefficients();B,D=self.latent.matrix[:,2]
        edges=np.r_[-np.inf,self.input_thresholds,np.inf];roots=[]
        anchors=np.zeros(len(theta)) if anchors is None else np.asarray(anchors)
        for j,th in enumerate(theta):
            candidates=[]
            for s in range(len(edges)-1):
                q0=B*local[s,j,1,0]-D*local[s,j,0,0]
                qi=B*local[s,j,1,3]-D*local[s,j,0,3]
                denominator=qi-k[j]*self.threshold_slopes[j]
                if abs(denominator)<1e-12:continue
                root=(k[j]*(th-self.threshold_slopes[j]*self.reference_input)-q0)/denominator
                if edges[s]-1e-12<=root<=edges[s+1]+1e-12:candidates.append(root)
            roots.append(min(candidates,key=lambda v:abs(v-anchors[j])) if candidates else 10.)
        return np.array(roots)

    def input_crossing_times(self,initial_ip3,tau,baseline,duration):
        if not np.isfinite(tau) or tau<=0:raise ValueError('Invalid tau')
        result=[]
        for knot in self.input_thresholds:
            if min(initial_ip3,baseline)<knot<max(initial_ip3,baseline):
                when=tau*np.log((initial_ip3-baseline)/(knot-baseline))
                if 0<when<duration:result.append((float(when),float(knot)))
        return sorted(result)

    @property
    def gate(self):return GateCoordinate(self.h_kind,self.h_reference)

    @property
    def variant(self):
        intrinsic=self.coordinate.kind if self.h_kind=='logit' else f'{self.coordinate.kind}_h_{self.h_kind}'
        return self.metadata.get('experiment_variant',intrinsic)

    @property
    def case(self):return self.latent.case

    @property
    def metadata(self):return self.latent.metadata

    @property
    def zref(self):return float(logit(self.h_reference))

    def to_latent(self,x):
        x=np.asarray(x,dtype=float)
        if x.ndim<1 or x.shape[-1]!=3 or not np.isfinite(x).all() or np.any(x[...,1]<=0) or np.any(x[...,1]>=1):
            raise ValueError('Require finite (C,h,I), C>0 and 0<h<1')
        return np.stack([self.coordinate.forward(x[...,0]),self.gate.forward(x[...,1]),x[...,2]],axis=-1)

    def to_physical(self,y):
        y=np.asarray(y)
        if y.ndim<1 or y.shape[-1]!=3 or not np.isfinite(y).all():raise ValueError('Invalid latent state')
        h=self.gate.inverse(y[...,1])
        return np.stack([self.coordinate.inverse(y[...,0]),h,y[...,2]],axis=-1)

    def source_field(self,y):
        x=self.to_physical(y);f=lr_field(x,self.case);h=x[...,1]
        return np.stack([self.coordinate.derivative(x[...,0])*f[...,0],self.gate.derivative(h)*f[...,1]],axis=-1)

    def save(self,path):
        record=dict(case=self.case,
            calcium_coordinate=dict(kind=self.coordinate.kind,reference=self.coordinate.reference,scale=self.coordinate.scale,
                                    normalization='phi(C_ref)=0, phi_prime(C_ref)=1/scale'),h_reference=self.h_reference,
            h_coordinate=dict(kind=self.h_kind,normalization='psi(h_ref)=0, psi_prime(h_ref)=1/[h_ref*(1-h_ref)]'),
            thresholds_u=self.latent.thresholds.tolist(),thresholds_C=self.metadata['physical_thresholds_C'],
            basis=['1','u','v','I']+[f'max(0,u-{t:.17g})' for t in self.latent.thresholds],
            coefficients=self.latent.matrix.tolist(),metadata=self.metadata)
        if self.coordinate.kind=='hill':record['calcium_coordinate']['half_saturation']=self.coordinate.half_saturation
        if self.coordinate.kind=='sqrt_tails':record['calcium_coordinate']['tail_joins_C']=list(CALCIUM_TAIL_JOINS)
        if self.has_geometry:
            record['basis']=['1','u','v','I']+[f'max(0,u-({t:.17g})-({s:.17g})*(I-{self.reference_input:.17g}))' for t,s in zip(self.latent.thresholds,self.threshold_slopes)]
            record['geometry']=dict(input_thresholds=self.input_thresholds.tolist(),input_weights=self.input_weights.tolist(),
                threshold_slopes=self.threshold_slopes.tolist(),reference_input=self.reference_input,
                definition='base affine + sum wj*(u-theta_j-sj*(I-Iref))_+ + sum zk*(I-eta_k)_+')
        Path(path).write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')

    @classmethod
    def load(cls,path):
        z=json.loads(Path(path).read_text());co=z['calcium_coordinate']
        if co['kind']=='sqrt_tails' and co.get('tail_joins_C',list(CALCIUM_TAIL_JOINS))!=list(CALCIUM_TAIL_JOINS):
            raise ValueError('Unsupported calcium tail joins')
        geometry=z.get('geometry',{})
        return cls(CalciumStrips(z['case'],np.array(z['thresholds_u']),np.array(z['coefficients']),z['metadata']),
                   CalciumCoordinate(co['kind'],co['reference'],co.get('scale',1.),co.get('half_saturation')),float(z['h_reference']),z.get('h_coordinate',{}).get('kind','logit'),
                   geometry.get('input_thresholds'),geometry.get('input_weights'),geometry.get('threshold_slopes'),geometry.get('reference_input',.65))


def decode_coordinates(case,thresholds,parameters,mu_sign,k_sign,metadata,coordinate,h_reference,h_kind='logit'):
    """Rest-force intercept expressed as an IP3 offset: q0=qb*offset.

    The reference source equilibrium has u=0 at I=0. This parameterization avoids
    imposing transform-dependent bounds on a large uncentered q0 intercept.
    """
    z=np.array(parameters);n=len(thresholds)+1;z[2*n]=0.
    latent=decode(case,thresholds,z,mu_sign,k_sign,metadata.copy())
    qb=latent.spectral_coefficients()[3]
    B,D=latent.matrix[0,2],latent.matrix[1,2]
    latent.matrix[1,0]=(qb*parameters[2*n]+D*latent.matrix[0,0])/B
    return CalciumTransformPWA(latent,coordinate,h_reference,h_kind)


class NativeCoordinateDynamic:
    """Domain-aware RK4 fitting helper; guards every accepted internal step."""
    def __init__(self):
        lib=ct.CDLL(str(ROOT/'.build/strips_rk4.so'));self.lib=lib
        ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        lib.strips_rk4_domain.argtypes=[ptr,ct.c_int,ptr,ptr,ct.c_double,ct.c_double,
            ct.c_double,ct.c_int,ct.c_double,ptr,ct.c_double,ct.c_double]
        lib.strips_rk4_domain.restype=ct.c_int
        lib.strips_rk4_box.argtypes=lib.strips_rk4_domain.argtypes+[ct.c_double,ct.c_double]
        lib.strips_rk4_box.restype=ct.c_int
        lib.strips_rk4_geometry.argtypes=[ptr,ptr,ct.c_int,ct.c_double,ptr,ct.c_int,ptr,ptr,
            ct.c_double,ct.c_double,ct.c_double,ct.c_int,ct.c_double,ptr,
            ct.c_double,ct.c_double,ct.c_double,ct.c_double]
        lib.strips_rk4_geometry.restype=ct.c_int

    def run(self,model,initial,duration,tau,baseline,outputs=361,maxdt=.025):
        mu,k,_,_=model.latent.spectral_coefficients()
        maxdt=min(maxdt,.7/max(1.,float(abs(mu).max()),float(np.sqrt(abs(k).max()))))
        out=np.empty((outputs,2));y0=model.to_latent(initial)
        if model.has_geometry:
            code=self.lib.strips_rk4_geometry(np.ascontiguousarray(model.latent.thresholds),np.ascontiguousarray(model.threshold_slopes),
                model.latent.region_count,model.reference_input,np.ascontiguousarray(model.input_thresholds),len(model.input_thresholds),
                np.ascontiguousarray(model.geometry_coefficients()),np.ascontiguousarray(y0),baseline,tau,duration,outputs,maxdt,out,
                model.coordinate.lower,model.coordinate.guard_upper,model.gate.lower,model.gate.upper)
            return None if code else out
        code=self.lib.strips_rk4_box(np.ascontiguousarray(model.latent.thresholds),model.latent.region_count,
            np.ascontiguousarray(model.latent.local_coefficients()),np.ascontiguousarray(y0),baseline,tau,
            duration,outputs,maxdt,out,model.coordinate.lower,model.coordinate.guard_upper,model.gate.lower,model.gate.upper)
        return None if code else out


def simulate_coordinates(model,initial,duration,tau=20.,baseline=.2,jumps=(),dt=.1,
                         rtol=2e-9,atol=2e-11,max_step=.15):
    state=model.to_latent(initial)
    if duration<=0 or dt<=0 or not np.isfinite(tau) or tau<=0 or not 0<=baseline<=1.3:
        raise ValueError('Invalid protocol')
    events=sorted((float(t),float(d)) for t,d in jumps)
    if any(not 0<t<duration or not np.isfinite(d) for t,d in events):raise ValueError('Invalid jumps')
    if len({t for t,_ in events})!=len(events):raise ValueError('Combine simultaneous jumps')
    if not 0<=state[2]<=1.3:raise ValueError('IP3 outside fitted domain')
    left=0.;parts=[];times=[];log=[];error=0.;evaluations=0
    rhs=model.rhs(tau,baseline)
    upper=model.coordinate.guard_upper
    def runaway(t,y):return min(upper-y[0],100.-abs(y[1]))
    runaway.terminal=True;runaway.direction=-1
    safety=[runaway];reasons=['runaway']
    def add_boundary(axis,bound,direction,reason):
        def event(t,y):return direction*(y[axis]-bound)
        event.terminal=True;event.direction=-1
        safety.append(event);reasons.append(reason)
    if np.isfinite(model.coordinate.lower):add_boundary(0,model.coordinate.lower,1,'positive-root branch exit')
    if np.isfinite(model.gate.lower):add_boundary(1,model.gate.lower,1,'gate domain exit h=0')
    if np.isfinite(model.gate.upper):add_boundary(1,model.gate.upper,-1,'gate domain exit h=1')
    for right,delta in events+[(float(duration),0.)]:
        sample=np.linspace(left,right,int(np.ceil((right-left)/dt))+1)
        sol=solve_ivp(rhs,(left,right),state,t_eval=sample,method='DOP853',
                      rtol=rtol,atol=atol,max_step=max_step,events=safety)
        if not sol.success or sol.t[-1]!=right or not np.isfinite(sol.y).all():
            reason=sol.message;when=float(sol.t[-1])
            if sol.status==1:
                event_index=next(j for j,event_times in enumerate(sol.t_events) if len(event_times))
                when=float(sol.t_events[event_index][0]);reason=reasons[event_index]
                if event_index==0:
                    exit_state=sol.y_events[0][0]
                    reason='physical C reached 20 uM' if abs(exit_state[0]-upper)<1e-6 else 'latent state runaway'
            raise RuntimeError(f'Latent solver incomplete near t={when:.9g}: {reason}')
        exact=baseline+(state[2]-baseline)*np.exp(-(sol.t-left)/tau)
        error=max(error,float(abs(sol.y[2]-exact).max()));evaluations+=sol.nfev
        times.append(sol.t if not parts else sol.t[1:]);parts.append(sol.y.T if not parts else sol.y.T[1:])
        state=sol.y[:,-1].copy()
        if right!=duration:
            before=state.copy();state[2]+=delta
            if not 0<=state[2]<=1.3:raise ValueError('IP3 jump outside domain')
            log.append(dict(time=right,latent_before=before.tolist(),latent_after=state.tolist(),
                            physical_before=model.to_physical(before).tolist(),physical_after=model.to_physical(state).tolist()))
        left=right
    y=np.vstack(parts);x=model.to_physical(y)
    return np.concatenate(times),x,dict(latent=y,jumps=log,ip3_max_abs_error=error,rhs_evaluations=evaluations,
         physical=bool(x[:,0].min()>0 and x[:,0].max()<=C_MAX+1e-7 and x[:,1].min()>0 and x[:,1].max()<1),
         logistic_saturation_samples=int(np.sum((x[:,1]==0)|(x[:,1]==1))))
