"""PWA in (u=phi(C),v=logit(h)-logit(h_ref),I).

phi is a fixed power or logarithm, with optional constant scale normalization.
All variants have identical physical spectral thresholds and parameter counts.
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


@dataclass(frozen=True)
class CalciumCoordinate:
    kind: str
    reference: float
    scale: float = 1.

    def __post_init__(self):
        if (self.kind not in ('identity','sqrt','cbrt','log','square','cube')
                or not np.isfinite(self.reference) or self.reference<=0
                or not np.isfinite(self.scale) or self.scale<=0):
            raise ValueError('Invalid calcium coordinate')

    @property
    def power(self):return {'identity':1.,'sqrt':.5,'cbrt':1/3,'log':0.,'square':2.,'cube':3.}[self.kind]

    @property
    def lower(self):return -np.inf if self.kind=='log' else -self.reference/(self.power*self.scale)

    def forward(self,c):
        c=np.asarray(c)
        if not np.isfinite(c).all() or np.any(c.real<=0):raise ValueError('Require C>0; no clipping')
        if self.kind=='log':return self.reference*np.log(c/self.reference)/self.scale
        p=self.power
        u=self.reference/(p*self.scale)*((c/self.reference)**p-1)
        if not np.iscomplexobj(c) and np.any(u<=self.lower):
            raise ValueError('Coordinate loses positive branch at floating-point precision')
        return u

    def derivative(self,c):
        c=np.asarray(c)
        if np.any(c.real<=0):raise ValueError('Require C>0')
        return (self.reference/c if self.kind=='log' else (c/self.reference)**(self.power-1))/self.scale

    def inverse(self,u):
        u=np.asarray(u)
        if not np.isfinite(u).all():raise ValueError('Nonfinite calcium coordinate')
        with np.errstate(over='raise',invalid='raise',under='ignore'):
            if self.kind=='log':c=self.reference*np.exp(u*self.scale/self.reference)
            else:
                if np.any(u.real<=self.lower):raise ValueError('Root coordinate left the positive branch')
                base=1+self.power*u*self.scale/self.reference
                if np.any(base.real<=0):raise ValueError('Root coordinate left the positive branch')
                c=self.reference*base**(1/self.power)
        if not np.isfinite(c).all() or np.any(c.real<=0):raise ValueError('Calcium inverse overflow/underflow or nonpositive C')
        return c


@dataclass
class CalciumTransformPWA:
    latent: CalciumStrips
    coordinate: CalciumCoordinate
    h_reference: float

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
        return np.stack([self.coordinate.forward(x[...,0]),logit(x[...,1])-self.zref,x[...,2]],axis=-1)

    def to_physical(self,y):
        y=np.asarray(y)
        if y.ndim<1 or y.shape[-1]!=3 or not np.isfinite(y).all():raise ValueError('Invalid latent state')
        h=1/(1+np.exp(-y[...,1]-self.zref)) if np.iscomplexobj(y) else expit(y[...,1]+self.zref)
        return np.stack([self.coordinate.inverse(y[...,0]),h,y[...,2]],axis=-1)

    def source_field(self,y):
        x=self.to_physical(y);f=lr_field(x,self.case);h=x[...,1]
        return np.stack([self.coordinate.derivative(x[...,0])*f[...,0],f[...,1]/(h*(1-h))],axis=-1)

    def save(self,path):
        Path(path).write_text(json.dumps(dict(case=self.case,
            calcium_coordinate=dict(kind=self.coordinate.kind,reference=self.coordinate.reference,scale=self.coordinate.scale,
                                    normalization='phi(C_ref)=0, phi_prime(C_ref)=1/scale'),h_reference=self.h_reference,
            thresholds_u=self.latent.thresholds.tolist(),thresholds_C=self.metadata['physical_thresholds_C'],
            basis=['1','u','v','I']+[f'max(0,u-{t:.17g})' for t in self.latent.thresholds],
            coefficients=self.latent.matrix.tolist(),metadata=self.metadata),indent=2,allow_nan=False)+'\n')

    @classmethod
    def load(cls,path):
        z=json.loads(Path(path).read_text());co=z['calcium_coordinate']
        return cls(CalciumStrips(z['case'],np.array(z['thresholds_u']),np.array(z['coefficients']),z['metadata']),
                   CalciumCoordinate(co['kind'],co['reference'],co.get('scale',1.)),float(z['h_reference']))


def decode_coordinates(case,thresholds,parameters,mu_sign,k_sign,metadata,coordinate,h_reference):
    """Rest-force intercept expressed as an IP3 offset: q0=qb*offset.

    The reference source equilibrium has u=0 at I=0. This parameterization avoids
    imposing transform-dependent bounds on a large uncentered q0 intercept.
    """
    z=np.array(parameters);n=len(thresholds)+1;z[2*n]=0.
    latent=decode(case,thresholds,z,mu_sign,k_sign,metadata.copy())
    qb=latent.spectral_coefficients()[3]
    B,D=latent.matrix[0,2],latent.matrix[1,2]
    latent.matrix[1,0]=(qb*parameters[2*n]+D*latent.matrix[0,0])/B
    return CalciumTransformPWA(latent,coordinate,h_reference)


class NativeCoordinateDynamic:
    """Domain-aware RK4 fitting helper; guards every accepted internal step."""
    def __init__(self):
        lib=ct.CDLL(str(ROOT/'.build/strips_rk4.so'));self.lib=lib
        ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        lib.strips_rk4_domain.argtypes=[ptr,ct.c_int,ptr,ptr,ct.c_double,ct.c_double,
            ct.c_double,ct.c_int,ct.c_double,ptr,ct.c_double,ct.c_double]
        lib.strips_rk4_domain.restype=ct.c_int

    def run(self,model,initial,duration,tau,baseline,outputs=361,maxdt=.025):
        mu,k,_,_=model.latent.spectral_coefficients()
        maxdt=min(maxdt,.7/max(1.,float(abs(mu).max()),float(np.sqrt(abs(k).max()))))
        out=np.empty((outputs,2));y0=model.to_latent(initial)
        code=self.lib.strips_rk4_domain(np.ascontiguousarray(model.latent.thresholds),model.latent.region_count,
            np.ascontiguousarray(model.latent.local_coefficients()),np.ascontiguousarray(y0),baseline,tau,
            duration,outputs,maxdt,out,model.coordinate.lower,float(model.coordinate.forward(20.)))
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
    rhs=model.latent.rhs(tau,baseline)
    upper=float(model.coordinate.forward(20.))
    def runaway(t,y):return min(upper-y[0],100.-abs(y[1]))
    runaway.terminal=True;runaway.direction=-1
    def branch(t,y):return y[0]-model.coordinate.lower
    branch.terminal=True;branch.direction=-1
    safety=[runaway] if model.coordinate.kind=='log' else [runaway,branch]
    for right,delta in events+[(float(duration),0.)]:
        sample=np.linspace(left,right,int(np.ceil((right-left)/dt))+1)
        sol=solve_ivp(rhs,(left,right),state,t_eval=sample,method='DOP853',
                      rtol=rtol,atol=atol,max_step=max_step,events=safety)
        if not sol.success or sol.t[-1]!=right or not np.isfinite(sol.y).all():
            reason=sol.message;when=float(sol.t[-1])
            if sol.status==1:
                if model.coordinate.kind!='log' and len(sol.t_events[1]):
                    when=float(sol.t_events[1][0]);reason='positive-root branch exit'
                else:
                    when=float(sol.t_events[0][0]);exit_state=sol.y_events[0][0]
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
         physical=bool(x[:,0].min()>0 and x[:,0].max()<=C_MAX+1e-7),
         logistic_saturation_samples=int(np.sum((x[:,1]==0)|(x[:,1]==1))))
