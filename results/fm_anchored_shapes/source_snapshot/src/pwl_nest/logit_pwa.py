"""PWA in Y=(C,z=logit(h),I); physical field is generally NOT PWA.

No extra dimensions, thresholds or polynomial lifting. Internal integration and
regional exponentials use Y; inverse mapping is only for physical outputs.
"""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from scipy.special import expit,logit
from scipy.integrate import solve_ivp
from .calcium_strips import CalciumStrips
from .strict_pwa import lr_field,C_MAX


def to_latent(physical):
    p=np.asarray(physical,dtype=float)
    if p.shape[-1]!=3 or not np.isfinite(p).all() or np.any(p[...,1]<=0) or np.any(p[...,1]>=1):
        raise ValueError('Logit coordinates require finite states with 0<h<1; no clipping')
    y=p.copy();y[...,1]=logit(p[...,1]);return y


def to_physical(latent):
    y=np.asarray(latent,dtype=float)
    if y.shape[-1]!=3 or not np.isfinite(y).all():raise ValueError('Invalid latent state')
    x=y.copy();x[...,1]=expit(y[...,1]);return x


def transformed_lr_field(points,case):
    """Exact chain rule: (Cdot, zdot)=(f_C, f_h/[h(1-h)]).

    Complex arithmetic supported for Jacobian checks; real expit is stable.
    """
    y=np.asarray(points)
    h=1/(1+np.exp(-y[...,1])) if np.iscomplexobj(y) else expit(y[...,1])
    x=np.stack([y[...,0],h,y[...,2]],axis=-1)
    f=lr_field(x,case)
    return np.stack([f[...,0],f[...,1]/(h*(1-h))],axis=-1)


@dataclass
class LogitPWA:
    latent: CalciumStrips

    @property
    def case(self):return self.latent.case

    @property
    def metadata(self):return self.latent.metadata

    @property
    def region_count(self):return self.latent.region_count

    def save(self,path):
        Path(path).write_text(json.dumps(dict(case=self.case,thresholds_C=self.latent.thresholds.tolist(),
            basis=['1','C','z','I']+[f'max(0,C-{t:.17g})' for t in self.latent.thresholds],
            coefficients=self.latent.matrix.tolist(),metadata=self.metadata),indent=2,allow_nan=False)+'\n')

    @classmethod
    def load(cls,path):
        latent=CalciumStrips.load(path)
        if latent.metadata.get('transform')!='C,logit(h),I':raise ValueError('Not a logit model')
        return cls(latent)

    def physical_rhs(self,tau,baseline):
        """For diagnostics only; actual solver stores and evolves latent Y."""
        f=self.latent.rhs(tau,baseline)
        def rhs(time,x):
            y=to_latent(x);v=np.asarray(f(time,y));v[1]*=x[1]*(1-x[1]);return v
        return rhs


def calcium_matched_seed(physical,alpha,anchor_h=.8):
    """Control: identical C(t) for all protocols starting at anchor_h.

    Affine change h=alpha*z+beta on the old coefficient matrix, followed by
    interpreting z through expit for physical output. This control is NOT the
    nonlinear push-forward of the old model. It preserves its latent C dynamics
    and initial Cdot, and exposes what is gained solely by the bounded readout.
    """
    if alpha<=0:raise ValueError('Positive alpha required')
    beta=anchor_h-alpha*logit(anchor_h)
    matrix=physical.matrix.copy()
    matrix[:,0]+=matrix[:,2]*beta
    matrix[:,2]*=alpha
    matrix[1]/=alpha
    metadata={**physical.metadata,'coordinates':['C','z','I'],'transform':'C,logit(h),I',
              'class_name':'calcium_matched_logit_control','physical_field_PWA':False,
              'initialization':'C-dynamics-matched affine control; not exact nonlinear push-forward',
              'alpha':float(alpha),'beta':float(beta),'anchor_h':anchor_h,'accepted':False}
    return LogitPWA(CalciumStrips(physical.case,physical.thresholds.copy(),matrix,metadata))


def simulate_logit(model,initial,duration,tau=20.,baseline=.2,jumps=(),dt=.1,
                   rtol=2e-9,atol=2e-11,max_step=.15):
    """Solve in latent coordinates. Returned states are physical; metadata has Y.

    Samples at a jump time contain the left state; jump logs contain both sides.
    Out-of-range C is reported, never clipped. h output is a logistic readout.
    """
    state=to_latent(initial)
    if duration<=0 or dt<=0 or not np.isfinite(tau) or tau<=0 or not 0<=baseline<=1.3:
        raise ValueError('Invalid protocol')
    events=sorted((float(t),float(delta)) for t,delta in jumps)
    if any(not 0<t<duration or not np.isfinite(delta) for t,delta in events):raise ValueError('Invalid jumps')
    if len({t for t,_ in events})!=len(events):raise ValueError('Combine simultaneous jumps')
    if not 0<=state[2]<=1.3:raise ValueError('Initial IP3 outside domain')
    rhs=model.latent.rhs(tau,baseline)
    left=0.;time_parts=[];parts=[];log=[];error=0.;evaluations=0
    for right,delta in events+[(float(duration),0.)]:
        times=np.linspace(left,right,int(np.ceil((right-left)/dt))+1)
        def runaway(t,y):return 100.-max(abs(y[0]),abs(y[1]))
        runaway.terminal=True;runaway.direction=-1
        sol=solve_ivp(rhs,(left,right),state,t_eval=times,method='DOP853',
                      rtol=rtol,atol=atol,max_step=max_step,events=runaway)
        if not sol.success or sol.t[-1]!=right or not np.isfinite(sol.y).all():
            raise RuntimeError(f'Latent solver did not finish: {sol.message}')
        exact_i=baseline+(state[2]-baseline)*np.exp(-(sol.t-left)/tau)
        error=max(error,float(np.max(abs(sol.y[2]-exact_i))));evaluations+=sol.nfev
        time_parts.append(sol.t if not parts else sol.t[1:]);parts.append(sol.y.T if not parts else sol.y.T[1:])
        state=sol.y[:,-1].copy()
        if right!=duration:
            before=state.copy();state[2]+=delta
            if not 0<=state[2]<=1.3:raise ValueError('Jump outside IP3 domain')
            log.append(dict(time=right,latent_before=before.tolist(),latent_after=state.tolist(),
                            physical_before=to_physical(before).tolist(),physical_after=to_physical(state).tolist()))
        left=right
    y=np.vstack(parts);x=to_physical(y)
    return np.concatenate(time_parts),x,dict(latent=y,jumps=log,ip3_max_abs_error=error,
           rhs_evaluations=evaluations,physical=bool(x[:,0].min()>=-1e-7 and x[:,0].max()<=C_MAX+1e-7),
           logistic_saturation_samples=int(np.sum((x[:,1]==0)|(x[:,1]==1))))
