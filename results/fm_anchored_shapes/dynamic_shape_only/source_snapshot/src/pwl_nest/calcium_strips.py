"""Compact continuous 3D PWA, with ONLY fixed spectral C thresholds.

F(C,h,I)=a*C+B*h+cI*I+c0+sum(u_j*(C-theta_j)_+) (two output rows).
Only the C-column and offsets differ across regions. Physical coordinates
are always (C,h,I); the optional Cdot coordinate is internal to statistics.
"""
from dataclasses import dataclass
import json
import ctypes as ct
from pathlib import Path
import numpy as np
from scipy.optimize import brentq,lsq_linear
from .strict_pwa import C_MAX,lr_field,equilibrium_h,spectral_events
from .models import ROOT


@dataclass
class CalciumStrips:
    case: str
    thresholds: np.ndarray
    matrix: np.ndarray  # two rows: [constant,C,h,I,hinge_1,...]
    metadata: dict

    @property
    def axes(self):
        return (np.array([0.,C_MAX]),np.array([0.,1.]),np.array([0.,1.3]))

    @property
    def region_count(self):
        return len(self.thresholds)+1

    def save(self,path):
        Path(path).write_text(json.dumps(dict(case=self.case,thresholds_C=self.thresholds.tolist(),
            basis=['1','C','h','I']+[f'max(0,C-{x:.17g})' for x in self.thresholds],
            coefficients=self.matrix.tolist(),metadata=self.metadata),indent=2,allow_nan=False)+'\n')

    @classmethod
    def load(cls,path):
        z=json.loads(Path(path).read_text())
        return cls(z['case'],np.array(z['thresholds_C']),np.array(z['coefficients']),z['metadata'])

    def field(self,points):
        p=np.asarray(points)
        phi=np.concatenate([np.ones(p.shape[:-1]+(1,)),p,
                            np.maximum(0,p[...,0,None]-self.thresholds)],axis=-1)
        return phi@self.matrix.T

    def rhs(self,tau,baseline,*,stage_extrapolation=False):
        if not np.isfinite(tau) or tau<=0 or not 0<=baseline<=1.3:
            raise ValueError('Invalid IP3 dynamics')
        local=self.local_coefficients()
        theta=self.thresholds.tolist()
        import bisect
        def rhs(t,x):
            j=bisect.bisect_right(theta,x[0])
            p=local[j]
            return [p[0,0]+p[0,1]*x[0]+p[0,2]*x[1]+p[0,3]*x[2],
                    p[1,0]+p[1,1]*x[0]+p[1,2]*x[1]+p[1,3]*x[2],
                    -(x[2]-baseline)/tau]
        return rhs

    def local_coefficients(self):
        local=np.repeat(self.matrix[None,:,:4],self.region_count,axis=0)
        for j,theta in enumerate(self.thresholds):
            local[j+1:,:,1]+=self.matrix[:,j+4]
            local[j+1:,:,0]-=theta*self.matrix[:,j+4]
        return local

    def region(self,point,tau,baseline):
        j=np.searchsorted(self.thresholds,point[0],side='right')
        local=self.local_coefficients()[j]
        a,b=np.zeros((3,3)),np.zeros(3)
        a[:2],b[:2]=local[:,1:],local[:,0]
        a[2,2],b[2]=-1/tau,baseline/tau
        return a,b,j

    def spectral_coefficients(self):
        local=self.local_coefficients()
        a=local[:,:,1:3]
        mu=np.trace(a,axis1=1,axis2=2)
        k=np.linalg.det(a)
        q=self.matrix[0,2]*local[:,1,0]-self.matrix[1,2]*local[:,0,0]
        qb=self.matrix[0,2]*self.matrix[1,3]-self.matrix[1,2]*self.matrix[0,3]
        return mu,k,q,float(qb)

    def equilibria(self,ip3):
        found=[]
        for j,p in enumerate(self.local_coefficients()):
            a=p[:,1:3]
            if abs(np.linalg.det(a))<1e-12:
                continue
            state=np.linalg.solve(a,-p[:,0]-p[:,3]*ip3)
            if (j==0 or state[0]>=self.thresholds[j-1]-1e-10) and (j==self.region_count-1 or state[0]<self.thresholds[j]-1e-10):
                eig=np.linalg.eigvals(a)
                tr,det=np.trace(a),np.linalg.det(a)
                label='saddle' if det<0 else ('stable_' if tr<0 else 'unstable_')+('focus' if tr*tr<4*det else 'node')
                found.append(dict(C=float(state[0]),h=float(state[1]),region=j,type=label,
                                  eigenvalues=[[float(e.real),float(e.imag)] for e in eig]))
        return found


def decode(case,thresholds,z,mu_sign,k_sign,metadata=None):
    """All 2n+6 free parameters; positive B, negative h self-coupling D.

    mu,k are traces/determinants, q0,qb restoring-force intercept/input slope.
    Physical embedding B,D,cI,c0 completes the representation without changing
    state definitions. No coefficients depend on the running IP3.
    """
    n=len(thresholds)+1
    mu=mu_sign*np.exp(z[:n]);k=k_sign*np.exp(z[n:2*n])
    if metadata and metadata.get('enforce_types'):
        from scipy.special import expit
        for j,label in enumerate(metadata['target_types']):
            if label.endswith('focus'):
                k[j]=mu[j]*mu[j]/4+np.exp(z[n+j])
            elif label.endswith('node'):
                k[j]=mu[j]*mu[j]/4*expit(z[n+j])
    q0,qb=z[2*n]*.05,np.exp(z[2*n+1])
    B,D,cI,c0=np.exp(z[2*n+2]),-np.exp(z[2*n+3]),z[2*n+4],z[2*n+5]
    ac=mu-D
    dc=(D*ac-k)/B
    matrix=np.zeros((2,n+3))
    matrix[0,:4]=[c0,ac[0],B,cI]
    matrix[1,:4]=[(q0+D*c0)/B,dc[0],D,(qb+D*cI)/B]
    matrix[:,4:]=np.array([np.diff(ac),np.diff(dc)])
    return CalciumStrips(case,np.array(thresholds),matrix,{} if metadata is None else metadata)


def spectral_seed(case,training_points):
    events=sorted(spectral_events(case),key=lambda e:e['C'])
    theta=np.array([e['C'] for e in events])
    def force(c,ip3):return lr_field([c,equilibrium_h(c,ip3),ip3],case)[0]
    def input_at(c):return brentq(lambda ip3:force(c,ip3),0,1.3,xtol=1e-13)
    lower=brentq(lambda c:force(c,0),1e-9,C_MAX)
    upper=brentq(lambda c:force(c,1.3),1e-9,C_MAX)
    cs=np.r_[lower,theta,upper]
    inputs=np.array([0.]+[e['I'] for e in events]+[1.3])
    slopes=np.diff(inputs)/np.diff(cs)
    mus,dets,weights,types=[],[],[],[]
    # Jacobians evaluated directly by complex-step differentiation of LR field.
    for left,right in zip(cs[:-1],cs[1:]):
        c=np.linspace(left+1e-10,right-1e-10,301)
        ip3=np.array([input_at(v) for v in c]);h=equilibrium_h(c,ip3)
        points=np.column_stack([c,h,ip3]).astype(complex)
        jac=np.stack([lr_field(points+1e-25j*np.eye(3)[d],case).imag/1e-25 for d in [0,1]],axis=-1)
        tr=np.trace(jac,axis1=1,axis2=2);det=np.linalg.det(jac)
        w=abs(np.diff(ip3));weights.append(float(w.sum()))
        mus.append(float(np.dot((tr[:-1]+tr[1:])/2,w)/w.sum()))
        dets.append(float(np.dot((det[:-1]+det[1:])/2,w)/w.sum()))
        j=len(c)//2
        types.append('saddle' if det[j]<0 else ('stable_' if tr[j]<0 else 'unstable_')+('focus' if tr[j]*tr[j]<4*det[j] else 'node'))
    mus,dets,weights=map(np.array,[mus,dets,weights])
    qb=max(1e-5,float(np.dot(weights*slopes,dets)/np.dot(weights,slopes*slopes)))
    k=qb*slopes
    g=mus[0]*training_points[:,0]+np.maximum(0,training_points[:,0,None]-theta)@np.diff(mus)
    # Fit physical embedding to source Cdot without changing spectral geometry.
    design=np.column_stack([training_points[:,0],training_points[:,1],training_points[:,2],np.ones(len(training_points))])
    fit=lsq_linear(design,lr_field(training_points,case)[:,0]-g,
                   bounds=([.001,.01,-5,-5],[5,20,5,5]),tol=1e-12)
    minusD,B,cI,c0=fit.x
    z=np.r_[np.log(abs(mus)),np.log(abs(k)),k[0]*lower/.05,np.log(qb),np.log(B),np.log(minusD),cI,c0]
    meta=dict(class_name='continuous_3D_C_strips',coordinates=['C','h','I'],
              thresholds_policy='source equilibrium spectral type changes; fixed across all IP3 and trajectories',
              events=events,source_average_trace=mus.tolist(),source_average_determinant=dets.tolist(),
              spectral_weight='absolute dI along the full equilibrium branch',
              source_inverse_equilibrium_slopes=slopes.tolist(),restoring_scale=qb,
              regions=len(theta)+1,free_parameters=len(z),target_types=types,
              IP3='exact -(I-baseline)/tau; no fitted row',
              source='Li-Rinzel legacy parameter set; not yet compared against NEST implementation',
              scope='candidate, not accepted; fixed threshold heuristic under test')
    return z,np.sign(mus),np.sign(k),meta


class NativeStatistics:
    """Exact planar regional statistics for CONSTANT IP3, rebuilt legacy engine."""
    def __init__(self):
        self.lib=ct.CDLL(str(ROOT/'.build/am_engine.so'))
        ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        self.lib.simulate.argtypes=[ct.c_double,ptr,ptr,ptr,ptr,ptr,ct.c_int,
                                    ct.c_double,ct.c_double,ct.c_double,ptr]
        self.lib.simulate.restype=ct.c_int
        self.lib.simulate_trace.argtypes=self.lib.simulate.argtypes+[ptr,ct.c_int]
        self.lib.simulate_trace.restype=ct.c_int
        self.trace=np.zeros((120000,3))

    def run(self,model,ip3,initial,duration=1600.,v0_override=None,diagnostics=False):
        mu,k,q,qb=model.spectral_coefficients()
        if min(abs(k))<1e-8:return None
        output=np.zeros(6)
        v0=model.field([initial[0],initial[1],ip3])[0] if v0_override is None else v0_override
        arrays=[np.ascontiguousarray(x) for x in [model.thresholds,mu,k,q+qb*ip3,np.zeros(len(k))]]
        args=(ip3,*arrays,len(k),float(initial[0]),float(v0),duration,output)
        if diagnostics:
            count=self.lib.simulate_trace(*args,self.trace,len(self.trace))
            if count<0 or count>=len(self.trace):return None
            tr=self.trace[:count]
            c=np.r_[initial[0],tr[:,1]]
            v=np.r_[v0,tr[:,2]]
            local=model.local_coefficients()[np.searchsorted(model.thresholds,c,side='right')]
            h=(v-local[:,0,0]-local[:,0,1]*c-local[:,0,3]*ip3)/model.matrix[0,2]
            return output,dict(min_h=float(h.min()),max_h=float(h.max()),max_C=float(c.max()))
        code=self.lib.simulate(*args)
        if code or not np.isfinite(output[[0,1,3,4,5]]).all():return None
        return output


class NativeDynamic:
    """Fast approximate physical-state trajectories used only during fitting."""
    def __init__(self):
        self.lib=ct.CDLL(str(ROOT/'.build/strips_rk4.so'))
        ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        self.lib.strips_rk4.argtypes=[ptr,ct.c_int,ptr,ptr,ct.c_double,ct.c_double,
                                     ct.c_double,ct.c_int,ct.c_double,ptr]
        self.lib.strips_rk4.restype=ct.c_int

    def run(self,model,initial,duration,tau,baseline,outputs=361,maxdt=.025):
        result=np.empty((outputs,2))
        mu,k,_,_=model.spectral_coefficients()
        maxdt=min(maxdt,.7/max(1.,float(np.max(abs(mu))),float(np.sqrt(np.max(abs(k))))))
        code=self.lib.strips_rk4(np.ascontiguousarray(model.thresholds),model.region_count,
            np.ascontiguousarray(model.local_coefficients()),np.ascontiguousarray(initial,dtype=float),
            baseline,tau,duration,outputs,maxdt,result)
        return None if code else result
