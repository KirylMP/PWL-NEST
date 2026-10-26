"""Shared coordinate map and equality-constrained geometry fitting support."""
import numpy as np
from scipy.linalg import solve_triangular
from geometry_reference import gauss,conditional_mean,CA,CB,MI
REF=.5

def phi(c,shape):
    if shape is None:return np.asarray(c)/REF-1
    p,e,n,k=shape;t=np.asarray(c)/REF
    w=np.exp(np.array([p,e,0])-max(p,e,0));w/=w.sum()
    power=np.expm1(n*np.log(t))/n
    exp=np.expm1(np.clip(k*(t-1),-700,700))/k if abs(k)>1e-7 else t-1
    return w[0]*power+w[1]*exp+w[2]*np.log(t)

def dphi(c,shape):
    if shape is None:return np.ones_like(np.asarray(c))/REF
    p,e,n,k=shape;t=np.asarray(c)/REF
    w=np.exp(np.array([p,e,0])-max(p,e,0));w/=w.sum()
    return (w[0]*t**(n-1)+w[1]*np.exp(np.clip(k*(t-1),-700,700))+w[2]/t)/REF

class Fit:
    def __init__(self,th,active,target,ai):
        self.th=np.array(th);self.active=active;self.target=target;self.ai=ai
        pairs=[gauss(36,a,b) for a,b in zip([CA,*th],[*th,CB])]
        self.c=np.concatenate([a for a,b in pairs]);self.w=np.concatenate([b for a,b in pairs])/(CB-CA);self.sw=np.sqrt(self.w)
        self.y=conditional_mean(self.c)
        A=np.zeros((len(active),len(th)+2));A[:,1]=1
        for row,j in enumerate(active):A[row,2:2+j]=1
        self.A=A
    def calculate(self,x,warped=True,full=False):
        a,b=np.exp(x[:2]);shape=x[2:] if warped else None;z=phi(self.c,shape);zt=phi(self.th,shape)
        B=np.column_stack([np.ones_like(z),z,*[np.maximum(z-t,0) for t in zt]])
        Q,R=np.linalg.qr(self.sw[:,None]*B,mode='reduced');coef0=solve_triangular(R,Q.T@(self.sw[:,None]*self.y))
        U=solve_triangular(R.T,self.A.T,lower=True).T
        tr,det=self.target.T;sn=-(tr+b)/a;sh=sn-det/(a*b);slopes=np.column_stack([sn,sh])
        correction=U.T@np.linalg.solve(U@U.T,slopes-self.A@coef0)
        co=coef0+solve_triangular(R,correction)
        residual=self.sw[:,None]*(B@co-self.y)
        if not full:return residual.ravel()
        coeff=np.vstack([co[0]-MI*self.ai,co[1],self.ai,co[2:]])
        field=np.zeros((2,len(self.th)+4));idx=[0,1,3,*range(4,len(self.th)+4)]
        field[0,idx]=-a*coeff[:,0];field[0,2]=a;field[1,idx]=b*coeff[:,1];field[1,2]=-b
        T=np.tril(np.ones((len(self.th)+1,len(self.th)+1)));ns=T@coeff[np.r_[1,np.arange(3,len(self.th)+3)],0];hs=T@coeff[np.r_[1,np.arange(3,len(self.th)+3)],1]
        return dict(a=float(a),b=float(b),shape=None if shape is None else shape.tolist(),C_thresholds=self.th.tolist(),z_thresholds=zt.tolist(),I_thresholds=[],N_coefficients=coeff[:,0].tolist(),H_coefficients=coeff[:,1].tolist(),field_coefficients=field.tolist(),trace=(-a*ns-b).tolist(),determinant=(a*b*(ns-hs)).tolist(),slope_constraint_error=float(abs(self.A@co-slopes).max()),profile_geometry_MSE=float(np.sum(residual**2)))

