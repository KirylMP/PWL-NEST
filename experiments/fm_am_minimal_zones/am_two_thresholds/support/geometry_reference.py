"""Pure FM nullcline MSE, 7 moving C hinges, no I hinges or other penalties.
Variable projection eliminates all 20 linear geometry coefficients.
"""
import json
from pathlib import Path
from functools import lru_cache
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize,root
OUT=Path(__file__).resolve().parent
CA,CB,IA,IB=.03,1.15,.4,1.2
AREA=(CB-CA)*(IB-IA);MI=(IA+IB)/2;VI=(IB-IA)**2/12
INITIAL=np.array([.04861935111371004,.05214116272147648,.05484469312764684,.06128930910964862,.1357042423799119,.25181841575270236,.3926251934101712])
@lru_cache(None)
def nodes(n):return leggauss(n)
def gauss(n,a,b):
    x,w=nodes(n);return a+(x+1)*(b-a)/2,w*(b-a)/2
def F(c):
    pump=.9*c*c/(c*c+.0025)-.11*(2-1.185*c)
    return np.cbrt(pump/(6*(c/(c+.08234))**3*(2-1.185*c)))
def truth(c,i):
    c,i=np.broadcast_arrays(c,i);q=1.049*(i+.13)/(i+.9434)
    return np.stack([F(c)*(1+.13/i),q/(q+c)],axis=-1)
EA=1+.13*np.log(IB/IA)/(IB-IA)
def conditional_mean(c):
    A=1.049+c;B=1.049*.13+c*.9434;R=1.049*.13-1.049*B/A
    h=1.049/A+R/A*np.log((A*IB+B)/(A*IA+B))/(IB-IA)
    return np.column_stack([EA*F(c),h])
class Problem:
    def __init__(self):
        c,wc=gauss(768,CA,CB);wc/=CB-CA;i,wi=gauss(192,IA,IB);wi/=IB-IA
        target=truth(c[:,None],i[None,:]);mean=conditional_mean(c)
        self.ai=np.einsum('c,i,cij->j',wc,wi*(i-MI),target)/VI
        self.ef2=np.einsum('c,i,cij->j',wc,wi,target*target)
        self.floor=self.ef2-np.sum(wc[:,None]*mean**2,axis=0)-VI*self.ai**2
    def calc(self,t,n=48,full=False):
        t=np.asarray(t)
        if not CA<t[0] or not t[-1]<CB or np.min(np.diff(t))<=0:return 1e5+float(t@t),2*t
        edges=[CA,*t,CB];pairs=[gauss(n,a,b) for a,b in zip(edges[:-1],edges[1:])];c=np.concatenate([a for a,b in pairs]);w=np.concatenate([b for a,b in pairs])/(CB-CA)
        B=np.column_stack([np.ones_like(c),c,*[np.maximum(c-u,0) for u in t]]);y=conditional_mean(c)
        co=np.linalg.lstsq(B*np.sqrt(w[:,None]),y*np.sqrt(w[:,None]),rcond=None)[0];res=B@co-y
        value=float(np.sum(w[:,None]*res**2));grad=np.array([-2*np.sum(w[c>u,None]*res[c>u]*co[k+2]) for k,u in enumerate(t)])
        if not full:return value,grad
        coeff=np.vstack([co[0]-MI*self.ai,co[1],self.ai,co[2:]])
        mse=self.floor+np.sum(w[:,None]*res**2,axis=0)
        return dict(C_thresholds=t.tolist(),coefficient_order=['1','C','I',*[f'C_hinge{k+1}' for k in range(7)]],N_coefficients=coeff[:,0].tolist(),H_coefficients=coeff[:,1].tolist(),MSE_components=mse.tolist(),joint_MSE=float(sum(mse)),RMSE_components=np.sqrt(mse).tolist(),joint_RMSE=float(np.sqrt(sum(mse))),integral_squared_error=float(AREA*sum(mse)),profile_MSE=value,threshold_gradient=grad.tolist(),irreducible_MSE_components=self.floor.tolist(),linear_gram_min_eigenvalue=float(np.linalg.eigvalsh(B.T@(w[:,None]*B)).min()))
    def direct(self,fit,nc=64,ni=64):
        t=np.array(fit['C_thresholds']);co=np.column_stack([fit['N_coefficients'],fit['H_coefficients']]);pairs=[gauss(nc,a,b) for a,b in zip([CA,*t],[*t,CB])];c=np.concatenate([a for a,b in pairs]);wc=np.concatenate([b for a,b in pairs])/(CB-CA);i,wi=gauss(ni,IA,IB);wi/=IB-IA
        C,I=np.meshgrid(c,i,indexing='ij');c=C.ravel();i=I.ravel();w=(wc[:,None]*wi[None,:]).ravel();B=np.column_stack([np.ones_like(c),c,i,*[np.maximum(c-u,0) for u in t]]);res=B@co-truth(c,i)
        gc=2*B.T@(w[:,None]*res);gt=np.array([-2*np.sum(w[c>u,None]*res[c>u]*co[k+3]) for k,u in enumerate(t)])
        return dict(nC_per_region=nc,nI=ni,MSE_components=np.sum(w[:,None]*res**2,axis=0).tolist(),joint_MSE=float(np.sum(w[:,None]*res**2)),coefficient_gradient_max=float(abs(gc).max()),threshold_gradient=gt.tolist(),threshold_gradient_max=float(abs(gt).max()))
def main():
    p=Problem();rng=np.random.default_rng(20261025);starts=[INITIAL,np.linspace(CA,CB,9)[1:-1],np.geomspace(CA+1e-3,CB-.02,7),np.array([.034,.043,.06,.1,.2,.45,.85])]
    starts += [np.sort(CA+(CB-CA)*rng.beta(.55,1.5,7)) for _ in range(32)]
    runs=[]
    for k,s in enumerate(starts):
        def objective(t):
            v,g=p.calc(t);return 1e5*v,1e5*g
        opt=minimize(objective,s,jac=True,method='SLSQP',bounds=[(CA+1e-6,CB-1e-6)]*7,constraints=[dict(type='ineq',fun=lambda t:np.diff(t)-1e-5)],options=dict(ftol=1e-11,maxiter=600))
        runs.append(dict(start=s.tolist(),thresholds=opt.x.tolist(),profile_MSE=float(opt.fun/1e5),success=bool(opt.success),message=str(opt.message)))
        if k%6==0:print('start',k,'profile MSE',runs[-1]['profile_MSE'],flush=True)
    best=min(runs,key=lambda r:r['profile_MSE']);rt=root(lambda t:p.calc(t,128)[1],best['thresholds'],tol=1e-10)
    t=rt.x if rt.success and CA<rt.x[0]<rt.x[-1]<CB and np.min(np.diff(rt.x))>0 else np.array(best['thresholds'])
    fit=p.calc(t,160,True);fixed=p.calc(INITIAL,160,True);eps=2e-6
    Hess=np.column_stack([(p.calc(t+eps*np.eye(7)[j],128)[1]-p.calc(t-eps*np.eye(7)[j],128)[1])/(2*eps) for j in range(7)])
    output=dict(model='two continuous nullcline graphs, shared 7 C thresholds, globally affine I',domain=dict(C=[CA,CB],I=[IA,IB]),objective='MSE_N + MSE_H, each averaged uniformly over dC dI; no other penalties or constraints except ordered interior thresholds',free_geometry_parameters=27,threshold_count=7,I_threshold_count=0,fixed_threshold_fit=fixed,moving_threshold_fit=fit,root_success=bool(rt.success),profile_Hessian_eigenvalues=np.linalg.eigvalsh(Hess).tolist(),direct_checks=[p.direct(fit,48,64),p.direct(fit,96,128)],runs=runs,notes=['No equilibrium anchors, spectral penalties, positivity guards, cycle data or period targets','a,b speeds are not identified by nullcline geometry; no dynamic model selected','Global optimality over C thresholds is not proved'])
    (OUT/'results.json').write_text(json.dumps(output,indent=2));print(json.dumps({k:v for k,v in output.items() if k not in ['runs']},indent=2))
if __name__=='__main__':main()
