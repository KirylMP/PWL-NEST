"""LR AM equilibrium branch and analytic Jacobian spectrum, I in [0,1]."""
import csv,json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq

OUT=Path(__file__).resolve().parent
def rhs(z,I):
    C,h=z;q=1.049*(I+.13)/(I+.9434)
    return np.array([(6*(I/(I+.13))**3*(C/(C+.08234))**3*h**3+.11)*(2-1.185*C)-.9*C*C/(C*C+.01),.2*(q*(1-h)-C*h)])
CMAX=2/1.185
def eqs(I):return [eq(I)]
def eq(I):
    q=1.049*(I+.13)/(I+.9434)
    c=brentq(lambda c:rhs([c,q/(q+c)],I)[0],0,2/1.185,xtol=5e-15)
    return np.array([c,q/(q+c)])
def jac(z,I):
    c,h=z;q=1.049*(I+.13)/(I+.9434);m=I/(I+.13);n=c/(c+.08234)
    return np.array([[6*m**3*h**3*(3*n*n*.08234/(c+.08234)**2*(2-1.185*c)-1.185*n**3)-.13035-.018*c/(c*c+.01)**2,18*m**3*n**3*h*h*(2-1.185*c)],[-.2*h,-.2*(q+c)]])
def invs(I):
    J=jac(eq(I),I);tr=np.trace(J);det=np.linalg.det(J)
    return tr,det,tr*tr-4*det
def roots(fun):
    grid=np.linspace(0,1,2001);v=[fun(x) for x in grid]
    return [brentq(fun,a,b,xtol=1e-14) for a,b,va,vb in zip(grid[:-1],grid[1:],v[:-1],v[1:]) if va*vb<0]
def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    hops=roots(lambda i:invs(i)[0]);nf=roots(lambda i:invs(i)[2]);events=[]
    for kind,rr in [('Hopf candidate',hops),('node-focus',nf)]:
        for i in rr:
            z=eq(i);tr,det,disc=invs(i)
            step=1e-5;dt=(invs(i+step)[0]-invs(i-step)[0])/(2*step)
            events.append(dict(kind=kind,I=i,C=z[0],h=z[1],trace=tr,determinant=det,discriminant=disc,trace_derivative_I=dt,eigenvalues=np.linalg.eigvals(jac(z,i)).astype(complex).tolist()))
    ii=np.unique(np.r_[np.linspace(0,1,2001),hops,nf]);zz=np.array([eq(i) for i in ii]);vv=np.array([invs(i) for i in ii]);tr,det,disc=vv.T
    s=np.sqrt(disc.astype(complex));lam=np.column_stack([(tr+s)/2,(tr-s)/2])
    rows=np.column_stack([ii,zz,lam[:,0].real,lam[:,0].imag,lam[:,1].real,lam[:,1].imag,tr,det,disc])
    with (OUT/'spectrum.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['I','C_eq','h_eq','Re_lambda_plus','Im_lambda_plus','Re_lambda_minus','Im_lambda_minus','trace','determinant','discriminant']);w.writerows(rows)
    err=0.;res=0.
    for i in np.linspace(0,1,101):
        z=eq(i);eps=1e-6
        Jfd=np.column_stack([(rhs(z+eps*np.eye(2)[j],i)-rhs(z-eps*np.eye(2)[j],i))/(2*eps) for j in range(2)])
        err=max(err,float(abs(Jfd-jac(z,i)).max()));res=max(res,float(abs(rhs(z,i)).max()))
    for e in events:e['eigenvalues']=[[x.real,x.imag] for x in e['eigenvalues']]
    summary=dict(parameters=dict(rC=6,rL=.11,C0=2,c1=.185,vER=.9,KER=.1,d1=.13,d2=1.049,d3=.9434,d5=.08234,a2=.2),events=events,min_determinant=float(det.min()),max_equilibrium_residual=res,max_Jacobian_finite_difference_error=err,endpoint_equilibria=zz[[0,-1]].tolist(),caveat='trace roots are spectral Hopf candidates; nonlinear criticality not recomputed here; node-focus transitions do not change stability')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    blue='#1764a0';red='#c83e42'
    for a,y,title,ylabel in [(ax[0,0],zz[:,0],'Equilibrium calcium',r'$C_*$ ($\mu$M)'),(ax[0,1],zz[:,1],'Equilibrium gate',r'$h_*$')]:
        stable=tr<0;a.plot(ii,np.where(stable,y,np.nan),c=blue,lw=2.3,label='Stable equilibrium');a.plot(ii,np.where(~stable,y,np.nan),c=red,lw=2.3,ls='--',label='Unstable equilibrium');a.set(title=title,ylabel=ylabel);a.legend(fontsize=9)
    a=ax[1,0];a.plot(ii,lam[:,0].real,lw=2.5,label=r'Re $\lambda_+$',c=blue);a.plot(ii,lam[:,1].real,lw=1.7,ls='--',label=r'Re $\lambda_-$',c='#e18a14');a.axhline(0,color='black',lw=.9);a.set(title='Real parts: stability changes',ylabel=r'Re $\lambda$ (s$^{-1}$)');a.legend(fontsize=9)
    a=ax[1,1];a.plot(ii,lam[:,0].imag,lw=2,label=r'Im $\lambda_+$',c=blue);a.plot(ii,lam[:,1].imag,lw=2,ls='--',label=r'Im $\lambda_-$',c='#e18a14');a.axhline(0,color='black',lw=.9);a.set(title='Imaginary parts: local rotation',ylabel=r'Im $\lambda$ (s$^{-1}$)');a.legend(fontsize=9)
    for a in ax.flat:
        for i in hops:a.axvline(i,color=red,lw=1,alpha=.65)
        for i in nf:a.axvline(i,color='gray',lw=1,ls=':',alpha=.7)
        a.set(xlabel=r'$I$ ($\mu$M)',xlim=(0,1));a.grid(alpha=.18)
    inset=ax[1,0].inset_axes([.48,.48,.48,.43])
    inset.plot(ii,lam[:,0].real,c=blue,lw=2);inset.plot(ii,lam[:,1].real,c='#e18a14',ls='--',lw=1.4)
    inset.axhline(0,c='black',lw=.7)
    for i in hops:inset.axvline(i,c=red,lw=.8)
    inset.set(xlim=(.31,.8),ylim=(-.16,.15),title='Zoom near zero');inset.tick_params(labelsize=8);inset.title.set_fontsize(9);inset.grid(alpha=.2)
    fig.suptitle('Li–Rinzel AM: equilibrium branch and local Jacobian spectrum\nRed: Re eigenvalue = 0; gray dotted: node–focus transition',fontsize=14)
    fig.savefig(OUT/'am_equilibrium_spectrum.png',dpi=180);fig.savefig(OUT/'am_equilibrium_spectrum.pdf');plt.close(fig)
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
