#!/usr/bin/env python3
"""Continue one Li-Rinzel limit-cycle fold in K_ER using periodic shooting."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares, minimize_scalar, brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent

def rhs(y,I,K):
    C,h=y[:2]; q=1.049*(I+.13)/(I+.9434)
    m=I/(I+.13); n=C/(C+.08234)
    dc=(6*m**3*n**3*h**3+.11)*(2-1.185*C)-.9*C*C/(C*C+K*K)
    dh=.2*(q*(1-h)-C*h)
    trace=6*m**3*h**3*(3*n*n*.08234/(C+.08234)**2*(2-1.185*C)-1.185*n**3)-.13035-1.8*C*K*K/(C*C+K*K)**2-.2*(q+C)
    return np.array([dc,dh,trace])

def integrate(v,K,rtol=2e-8, dense=False):
    C,h,I,T=v
    return solve_ivp(lambda t,y:rhs(y,I,K),(0,T),[C,h,0],method='DOP853',rtol=rtol,atol=rtol*1e-2,max_step=.35,dense_output=dense)

def residual(v,K):
    C,h,I,T=v
    try:
        sol=integrate(v,K)
        if not sol.success or not np.isfinite(sol.y[:,-1]).all():return np.full(4,100.)
        return np.array([(sol.y[0,-1]-C)*3,(sol.y[1,-1]-h)*3,sol.y[2,-1],rhs([C,h],I,K)[0]])
    except (ValueError,FloatingPointError,OverflowError):return np.full(4,100.)

def compute(K,guess,section):
    n=6
    def fun(v,rtol=2e-8):
        nodes=v[:-2].reshape(n,2); I,T=v[-2:]; out=[]; logm=0.
        try:
            for j in range(n):
                s=solve_ivp(lambda t,y:rhs(y,I,K),(0,T/n),[*nodes[j],0.],method='DOP853',rtol=rtol,atol=rtol*.01,max_step=.4)
                if not s.success or not np.isfinite(s.y[:,-1]).all():return np.full(2*n+2,1e3)
                out.extend(3*(s.y[:2,-1]-nodes[(j+1)%n]));logm+=s.y[2,-1]
            return np.r_[out,3*(nodes[0,0]-section),.05*logm]
        except (ValueError,OverflowError,FloatingPointError):return np.full(2*n+2,1e3)
    lo=np.r_[np.tile([.001,.01],n),.05,3.];hi=np.r_[np.tile([1.6,.999],n),2.5,120.]
    sol=least_squares(fun,guess,bounds=(lo,hi),x_scale='jac',xtol=1e-10,ftol=1e-10,gtol=1e-10,max_nfev=500)
    v=sol.x; verify=fun(v,rtol=2e-10); nodes=v[:-2].reshape(n,2); I,T=v[-2:]
    closure=float(np.max(np.abs(verify[:2*n]))/3); logm=float(verify[-1]/.05)
    if closure>2e-6 or abs(logm)>5e-5:raise ValueError(f'fold unresolved: closure={closure:.3g}, logM={logm:.3g}, nfev={sol.nfev}')
    cmin=10.;cmax=-1.;hmax=None
    for j in range(n):
        s=solve_ivp(lambda t,y:rhs(y,I,K),(0,T/n),[*nodes[j],0],method='DOP853',rtol=2e-10,atol=2e-12,max_step=.2,dense_output=True)
        ts=np.linspace(0,T/n,101); cs=s.sol(ts)[0]
        for sign in (1,-1):
            idx=int(np.argmin(sign*cs));l=ts[max(0,idx-1)];r=ts[min(100,idx+1)]
            if l==r:continue
            opt=minimize_scalar(lambda t:sign*float(s.sol(t)[0]),bounds=(l,r),method='bounded')
            value=float(s.sol(opt.x)[0])
            if sign==1:cmin=min(cmin,value)
            elif value>cmax:cmax=value;hmax=float(s.sol(opt.x)[1])
    amp=cmax-cmin
    if amp<.01 or cmin<=0:raise ValueError(f'cycle invalid or collapsed: amplitude={amp:.4g}, Cmin={cmin:.4g}')
    return dict(K=float(K),I=float(I),C_max=cmax,C_min=cmin,amplitude=amp,period=float(T),h_at_max=hmax,closure=closure,log_multiplier=logm,nfev=sol.nfev),v

def shift_section(v,K,section):
    n=6;nodes=v[:-2].reshape(n,2);I,T=v[-2:];dt=T/n;sols=[]
    for node in nodes:
        sols.append(solve_ivp(lambda t,y:rhs(y,I,K),(0,dt),[*node,0],method='DOP853',rtol=2e-10,atol=2e-12,max_step=.1,dense_output=True).sol)
    def state(t):
        t=t%T;j=min(int(t/dt),n-1);return sols[j](t-j*dt)[:2]
    grid=np.linspace(0,T,1001);vals=[state(t)[0]-section for t in grid]
    crossings=[(a,b) for a,b,fa,fb in zip(grid[:-1],grid[1:],vals[:-1],vals[1:]) if fa<0<=fb]
    if not crossings:raise ValueError('Cannot find upward crossing for new section')
    a,b=crossings[0];t0=brentq(lambda t:state(t)[0]-section,a,b)
    return np.r_[np.array([state(t0+j*dt) for j in range(n)]).ravel(),I,T]

def main():
    p=argparse.ArgumentParser();p.add_argument('--stop',type=float,default=.1);p.add_argument('--step',type=float,default=.001);p.add_argument('--out',type=Path,default=HERE/'results');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    guess=np.array([.46889887,.82210719,.78180033,.55446419,.33229722,.55745340,.16917768,.63324862,.10501201,.70912304,.04087355,.78510171,1.12315035,19.10456187])
    section=.46889887; rows=[];fail=None; history=[]
    if a.step<=0 or a.stop<=0:raise ValueError('step and stop must be positive')
    direction=1 if a.stop>=.05 else -1
    grid=np.r_[.05,.05+direction*min(.0001,a.step),np.arange(.05+direction*a.step,a.stop+direction*a.step*.25,direction*a.step)]
    grid=np.unique(grid);grid=grid if direction==1 else grid[::-1]
    for K in grid:
        try:
            if len(history)>=2:
                k0,v0=history[-2];k1,v1=history[-1];guess=v1+(v1-v0)*(K-k1)/(k1-k0)
            row,guess=compute(K,guess,section);rows.append(row)
            if row['C_max']-section<.08:
                section=(row['C_max']+row['C_min'])/2
                guess=shift_section(guess,K,section);history=[]
            history.append((K,guess.copy()))
            (a.out/'checkpoint.json').write_text(json.dumps(dict(rows=rows),indent=2)+'\n')
            print(f"K={K:.6f} I={row['I']:.8f} Cmin={row['C_min']:.6f} Cmax={row['C_max']:.6f} T={row['period']:.4f} logM={row['log_multiplier']:.2g}",flush=True)
        except Exception as e:
            fail=dict(K=float(K),reason=str(e));print('STOP',fail,flush=True);break
    (a.out/'fold_vs_k.json').write_text(json.dumps(dict(method='Six-segment multiple shooting, fixed section, closure and integrated divergence=0; continuation from FM fold; tighter DOP853 check per segment. Fold nondegeneracy and branch completeness not established.',rows=rows,stop=fail),indent=2)+'\n')
    if rows:
        fig,ax=plt.subplots(1,4,figsize=(16,4),layout='constrained'); ks=[r['K'] for r in rows]
        for key,label in [('C_min','C min'),('C_max','C max')]:ax[0].plot(ks,[r[key] for r in rows],'.-',label=label)
        ax[0].legend();ax[0].set_ylabel('C (µM)')
        for a_,key,label in zip(ax[1:],['amplitude','I','period'],['Amplitude (µM)','I fold (µM)','Period (s)']):a_.plot(ks,[r[key] for r in rows],'.-');a_.set_ylabel(label)
        for a_ in ax:a_.set_xlabel('K_ER (µM)');a_.grid(alpha=.3)
        fig.suptitle('Li–Rinzel: continued fold of cycles');fig.savefig(a.out/'fold_vs_k.png',dpi=170);plt.close(fig)
    print('SAVED',a.out,flush=True)

if __name__=='__main__':main()
