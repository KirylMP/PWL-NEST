"""Check a Bessel-form Riccati flow under exponential IP3 in one fixed PWA cell."""
import importlib.util
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.special import ive, kve

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("dynamic_validation", ROOT/"dynamic_validation.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def riccati_bessel(s0, t, ip3_0, baseline, tau):
    """Exact scalar active-hinge solution for s'=a0+a1 exp(-t/tau)+b*s+k*s^2."""
    k = M.K
    b0 = np.array([M.CO[0, 0], M.CO[1, 0], M.P[4]*(-1.12)]) + baseline*np.array(
        [M.CO[0, 3], M.CO[1, 3], M.P[4]])
    b1 = (ip3_0-baseline)*np.array([M.CO[0, 3], M.CO[1, 3], M.P[4]])
    a0 = float(M.L@b0 + k*M.THETA**2)
    a1 = float(M.L@b1)
    B = M.LAM-2*k*M.THETA
    if abs(a1) < 1e-14:
        mat=np.array([[B/2,a0],[-k,-B/2]])
        pq=expm(mat*t)@np.array([s0,1.])
        return float(pq[0]/pq[1])
    # Here k*a1<0, so the imaginary-argument J/Y basis is represented more
    # stably by modified I/K Bessel functions at a positive real argument.
    if k*a1 >= 0:
        raise ValueError("This pilot check expects k*a1 < 0")
    x0 = 2*tau*np.sqrt(-k*a1)
    x = x0*np.exp(-t/(2*tau))
    nu = tau*np.sqrt(B*B-4*k*a0)
    def log_i_prime(arg):
        return (ive(nu-1,arg)+ive(nu+1,arg))/(2*ive(nu,arg))
    def log_k_prime(arg):
        return -(kve(nu-1,arg)+kve(nu+1,arg))/(2*kve(nu,arg))
    # Normalize both basis functions to 1 at t=0; this avoids I/K scale separation.
    phi0, psi0 = ive(nu,x0), kve(nu,x0)
    phi = np.exp(x-x0)*ive(nu,x)/phi0
    psi = np.exp(x0-x)*kve(nu,x)/psi0
    dphi = -x/(2*tau)*phi*log_i_prime(x)
    dpsi = -x/(2*tau)*psi*log_k_prime(x)
    dphi0 = -x0/(2*tau)*log_i_prime(x0)
    dpsi0 = -x0/(2*tau)*log_k_prime(x0)
    d0 = -k*s0-B/2
    cphi, cpsi = np.linalg.solve(np.array([[1.,1.],[dphi0,dpsi0]]), np.array([1.,d0]))
    z = cphi*phi+cpsi*psi
    dz = cphi*dphi+cpsi*dpsi
    return float(-(B/2+dz/z)/k)


def exact_segment(y0, t, ip3_0, baseline, tau):
    """Exact modal flow for fixed calcium region and active quadratic hinge."""
    y0=np.asarray(y0,dtype=float)
    l,v=M.L,M.V
    Q=np.eye(3)-np.outer(v,l)
    b0=np.array([M.CO[0,0],M.CO[1,0],M.P[4]*(-1.12)]) + baseline*np.array(
        [M.CO[0,3],M.CO[1,3],M.P[4]])
    b1=(ip3_0-baseline)*np.array([M.CO[0,3],M.CO[1,3],M.P[4]])
    aug=np.zeros((5,5));aug[:3,:3]=M.A;aug[:3,3]=Q@b0;aug[:3,4]=Q@b1;aug[4,4]=-1/tau
    transverse=(expm(aug*t)@np.r_[Q@y0,1.,1.])[:3]
    s=riccati_bessel(float(l@y0),t,ip3_0,baseline,tau)
    return transverse+v*s


def main():
    initial_cases=[
        {"u":.009,"v":.51,"r":.009,"I0":.70,"Ibase":.45,"tau":20.},
        {"u":.009,"v":.51,"r":.009,"I0":1.15,"Ibase":.45,"tau":65.},
        {"u":.009,"v":.51,"r":.009,"I0":.70,"Ibase":.45,"tau":170.},
    ]
    checks=[]
    for case in initial_cases:
        y0=np.array([case["u"],case["v"],case["r"]])
        physical_c=M.CREF+M.SCALE*y0[0]
        times=[.001,.005,.01,.025]
        for t in times:
            protocol={"baseline":case["Ibase"],"tau":case["tau"]}
            I0=case["I0"]
            yinit=np.r_[y0,I0]
            sol=solve_ivp(lambda tt,z:M.model_rhs(z,{
                "baseline":protocol["baseline"],"tau":protocol["tau"]},True),
                (0,t),yinit,t_eval=[t],method="DOP853",rtol=5e-13,atol=5e-15,max_step=.0005)
            pred=exact_segment(y0,t,I0,case["Ibase"],case["tau"])
            err=float(np.max(np.abs(pred-sol.y[:3,-1])))
            # ensure the numerical reference stayed in first C region and on active side
            svals=M.L@sol.y[:3]
            low=bool(np.max(sol.y[0])<M.TH[0] and np.min(sol.y[0])>=0)
            active=bool(np.min(svals)>M.THETA)
            checks.append({**case,"duration_s":t,"max_state_abs_error":err,
                           "stayed_first_C_region":low,"stayed_active":active,
                           "C_uM_range":[float(M.CREF+M.SCALE*sol.y[0].min()),
                                         float(M.CREF+M.SCALE*sol.y[0].max())],
                           "s_range":[float(svals.min()),float(svals.max())]})
    output={"formula":"Bessel solution for Riccati scalar mode; 5x5 matrix exponential for transverse modes",
            "parameters":{"k":M.K,"theta":M.THETA,"lambda":M.LAM,"l":M.L.tolist(),"vslow":M.V.tolist()},
            "checks":checks}
    path=ROOT/"exponential_flow_check.json"
    path.write_text(__import__("json").dumps(output,indent=2)+"\n")
    for x in checks:print(x["tau"],x["I0"],x["duration_s"],x["max_state_abs_error"],x["stayed_first_C_region"],x["stayed_active"])
    print("saved",path)


if __name__=="__main__":main()
