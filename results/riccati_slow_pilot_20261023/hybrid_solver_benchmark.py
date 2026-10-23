"""Prototype hybrid event solver and benchmark against DOP853 for held-out FM."""
import importlib.util
import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import brentq
from scipy.special import ive, kve

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("dynamic_validation",ROOT/"dynamic_validation.py")
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
spec2=importlib.util.spec_from_file_location("exact_flow",ROOT/"check_exponential_flow.py")
E=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(E)
DATA=M.DATA


def affine_region(region):
    a=M.A.copy()
    b0=np.array([M.CO[0,0],M.CO[1,0],M.P[4]*(-1.12)],float)
    b1=np.array([M.CO[0,3],M.CO[1,3],M.P[4]],float)
    for j in range(region):
        a[0,0]+=M.CO[0,4+j]
        a[1,0]+=M.CO[1,4+j]
        b0[0]-=M.CO[0,4+j]*M.TH[j]
        b0[1]-=M.CO[1,4+j]*M.TH[j]
    return a,b0,b1


def region_id(y):
    return int(np.searchsorted(M.TH,y[0],side="right"))


def exact_inactive(y, dt, baseline, tau, region):
    a,b0,b1=affine_region(region)
    big=np.zeros((5,5));big[:3,:3]=a;big[:3,3]=b0+baseline*b1;big[:3,4]=(y[3]-baseline)*b1;big[4,4]=-1/tau
    z=expm(big*dt)@np.r_[y[:3],1.,1.]
    ip=baseline+(y[3]-baseline)*np.exp(-dt/tau)
    return np.r_[z[:3],ip]


def exact_active_low(y,dt,baseline,tau):
    # Modal Riccati and transverse affine flow in first calcium region.
    l,v=M.L,M.V;Q=np.eye(3)-np.outer(v,l)
    b0=np.array([M.CO[0,0],M.CO[1,0],M.P[4]*(-1.12)])+baseline*np.array([M.CO[0,3],M.CO[1,3],M.P[4]])
    b1=(y[3]-baseline)*np.array([M.CO[0,3],M.CO[1,3],M.P[4]])
    big=np.zeros((5,5));big[:3,:3]=M.A;big[:3,3]=Q@b0;big[:3,4]=Q@b1;big[4,4]=-1/tau
    p=(expm(big*dt)@np.r_[Q@y[:3],1.,1.])[:3]
    s=E.riccati_bessel(float(l@y[:3]),dt,y[3],baseline,tau)
    ip=baseline+(y[3]-baseline)*np.exp(-dt/tau)
    return np.r_[p+v*s,ip]


def flow(y,dt,protocol):
    reg=region_id(y)
    active=M.L@y[:3]>M.THETA
    if not active:
        return exact_inactive(y,dt,protocol["baseline"],protocol["tau"],reg),"exact_affine"
    if reg==0:
        return exact_active_low(y,dt,protocol["baseline"],protocol["tau"]),"exact_riccati"
    sol=solve_ivp(lambda _,z:M.model_rhs(z,protocol,True),(0,dt),y,t_eval=[dt],
                  method="DOP853",rtol=2e-10,atol=2e-12,max_step=min(.02,dt))
    if not sol.success:raise RuntimeError(sol.message)
    return sol.y[:,-1],"numeric_fallback"


def surfaces(y):
    return np.r_[y[0]-M.TH,M.L@y[:3]-M.THETA]


def advance(y,dt,protocol,counters,scan_step=.2):
    """Advance with small brackets; refine the first detected switching time."""
    elapsed=0.; remaining=dt
    while remaining>2e-11:
        h=min(scan_step,remaining)
        start=y.copy();g0=surfaces(start)
        end,mode=flow(start,h,protocol);counters[mode]+=1
        g1=surfaces(end)
        crossing=np.flatnonzero(g0*g1 < -1e-18)
        if len(crossing):
            # Locate each sign-changing plane on the current fixed-mode flow;
            # choose the earliest root. Same-side tangencies do not switch modes.
            roots=[]
            for ix in crossing:
                def fun(t):return surfaces(flow(start,t,protocol)[0])[ix]
                try:roots.append((brentq(fun,1e-12,h,xtol=1e-12),int(ix)))
                except ValueError:pass
            if roots:
                root,ix=min(roots)
                y=flow(start,root,protocol)[0]
                elapsed+=root;remaining-=root;counters["switches"]+=1
                # Take a tiny RK4 step to select the outgoing side at the surface.
                eps=min(1e-8,remaining)
                if eps>0:
                    f=lambda z:M.model_rhs(z,protocol,True)
                    k1=np.asarray(f(y));k2=np.asarray(f(y+.5*eps*k1));k3=np.asarray(f(y+.5*eps*k2));k4=np.asarray(f(y+eps*k3))
                    y=y+eps*(k1+2*k2+2*k3+k4)/6
                    elapsed+=eps;remaining-=eps
                continue
        y=end;elapsed+=h;remaining-=h
    return y


def hybrid_run(protocol,initial,scan_step=.2):
    duration=float(protocol["duration"]);y=np.array([(initial[0]-M.CREF)/M.SCALE,
        (initial[1]-M.HR)/M.HS,(initial[0]-M.CREF)/M.SCALE,initial[2]],float)
    events=sorted((float(t),float(d)) for t,d in protocol.get("jumps",[]))
    chunks_t=[];chunks_y=[];left=0.;counters={"exact_affine":0,"exact_riccati":0,"numeric_fallback":0,"switches":0}
    for right,delta in events+[(duration,0.)]:
        n=max(1,int(np.ceil((right-left)/.2)));grid=np.linspace(left,right,n+1)
        states=[y.copy()]
        for a,b in zip(grid[:-1],grid[1:]):
            y=advance(y,float(b-a),protocol,counters,scan_step);states.append(y.copy())
        chunks_t.append(grid if not chunks_t else grid[1:]);z=np.asarray(states)
        chunks_y.append(z if not chunks_y else z[1:])
        if right!=duration:y[3]+=delta
        left=right
    t,z=np.concatenate(chunks_t),np.vstack(chunks_y)
    physical=np.column_stack([M.CREF+M.SCALE*z[:,0],M.HR+M.HS*z[:,1],z[:,3]])
    return t,physical,counters


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--candidate",choices=["incumbent","local_refit"],default="incumbent")
    args=parser.parse_args()
    if args.candidate=="local_refit":
        fit=json.loads((ROOT/"all_training_refit_results.json").read_text())["selected"]
        M.K=float(fit["k"]);M.THETA=float(fit["theta"])
        E.M.K=M.K;E.M.THETA=M.THETA
    else:
        M.K=-.5956250890184878;M.THETA=.4631857945184875
        E.M.K=M.K;E.M.THETA=M.THETA
    protocols=DATA["test"]
    rows=[]
    for p in protocols:
        cache=np.load(Path("results/fm_phase_continuation/cache/FM")/(p["name"]+".npz"));initial=cache["x"][0]
        # Compare the production .2 s scan with a finer .025 s reference scan.
        hybrid_run({**p,"duration":.2,"jumps":[]},initial)
        t0=time.perf_counter();ht,hx,counters=hybrid_run(p,initial,.2);coarse_time=time.perf_counter()-t0
        t0=time.perf_counter();ft,fx,fine_counters=hybrid_run(p,initial,.025);fine_time=time.perf_counter()-t0
        t0=time.perf_counter();dt,dx=M.run(p,initial,"riccati");dop_seconds=time.perf_counter()-t0
        hybrid_runs=[];dop_runs=[]
        for _ in range(2):
            t0=time.perf_counter();ht,hx,counters=hybrid_run(p,initial,.2);hybrid_runs.append(time.perf_counter()-t0)
            t0=time.perf_counter();dt,dx=M.run(p,initial,"riccati");dop_runs.append(time.perf_counter()-t0)
        hybrid_seconds=float(np.median(hybrid_runs));dop_seconds=float(np.median(dop_runs))
        if len(ht)!=len(dt) or np.max(abs(ht-dt))>1e-9:raise RuntimeError("Output grids differ")
        err=np.sqrt(np.mean((hx[:,0]-dx[:,0])**2))
        maxerr=float(np.max(abs(hx[:,0]-dx[:,0])))
        scan_diff=float(np.sqrt(np.mean((hx[:,0]-fx[:,0])**2)))
        rows.append({"protocol":p["name"],"hybrid_seconds_median":hybrid_seconds,"DOP853_seconds_median":dop_seconds,
            "hybrid_seconds_repeats":hybrid_runs,"DOP853_seconds_repeats":dop_runs,
            "speedup_vs_DOP853":dop_seconds/hybrid_seconds,"C_RMSE_vs_DOP853_uM":float(err),
            "C_max_abs_vs_DOP853_uM":maxerr,"C_RMSE_coarse_vs_fine_scan_uM":scan_diff,
            "coarse_vs_fine_switch_count":[counters["switches"],fine_counters["switches"]],
            "fine_scan_seconds":fine_time,"coarse_scan_seconds":coarse_time,"counters":counters})
        print(p["name"],"speed",round(dop_seconds/hybrid_seconds,3),"RMSE/DOP",f"{err:.3g}",
              "coarse/fine",f"{scan_diff:.3g}","switches",counters["switches"],fine_counters["switches"],flush=True)
    report={"candidate":args.candidate,"parameters":{"k":M.K,"theta":M.THETA},
            "note":"Prototype hybrid; exact affine and Bessel modal segments, DOP853 fallback when quadratic correction is active outside first C-region; .2 s and .025 s scans bracket roots and refine them with Brent.",
            "protocol_count":len(rows),"mean_speedup_vs_DOP853":float(np.mean([r["speedup_vs_DOP853"] for r in rows])),
            "median_speedup_vs_DOP853":float(np.median([r["speedup_vs_DOP853"] for r in rows])),
            "max_C_RMSE_vs_DOP853_uM":max(r["C_RMSE_vs_DOP853_uM"] for r in rows),"protocols":rows}
    target=ROOT/("hybrid_solver_benchmark.json" if args.candidate=="incumbent" else "hybrid_solver_refit_benchmark.json")
    target.write_text(json.dumps(report,indent=2)+"\n")
    print("saved",target)


if __name__=="__main__":main()
