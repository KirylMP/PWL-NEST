"""Limited two-parameter Riccati refit on saved FM training protocols only."""
import json
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
from scipy.signal import find_peaks

import dynamic_validation as M

ROOT=Path(__file__).resolve().parent
DATA=M.DATA
TRAIN=DATA["train"]
TEST=DATA["test"]
FIT_NAMES={"pulse_tau_100.0","promoted_tau_35.0","continuation_heldout_irregular"}
FIT_TRAIN=[p for p in TRAIN if p["name"] in FIT_NAMES]
INITIALS={p["name"]:np.load(Path("results/fm_phase_continuation/cache/FM")/(p["name"]+".npz"))["x"][0]
          for p in TRAIN+TEST}
REFERENCE={p["name"]:np.load(Path("results/fm_phase_continuation/cache/FM")/(p["name"]+".npz"))
           for p in TRAIN+TEST}
DT=.5
EVALUATIONS=0
BEST=(float("inf"),None)
START_TIME=0.


def simulate(protocol,initial,k,theta):
    y=np.array([(initial[0]-M.CREF)/M.SCALE,(initial[1]-M.HR)/M.HS,
                (initial[0]-M.CREF)/M.SCALE,initial[2]],float)
    out_t=[];out_y=[];left=0.
    for right,delta in sorted((float(t),float(d)) for t,d in protocol.get("jumps",[]))+[(float(protocol["duration"]),0.)]:
        count=max(1,int(np.ceil((right-left)/DT)));grid=np.linspace(left,right,count+1)
        def f(_,z):
            if not np.isfinite(z).all() or np.max(np.abs(z[:3]))>12 or z[3]<-.2 or z[3]>1.5:
                raise FloatingPointError(f"candidate escaped bounded search domain: {z}")
            return M.model_rhs(z,protocol,True)
        sol=solve_ivp(f,(left,right),y,t_eval=grid,method="DOP853",rtol=2e-7,atol=2e-9,max_step=.12)
        if not sol.success or not np.isfinite(sol.y).all():raise RuntimeError(sol.message)
        out_t.append(sol.t if not out_t else sol.t[1:]);out_y.append(sol.y.T if not out_y else sol.y.T[1:])
        y=sol.y[:,-1].copy()
        if right!=protocol["duration"]:y[3]+=delta
        left=right
    t,z=np.concatenate(out_t),np.vstack(out_y)
    c=M.CREF+M.SCALE*z[:,0]
    return t,c


def score_one(protocol,k,theta):
    M.K=float(k);M.THETA=float(theta)
    t,c=simulate(protocol,INITIALS[protocol["name"]],k,theta)
    ref=REFERENCE[protocol["name"]]
    tc,cc=ref["t"],ref["x"][:,0]
    truth=np.interp(t,tc,cc)
    rmse=float(np.sqrt(np.mean((c-truth)**2)))
    p0,_=find_peaks(truth,prominence=.02,distance=8)
    p1,_=find_peaks(c,prominence=.02,distance=8)
    nerr=abs(len(p0)-len(p1))
    time_mae=float(np.mean(abs(t[p0]-t[p1]))) if len(p0)==len(p1) and len(p0) else None
    # Trajectory fit dominates; count mismatches and timing drift receive explicit costs.
    cost=(rmse/.1)**2+3.0*nerr
    if time_mae is not None:cost+=.35*(time_mae/1.0)**2
    return cost,{"name":protocol["name"],"rmse_C_uM":rmse,"peak_counts":[int(len(p0)),int(len(p1))],
                 "peak_time_mae_s":time_mae,"max_C_uM":float(c.max()),"min_C_uM":float(c.min())}


def objective(x):
    global EVALUATIONS,BEST
    EVALUATIONS+=1
    k,theta=map(float,x)
    try:
        records=[score_one(p,k,theta) for p in FIT_TRAIN]
        value=float(np.mean([r[0] for r in records]))
        if not np.isfinite(value) or max(r[1]["max_C_uM"] for r in records)>1.8:return 1e4
        if value<BEST[0]:BEST=(value,(k,theta))
        if EVALUATIONS%8==0:print("eval",EVALUATIONS,"current best",round(BEST[0],5),BEST[1],"elapsed_s",round(time.perf_counter()-START_TIME,1),flush=True)
        return value
    except (RuntimeError,ValueError,FloatingPointError,OverflowError):
        return 1e4


def evaluate_set(protocols,k,theta):
    rows=[]
    for p in protocols:
        _,row=score_one(p,k,theta);rows.append(row)
    return {"mean_C_RMSE_uM":float(np.mean([r["rmse_C_uM"] for r in rows])),
            "peak_count_matches":sum(a==b for r in rows for a,b in [r["peak_counts"]]),
            "protocol_count":len(rows),
            "mean_peak_time_MAE_s":float(np.mean([r["peak_time_mae_s"] for r in rows if r["peak_time_mae_s"] is not None])) if any(r["peak_time_mae_s"] is not None for r in rows) else None,
            "protocols":rows}


def fixed_periods(k,theta):
    from scipy.signal import find_peaks
    inputs=[.53,.55,.7,1.,1.12,1.13]
    rows=[]
    M.K=float(k);M.THETA=float(theta)
    for I in inputs:
        protocol={"name":f"fixed_{I}","duration":620.,"tau":20.,"baseline":I,"jumps":[]}
        initial=np.array([1.3,.8,I])
        # The fixed input means exact relaxation is identically zero.
        y0=np.array([(1.3-M.CREF)/M.SCALE,(.8-M.HR)/M.HS,(1.3-M.CREF)/M.SCALE,I])
        grid=np.arange(0,620.0001,.2)
        sol=solve_ivp(lambda _,z:M.model_rhs(z,protocol,True),(0,620),y0,t_eval=grid,
                      method="DOP853",rtol=2e-9,atol=2e-11,max_step=.12)
        c=M.CREF+M.SCALE*sol.y[0]
        pk,_=find_peaks(c,prominence=.04,distance=35);late=pk[grid[pk]>320]
        amp=float(np.ptp(c[grid>400]))
        period=float(np.median(np.diff(grid[late]))) if len(late)>4 and amp>.12 else None
        rows.append({"I":I,"period_s":period,"late_amplitude_uM":amp,"Cmax_uM":float(c.max())})
    return rows


def main():
    global START_TIME
    prior=(-.5956250890184878,.4631857945184875)
    # Fixed seeded population, with the already reported candidate included.
    rng=np.random.default_rng(20261023)
    population=np.column_stack([rng.uniform(-1.5,-.02,8),rng.uniform(.32,.65,8)])
    population[0]=prior
    t0=time.perf_counter()
    START_TIME=time.perf_counter()
    print("initial dynamic-only loss",objective(prior),flush=True)
    opt=differential_evolution(objective,[(-1.5,-.02),(.32,.65)],init=population,
        maxiter=4,popsize=4,polish=False,seed=20261023,workers=1,updating="immediate")
    best=tuple(map(float,opt.x));elapsed=time.perf_counter()-t0
    print("fit complete",best,"loss",opt.fun,"seconds",elapsed,flush=True)
    M.K,M.THETA=prior
    base_train=evaluate_set(TRAIN,*prior)
    base_test=evaluate_set(TEST,*prior)
    M.K,M.THETA=best
    fit_train=evaluate_set(TRAIN,*best)
    # Held-out test is evaluated only after optimization and never enters objective.
    fit_test=evaluate_set(TEST,*best)
    fixed_before=fixed_periods(*prior);fixed_after=fixed_periods(*best)
    out={"search":{"algorithm":"differential_evolution","seed":20261023,"maxiter":4,
                   "population_size":8,"evaluations":int(opt.nfev),"bounds":[[-1.5,-.02],[.32,.65]],
                   "fit_protocols":sorted(FIT_NAMES),
                   "objective":"three representative training dynamic protocols only: mean normalized C RMSE + explicit peak-count and peak-time penalties",
                   "elapsed_s":elapsed},
         "prior":{"k":prior[0],"theta":prior[1]},"fitted":{"k":best[0],"theta":best[1],"loss":float(opt.fun)},
         "metrics":{"prior_train":base_train,"fitted_train":fit_train,
                     "prior_heldout":base_test,"fitted_heldout":fit_test,
                     "fixed_I_prior":fixed_before,"fixed_I_fitted":fixed_after}}
    path=ROOT/"dynamic_fit_results.json";path.write_text(json.dumps(out,indent=2)+"\n")
    print("train RMSE",base_train["mean_C_RMSE_uM"],"->",fit_train["mean_C_RMSE_uM"],
          "heldout",base_test["mean_C_RMSE_uM"],"->",fit_test["mean_C_RMSE_uM"],flush=True)
    print("saved",path)


if __name__=="__main__":main()
