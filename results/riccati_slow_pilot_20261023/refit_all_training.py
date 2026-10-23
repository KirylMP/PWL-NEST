"""Small local two-parameter refit scored on all ten dynamic training protocols."""
import json
import time
from pathlib import Path
from scipy.optimize import minimize
import numpy as np

import fit_dynamic as F

ROOT=Path(__file__).resolve().parent
F.FIT_TRAIN=F.TRAIN
prior=(-.5956250890184878,.4631857945184875)
counter=0;best=(float("inf"),None);started=time.perf_counter()


def objective(x):
    global counter,best
    counter+=1
    k,theta=map(float,x)
    if not (-1.5<=k<=-.02 and .32<=theta<=.65):return 1e4
    value=F.objective((k,theta))
    if value<best[0]:best=(value,(k,theta))
    print("local_eval",counter,"loss",round(value,5),"best",round(best[0],5),best[1],
          "elapsed",round(time.perf_counter()-started,1),flush=True)
    return value


def main():
    print("baseline full-training loss",objective(prior),flush=True)
    opt=minimize(objective,np.array(prior),method="Powell",
        bounds=[(-1.5,-.02),(.32,.65)],options={"maxfev":28,"xtol":.01,"ftol":.02,"disp":False})
    selected=best[1]
    print("finished",selected,"loss",best[0],"evaluations",counter,flush=True)
    prior_train=F.evaluate_set(F.TRAIN,*prior)
    selected_train=F.evaluate_set(F.TRAIN,*selected)
    prior_test=F.evaluate_set(F.TEST,*prior)
    # Read the selected candidate's test results only after optimization.
    selected_test=F.evaluate_set(F.TEST,*selected)
    prior_fixed=F.fixed_periods(*prior)
    selected_fixed=F.fixed_periods(*selected)
    out={"search":{"algorithm":"Powell local search","max_function_evaluations":28,
                   "actual_evaluations":counter,"bounds":[[-1.5,-.02],[.32,.65]],
                   "objective":"all ten dynamic training protocols; held-out protocols excluded",
                   "elapsed_s":time.perf_counter()-started,"optimizer_success":bool(opt.success),
                   "optimizer_message":str(opt.message)},
         "prior":{"k":prior[0],"theta":prior[1]},
         "selected":{"k":selected[0],"theta":selected[1],"training_loss":best[0]},
         "metrics":{"prior_train":prior_train,"selected_train":selected_train,
                     "prior_heldout":prior_test,"selected_heldout":selected_test,
                     "fixed_I_prior":prior_fixed,"fixed_I_selected":selected_fixed}}
    path=ROOT/"all_training_refit_results.json";path.write_text(json.dumps(out,indent=2)+"\n")
    print("train mean RMSE",prior_train["mean_C_RMSE_uM"],"->",selected_train["mean_C_RMSE_uM"],
          "heldout",prior_test["mean_C_RMSE_uM"],"->",selected_test["mean_C_RMSE_uM"],"saved",path)


if __name__=="__main__":main()
