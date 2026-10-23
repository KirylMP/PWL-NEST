"""High-accuracy recheck of the small all-training refit and incumbent."""
import json
from pathlib import Path
import numpy as np
import dynamic_validation as M

ROOT=Path(__file__).resolve().parent
DATA=M.DATA
RESULT=json.loads((ROOT/"all_training_refit_results.json").read_text())
PRIOR=RESULT["prior"]; NEW=RESULT["selected"]


def eval_protocols(protocols,params):
    M.K=params["k"];M.THETA=params["theta"]
    records=[]
    for p in protocols:
        cache=np.load(Path("results/fm_phase_continuation/cache/FM")/(p["name"]+".npz"))
        t,lr=M.run(p,cache["x"][0],"LR")
        _,pred=M.run(p,cache["x"][0],"riccati")
        records.append({"protocol":p["name"],"metrics":M.compare(t,lr,pred)})
    rmse=[r["metrics"]["C_RMSE_uM"] for r in records]
    peaks=sum(r["metrics"]["peak_count_LR_model"][0]==r["metrics"]["peak_count_LR_model"][1] for r in records)
    phase=[r["metrics"]["peak_time_MAE_s"] for r in records if r["metrics"]["peak_time_MAE_s"] is not None]
    return {"mean_C_RMSE_uM":float(np.mean(rmse)),"peak_count_matches":peaks,
            "protocol_count":len(records),"mean_peak_time_MAE_s":float(np.mean(phase)) if phase else None,
            "protocols":records}


def main():
    output={"solver":{"method":"DOP853","rtol":2e-9,"atol":2e-11,"max_step_s":.12,"output_dt_s":.2},"candidates":{}}
    for label,params in [("incumbent",PRIOR),("local_refit",NEW)]:
        output["candidates"][label]={"parameters":params,
            "train":eval_protocols(DATA["train"],params),
            "heldout":eval_protocols(DATA["test"],params)}
    path=ROOT/"high_accuracy_refit_validation.json";path.write_text(json.dumps(output,indent=2)+"\n")
    for label,x in output["candidates"].items():
        print(label,"train",x["train"]["mean_C_RMSE_uM"],x["train"]["peak_count_matches"],
              "heldout",x["heldout"]["mean_C_RMSE_uM"],x["heldout"]["peak_count_matches"],flush=True)
    print("saved",path)


if __name__=="__main__":main()
