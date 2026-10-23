"""Run the uploaded hidden-state/hinge FM model on the saved full FM suite."""
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "input_archive/slow_threshold_test"
DATA = json.loads(Path("results/fm_phase_continuation/cache/FM/data.json").read_text())
BASE = json.loads((ARCHIVE / "validated.json").read_text())
AUGMENTED = json.loads((ARCHIVE / "augmented_results.json").read_text())
SEARCH = json.loads((ARCHIVE / "search_results.json").read_text())
CO, TH = np.asarray(BASE["fitted_coefficients"]), np.asarray(BASE["thresholds"])
P = np.asarray(AUGMENTED["candidates"][0]["parameters"])
CREF, SCALE, HR = .02812613673817445, .9833352639826316, .8371177795140701
HS = HR*(1-HR)
DT = .2


def rhs(state, ip3, candidate):
    u, v, r = state[:3]
    du = CO[0, 0] + CO[0, 1]*u + CO[0, 2]*v + CO[0, 3]*ip3
    dv = CO[1, 0] + CO[1, 1]*u + CO[1, 2]*v + CO[1, 3]*ip3
    for j, threshold in enumerate(TH):
        z = max(0., u-threshold)
        du += CO[0, 4+j]*z
        dv += CO[1, 4+j]*z
    du += P[0]*(r-u)
    dv += P[1]*(r-u)
    dr = P[2]*(u-r) + P[3]*v + P[4]*(ip3-1.12)
    if candidate != "baseline":
        params = SEARCH[candidate]
        if candidate == "C":
            surface = u
        elif candidate == "r":
            surface = r
        else:
            surface = u + params["beta"]*v
        hinge = max(0., params["theta"]-surface)
        du += params["k"][0]*hinge
        dv += params["k"][1]*hinge
        dr += params["k"][2]*hinge
    d_ip3 = 0.0  # set by protocol tau in integrate_segment wrapper
    return np.array([du, dv, dr, d_ip3])


def integrate(protocol, initial, candidate):
    duration, tau, baseline = protocol["duration"], protocol["tau"], protocol["baseline"]
    events = sorted((float(t), float(delta)) for t, delta in protocol.get("jumps", []))
    y = np.array([(initial[0]-CREF)/SCALE, (initial[1]-HR)/HS,
                  (initial[0]-CREF)/SCALE, initial[2]], dtype=float)
    parts_t, parts_y = [], []
    left = 0.0
    for right, delta in events + [(float(duration), 0.0)]:
        n = max(1, int(np.ceil((right-left)/DT)))
        times = np.linspace(left, right, n+1)
        def fun(t, state):
            z = rhs(state, state[3], candidate)
            z[3] = -(state[3]-baseline)/tau
            return z
        sol = solve_ivp(fun, (left, right), y, t_eval=times, method="DOP853",
                        rtol=2e-8, atol=2e-10, max_step=.2)
        if not sol.success or not np.isfinite(sol.y).all():
            raise RuntimeError(f"{protocol['name']} {candidate}: {sol.message}")
        parts_t.append(sol.t if not parts_t else sol.t[1:])
        parts_y.append(sol.y.T if not parts_y else sol.y.T[1:])
        y = sol.y[:, -1].copy()
        if right != duration:
            y[3] += delta
        left = right
    t, z = np.concatenate(parts_t), np.vstack(parts_y)
    physical = np.column_stack((CREF+SCALE*z[:, 0], HR+HS*z[:, 1], z[:, 3]))
    return t, physical


def initial_and_protocol(item, section):
    protocol = dict(item)
    if section == "fixed":
        return protocol["initial"], protocol
    cached = np.load(Path("results/fm_phase_continuation/cache/FM")/(item["name"]+".npz"))
    return cached["x"][0], protocol


def measure(t, reference, output, section):
    c0, c1 = reference[:, 0], output[:, 0]
    err = c1-c0
    peaks0, _ = find_peaks(c0, prominence=.02)
    peaks1, _ = find_peaks(c1, prominence=.02)
    row = {
        "rmse_C_uM": float(np.sqrt(np.mean(err**2))),
        "rmse_h": float(np.sqrt(np.mean((output[:, 1]-reference[:, 1])**2))),
        "max_abs_C_error_uM": float(np.max(np.abs(err))),
        "peak_count_LR": int(len(peaks0)), "peak_count_model": int(len(peaks1)),
        "peak_time_mae_s": (float(np.mean(np.abs(t[peaks1]-t[peaks0])))
                            if len(peaks0) == len(peaks1) and len(peaks0) else None),
        "C_range_uM": [float(c1.min()), float(c1.max())],
        "h_range": [float(output[:, 1].min()), float(output[:, 1].max())],
        "ip3_range": [float(output[:, 2].min()), float(output[:, 2].max())],
        "within_physical_C_h_bounds": bool(c1.min() >= 0 and c1.max() <= 2/1.185 and
                                           output[:, 1].min() >= 0 and output[:, 1].max() <= 1),
    }
    if section == "fixed":
        mask = t >= t[-1]-400
        q0, _ = find_peaks(c0[mask], prominence=.001)
        q1, _ = find_peaks(c1[mask], prominence=.001)
        periods = [float(np.median(np.diff(t[mask][q]))) if len(q) >= 3 else None for q in (q0, q1)]
        row["late_period_LR_model_s"] = periods
        row["late_status_LR_model"] = ["cycling" if x is not None else "rest/unresolved" for x in periods]
    return row


def main():
    suite = []
    for section in ("fixed", "train", "test"):
        for item in DATA[section]:
            initial, protocol = initial_and_protocol(item, section)
            suite.append((section, protocol, np.asarray(initial, dtype=float)))
    # Include the exact saved FM resting equilibrium as a constant-input check.
    rest_initial = np.asarray(DATA["rest"], dtype=float)
    suite.append(("rest", {"name":"saved_rest_equilibrium", "duration":1600., "tau":20.,
                           "baseline":float(rest_initial[2])}, rest_initial))
    names = ["baseline", "C", "r", "tilt"]
    results = []
    aggregate = {name: {section: [] for section in ("fixed", "train", "test", "rest")} for name in names}
    for section, protocol, initial in suite:
        # Compare candidates against exact LR, using the same jump boundaries and output grid.
        # The first state in the augmented coordinate is converted from physical C,h,I.
        # LR gets an additional passive r component solely for shared integration bookkeeping.
        # Use a local integration whose C,h,I derivatives are the published nonlinear equations.
        def lr_protocol():
            tau, baseline = protocol["tau"], protocol["baseline"]
            y = initial.copy(); times_all=[]; states_all=[]; left=0.
            for right, delta in sorted((float(a),float(b)) for a,b in protocol.get("jumps",[]))+[(float(protocol["duration"]),0.)]:
                grid=np.linspace(left,right,max(2,int(np.ceil((right-left)/DT))+1))
                def f(_, x):
                    c,h,ip3=x
                    q=1.049*(ip3+.13)/(ip3+.9434)
                    dc=(6*(ip3/(ip3+.13))**3*(c/(c+.08234))**3*h**3+.11)*(2-1.185*c)-.9*c*c/(c*c+.05**2)
                    dh=.2*(q-(q+c)*h)
                    return [dc,dh,-(ip3-baseline)/tau]
                sol=solve_ivp(f,(left,right),y,t_eval=grid,method="DOP853",rtol=2e-8,atol=2e-10,max_step=.2)
                if not sol.success: raise RuntimeError(f"LR {protocol['name']}: {sol.message}")
                times_all.append(sol.t if not times_all else sol.t[1:]);states_all.append(sol.y.T if not states_all else sol.y.T[1:])
                y=sol.y[:,-1].copy()
                if right != protocol["duration"]: y[2]+=delta
                left=right
            return np.concatenate(times_all),np.vstack(states_all)
        t, lr = lr_protocol()
        row = {"section":section,"protocol":protocol["name"],"initial":initial.tolist(),"metrics":{}}
        for name in names:
            _, candidate = integrate(protocol, initial, name)
            metrics = measure(t, lr, candidate, section)
            row["metrics"][name] = metrics
            aggregate[name][section].append(metrics)
        results.append(row)
        print(section, protocol["name"], "RMSE C", {n:round(row["metrics"][n]["rmse_C_uM"],4) for n in names}, flush=True)
    summary = {}
    for name in names:
        summary[name] = {}
        for section, metrics in aggregate[name].items():
            if not metrics: continue
            rmse = np.array([m["rmse_C_uM"] for m in metrics])
            summary[name][section] = {
                "protocol_count": len(metrics), "mean_C_RMSE_uM": float(rmse.mean()),
                "max_C_RMSE_uM": float(rmse.max()),
                "peak_count_matches": int(sum(m["peak_count_LR"] == m["peak_count_model"] for m in metrics)),
                "peak_time_MAE_mean_s": (float(np.mean([m["peak_time_mae_s"] for m in metrics if m["peak_time_mae_s"] is not None]))
                                          if any(m["peak_time_mae_s"] is not None for m in metrics) else None),
                "physical_bound_passes": int(sum(m["within_physical_C_h_bounds"] for m in metrics)),
            }
    output = {"description":"Full saved FM fixed/train/held-out protocol suite; DOP853 on uploaded 4-region PWA + latent r + each one-hinge candidate, compared with Li-Rinzel.",
              "solver":{"output_dt_s":DT,"rtol":2e-8,"atol":2e-10,"max_step_s":.2},
              "suite_counts":{"fixed":len(DATA["fixed"]),"train":len(DATA["train"]),"test":len(DATA["test"]),"rest":1},
              "summary":summary,"protocols":results}
    target=HERE/"full_protocol_validation.json"
    target.write_text(json.dumps(output,indent=2)+"\n")
    print("saved",target)


if __name__ == "__main__":
    main()
