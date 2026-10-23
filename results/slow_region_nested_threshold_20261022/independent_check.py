"""Independent fixed-input period check for the uploaded nested-hinge model."""
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "input_archive/slow_threshold_test"
BASE = json.loads((ARCHIVE / "validated.json").read_text())
AUGMENTED = json.loads((ARCHIVE / "augmented_results.json").read_text())
CO = np.asarray(BASE["fitted_coefficients"])
TH = np.asarray(BASE["thresholds"])
P = np.asarray(AUGMENTED["candidates"][0]["parameters"])
CREF, SCALE = .02812613673817445, .9833352639826316
HR = .8371177795140701
HS = HR*(1-HR)


def field(y, ip3, kind, beta, theta, k):
    u, v, r = y
    du = CO[0, 0] + CO[0, 1]*u + CO[0, 2]*v + CO[0, 3]*ip3
    dv = CO[1, 0] + CO[1, 1]*u + CO[1, 2]*v + CO[1, 3]*ip3
    for j, threshold in enumerate(TH):
        hinge = max(0.0, u-threshold)
        du += CO[0, 4+j]*hinge
        dv += CO[1, 4+j]*hinge
    du += P[0]*(r-u); dv += P[1]*(r-u)
    dr = P[2]*(u-r) + P[3]*v + P[4]*(ip3-1.12)
    s = u if kind == 0 else (r if kind == 1 else u+beta*v)
    hinge = max(0.0, theta-s)
    return [du+k[0]*hinge, dv+k[1]*hinge, dr+k[2]*hinge]

INPUTS = [0.53, 0.55, 0.7, 1.0, 1.12, 1.13]
HORIZON = 620.0
DT = 0.2


def metrics(t, c, h):
    peaks, _ = find_peaks(c, prominence=0.04, distance=35)
    late = peaks[t[peaks] > 320.0]
    late_amp = float(np.ptp(c[t > 400.0]))
    period = float(np.median(np.diff(t[late]))) if len(late) > 4 and late_amp > 0.12 else None
    return {"period_s": period, "late_amplitude_uM": late_amp,
            "C_min_uM": float(c.min()), "C_max_uM": float(c.max()),
            "h_min": float(h.min()), "h_max": float(h.max()), "late_peaks": int(len(late))}


def integrate_candidate(name, params, input_level):
    kind = {"C": 0, "r": 1, "tilt": 2}[name]
    y0 = np.array([(1.3-CREF)/SCALE, (.8-HR)/HS, (1.3-CREF)/SCALE])
    t = np.arange(0.0, HORIZON + DT/2, DT)
    sol = solve_ivp(lambda _, y: field(y, input_level, kind, params["beta"],
                                       params["theta"], np.asarray(params["k"])),
                    (0.0, HORIZON), y0, t_eval=t, method="DOP853", rtol=2e-9,
                    atol=2e-11, max_step=0.12)
    if not sol.success:
        raise RuntimeError(f"{name}, I={input_level}: {sol.message}")
    c = CREF + SCALE * sol.y[0]
    h = HR + HS * sol.y[1]
    return metrics(t, c, h)


def integrate_lr(input_level):
    def rhs(_, y):
        c, h = y
        q = 1.049*(input_level + .13)/(input_level + .9434)
        dc = (6*(input_level/(input_level+.13))**3*(c/(c+.08234))**3*h**3+.11)*(2-1.185*c)
        dc -= .9*c*c/(c*c+.05**2)
        dh = .2*(q-(q+c)*h)
        return [dc, dh]
    t = np.arange(0.0, HORIZON + DT/2, DT)
    sol = solve_ivp(rhs, (0.0, HORIZON), [1.3, .8], t_eval=t,
                    method="DOP853", rtol=2e-9, atol=2e-11, max_step=.12)
    if not sol.success:
        raise RuntimeError(f"LR I={input_level}: {sol.message}")
    return metrics(t, sol.y[0], sol.y[1])


if __name__ == "__main__":
    saved = json.loads((ARCHIVE / "validated_slow.json").read_text())
    # The saved C/r/tilt coefficients are in search_results.json; baseline is the
    # same augmented model with a zero hinge vector.
    fitted = json.loads((ARCHIVE / "search_results.json").read_text())
    baseline = {"theta": (0.037-CREF)/SCALE, "beta": 0.0, "k": [0, 0, 0]}
    candidates = {"baseline": baseline, **fitted}
    result = {
        "method": "DOP853, same initial state C=1.3 uM/h=0.8/r=u, horizon 620 s, dt output 0.2 s",
        "solver": {"rtol": 2e-9, "atol": 2e-11, "max_step_s": .12},
        "inputs": {},
    }
    for I in INPUTS:
        row = {"Li_Rinzel": integrate_lr(I)}
        for name, params in candidates.items():
            row[name] = integrate_candidate("C" if name == "baseline" else name, params, I)
        # Preserve archive's independently saved values so discrepancies are visible.
        row["archive_reported"] = {name: saved[name]["DOP853"][INPUTS.index(I)] for name in candidates}
        result["inputs"][str(I)] = row
        print(I, {k: (None if v["period_s"] is None else round(v["period_s"], 3))
                  for k, v in row.items() if k != "archive_reported"}, flush=True)
    out = ROOT / "independent_validation.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"saved {out}")
