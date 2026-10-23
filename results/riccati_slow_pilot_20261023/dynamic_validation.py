"""Independent DOP853 evaluation of the uploaded modal-Riccati FM model."""
import json
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parent
ARCH = ROOT / "input_archive"
DATA = json.loads(Path("results/fm_phase_continuation/cache/FM/data.json").read_text())
BASE = json.loads((ARCH / "validated.json").read_text())
AUG = json.loads((ARCH / "augmented_results.json").read_text())
CO, TH = np.asarray(BASE["fitted_coefficients"]), np.asarray(BASE["thresholds"])
P = np.asarray(AUG["candidates"][0]["parameters"])
K, THETA = -0.5956250890184878, 0.4631857945184875
CREF, SCALE, HR = .02812613673817445, .9833352639826316, .8371177795140701
HS = HR*(1-HR)

# Jacobian in the first calcium region, in the archived (u,v,r) coordinates.
A = np.zeros((3, 3))
A[:2, :2] = CO[:, 1:3]
A[0, 0] -= P[0]; A[0, 2] += P[0]
A[1, 0] -= P[1]; A[1, 2] += P[1]
A[2] = [P[2], P[3], -P[2]]
evals, vecs = np.linalg.eig(A)
idx = np.argmin(np.abs(evals + .1))
LAM = float(evals[idx].real)
V = vecs[:, idx].real
L = np.linalg.inv(vecs)[idx, :].real
if L @ V < 0:
    V, L = -V, -L
V /= L @ V


def model_rhs(state, protocol, corrected):
    u, v, r, ip3 = state
    du = CO[0, 0] + CO[0, 1]*u + CO[0, 2]*v + CO[0, 3]*ip3
    dv = CO[1, 0] + CO[1, 1]*u + CO[1, 2]*v + CO[1, 3]*ip3
    for j, threshold in enumerate(TH):
        hinge = max(0., u-threshold)
        du += CO[0, 4+j]*hinge
        dv += CO[1, 4+j]*hinge
    du += P[0]*(r-u); dv += P[1]*(r-u)
    dr = P[2]*(u-r) + P[3]*v + P[4]*(ip3-1.12)
    if corrected:
        s = float(L @ np.array([u, v, r]))
        q = max(0., s-THETA)
        du += K*V[0]*q*q
        dv += K*V[1]*q*q
        dr += K*V[2]*q*q
    return [du, dv, dr, -(ip3-protocol["baseline"])/protocol["tau"]]


def lr_rhs(state, protocol):
    c, h, ip3 = state
    q = 1.049*(ip3+.13)/(ip3+.9434)
    dc = (6*(ip3/(ip3+.13))**3*(c/(c+.08234))**3*h**3+.11)*(2-1.185*c)
    dc -= .9*c*c/(c*c+.05**2)
    dh = .2*(q-(q+c)*h)
    return [dc, dh, -(ip3-protocol["baseline"])/protocol["tau"]]


def run(protocol, initial, kind):
    duration = float(protocol["duration"])
    y = np.array(initial, dtype=float)
    if kind != "LR":
        y = np.array([(y[0]-CREF)/SCALE, (y[1]-HR)/HS,
                      (y[0]-CREF)/SCALE, y[2]])
    points_t, points_y = [], []
    left = 0.
    for right, delta in sorted((float(t), float(d)) for t, d in protocol.get("jumps", [])) + [(duration, 0.)]:
        count = max(1, int(np.ceil((right-left)/.2)))
        times = np.linspace(left, right, count+1)
        f = (lambda _, z: lr_rhs(z, protocol)) if kind == "LR" else (
            lambda _, z: model_rhs(z, protocol, kind == "riccati"))
        sol = solve_ivp(f, (left, right), y, t_eval=times, method="DOP853",
                        rtol=2e-9, atol=2e-11, max_step=.12)
        if not sol.success or not np.isfinite(sol.y).all():
            raise RuntimeError(f"{protocol['name']} {kind}: {sol.message}")
        points_t.append(sol.t if not points_t else sol.t[1:])
        points_y.append(sol.y.T if not points_y else sol.y.T[1:])
        y = sol.y[:, -1].copy()
        if right != duration:
            y[-1] += delta
        left = right
    t, z = np.concatenate(points_t), np.vstack(points_y)
    if kind == "LR":
        phys = z
    else:
        us = z[:, :3]
        slow_mode = us@L
        phys = np.column_stack([CREF+SCALE*z[:, 0], HR+HS*z[:, 1], z[:, 3],
                                z[:, 2], slow_mode])
    return t, phys


def compare(t, truth, pred):
    c0, c1 = truth[:, 0], pred[:, 0]
    p0, _ = find_peaks(c0, prominence=.02)
    p1, _ = find_peaks(c1, prominence=.02)
    same = len(p0) == len(p1)
    result = {
        "C_RMSE_uM": float(np.sqrt(np.mean((c1-c0)**2)),),
        "C_max_abs_error_uM": float(np.max(np.abs(c1-c0))),
        "h_RMSE": float(np.sqrt(np.mean((pred[:, 1]-truth[:, 1])**2))),
        "peak_count_LR_model": [int(len(p0)), int(len(p1))],
        "peak_time_MAE_s": float(np.mean(np.abs(t[p0]-t[p1]))) if same and len(p0) else None,
        "C_range_uM": [float(c1.min()), float(c1.max())],
        "h_range": [float(pred[:, 1].min()), float(pred[:, 1].max())],
        "IP3_range": [float(pred[:, 2].min()), float(pred[:, 2].max())],
    }
    if pred.shape[1] >= 5:
        active = pred[:, 4] > THETA
        low_region = pred[:, 0] < CREF+SCALE*TH[0]
        result.update({
            "quadratic_active_fraction": float(active.mean()),
            "active_outside_first_C_region_fraction": float((active & ~low_region).mean()),
            "first_C_region_fraction": float(low_region.mean()),
            "latent_r_range": [float(pred[:, 3].min()), float(pred[:, 3].max())],
        })
    return result


def main():
    records = []
    for section in ("train", "test"):
        for p in DATA[section]:
            cached = np.load(Path("results/fm_phase_continuation/cache/FM")/(p["name"]+".npz"))
            initial = cached["x"][0]
            t, lr = run(p, initial, "LR")
            _, base = run(p, initial, "base")
            _, ric = run(p, initial, "riccati")
            records.append({"section": section, "protocol": p["name"],
                            "initial": initial.tolist(),
                            "base": compare(t, lr, base), "riccati": compare(t, lr, ric)})
            print(section, p["name"],
                  "C-RMSE", round(records[-1]["base"]["C_RMSE_uM"], 4),
                  "->", round(records[-1]["riccati"]["C_RMSE_uM"], 4),
                  "peaks", records[-1]["riccati"]["peak_count_LR_model"], flush=True)
    summary = {}
    for section in ("train", "test"):
        sub = [r for r in records if r["section"] == section]
        summary[section] = {}
        for name in ("base", "riccati"):
            e = np.array([r[name]["C_RMSE_uM"] for r in sub])
            summary[section][name] = {
                "mean_C_RMSE_uM": float(e.mean()),
                "peak_count_matches": sum(r[name]["peak_count_LR_model"][0] == r[name]["peak_count_LR_model"][1] for r in sub),
                "protocols": len(sub),
                "mean_peak_time_MAE_s": (float(np.mean([r[name]["peak_time_MAE_s"] for r in sub if r[name]["peak_time_MAE_s"] is not None]))
                                          if any(r[name]["peak_time_MAE_s"] is not None for r in sub) else None),
            }
    out = {"parameters": {"k": K, "theta": THETA, "lambda": LAM, "left": L.tolist(), "right": V.tolist()},
           "solver": {"method":"DOP853", "rtol":2e-9, "atol":2e-11, "max_step_s":.12, "output_dt_s":.2},
           "summary": summary, "protocols": records}
    path = ROOT/"dynamic_protocol_validation.json"
    path.write_text(json.dumps(out, indent=2)+"\n")
    print("saved", path)


if __name__ == "__main__":
    main()
