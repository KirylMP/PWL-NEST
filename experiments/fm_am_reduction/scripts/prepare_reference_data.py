#!/usr/bin/env python3
"""Prepare immutable LR nullcline, equilibrium and spectrum data per planned run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from plan import PACKAGE, ROOT, read_json

OUTPUT = ROOT / "runs" / "fm_am_reduction"


def lr_field(c: np.ndarray | float, h: np.ndarray | float, ip3: np.ndarray | float,
             p: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    c, h, ip3 = np.asarray(c), np.asarray(h), np.asarray(ip3)
    m = ip3 / (ip3 + p["d_1"])
    n = c / (c + p["d_5"])
    q = p["d_2"] * (ip3 + p["d_1"]) / (ip3 + p["d_3"])
    dc = (p["r_C"] * m**3 * n**3 * h**3 + p["r_L"]) * (p["C_0"] - (1+p["c_1"])*c)
    dc -= p["v_ER"] * c**2 / (c**2 + p["K_ER"]**2)
    dh = p["a_2"] * (q - (q+c)*h)
    return dc, dh


def equilibrium_h(c: float, ip3: float, p: dict[str, float]) -> float:
    q = p["d_2"] * (ip3+p["d_1"]) / (ip3+p["d_3"])
    return q/(q+c)


def jacobian(c: float, h: float, ip3: float, p: dict[str, float]) -> np.ndarray:
    m = ip3/(ip3+p["d_1"])
    n = c/(c+p["d_5"])
    q = p["d_2"]*(ip3+p["d_1"])/(ip3+p["d_3"])
    drive = p["C_0"]-(1+p["c_1"])*c
    conductance = p["r_C"]*m**3*n**3*h**3+p["r_L"]
    df_dc = (p["r_C"]*m**3*3*n**2*p["d_5"]/(c+p["d_5"])**2*h**3*drive
             - conductance*(1+p["c_1"])
             - p["v_ER"]*2*c*p["K_ER"]**2/(c*c+p["K_ER"]**2)**2)
    df_dh = 3*p["r_C"]*m**3*n**3*h**2*drive
    return np.array([[df_dc, df_dh], [-p["a_2"]*h, -p["a_2"]*(q+c)]])


def roots_for_ip3(ip3: float, p: dict[str, float], c_max: float, resolution: int) -> list[dict]:
    grid = np.linspace(1e-9, c_max-1e-9, resolution)
    h_grid = np.array([equilibrium_h(float(c), ip3, p) for c in grid])
    f_grid, _ = lr_field(grid, h_grid, ip3, p)
    roots: list[dict] = []
    for lo, hi, flo, fhi in zip(grid[:-1], grid[1:], f_grid[:-1], f_grid[1:]):
        if flo == 0:
            c = float(lo)
        elif flo*fhi < 0:
            c = float(brentq(lambda x: float(lr_field(x, equilibrium_h(x, ip3, p), ip3, p)[0]),
                             float(lo), float(hi), xtol=1e-13))
        else:
            continue
        h = equilibrium_h(c, ip3, p)
        j = jacobian(c, h, ip3, p)
        trace, determinant = float(np.trace(j)), float(np.linalg.det(j))
        discriminant = trace*trace-4*determinant
        roots.append({"C": c, "h": h, "I": ip3, "trace": trace,
                      "determinant": determinant, "discriminant": discriminant,
                      "eigenvalues": [[float(v.real), float(v.imag)] for v in np.linalg.eigvals(j)]})
    return roots


def prepare_run(run_dir: Path, protocol: dict) -> None:
    manifest_path = run_dir / "run.json"
    manifest = read_json(manifest_path)
    p = manifest["parameters"]
    target = run_dir / "reference"
    if (target / "manifest.json").exists():
        prior = read_json(target / "manifest.json")
        if prior.get("run_id") != manifest["run_id"]:
            raise RuntimeError(f"Reference data belongs to a different run: {target}")
        print(f"exists: {manifest['run_id']}")
        return
    target.mkdir(parents=True, exist_ok=True)

    n_c, n_i = 320, 121
    c_lo, c_hi = protocol["data"]["nullcline_C_domain"]
    i_lo, i_hi = protocol["data"]["nullcline_I_domain"]
    c_values = np.linspace(c_lo, c_hi, n_c)
    i_values = np.linspace(i_lo, i_hi, n_i)
    cc, ii = np.meshgrid(c_values, i_values, indexing="ij")
    q = p["d_2"]*(ii+p["d_1"])/(ii+p["d_3"])
    h_null = q/(q+cc)
    denom = p["C_0"]-(1+p["c_1"])*cc
    radicand = np.full_like(cc, np.nan)
    valid_denom = denom > 0
    radicand[valid_denom] = (
        p["v_ER"]*cc[valid_denom]**2
        / ((cc[valid_denom]**2+p["K_ER"]**2)*denom[valid_denom]) - p["r_L"]
    ) / (p["r_C"]*(ii[valid_denom]/(ii[valid_denom]+p["d_1"]))**3
         *(cc[valid_denom]/(cc[valid_denom]+p["d_5"]))**3)
    valid_n = np.isfinite(radicand) & (radicand >= 0)
    n_null = np.full_like(cc, np.nan)
    n_null[valid_n] = np.cbrt(radicand[valid_n])
    np.savez_compressed(target / "nullclines.npz", C=c_values, I=i_values,
                        H=h_null, N=n_null, valid_N=valid_n)

    i_spectral = np.linspace(0.0, 2.0, 161)
    equilibria = [root for ip3 in i_spectral
                  for root in roots_for_ip3(float(ip3), p, 2/(1+p["c_1"]), 1600)]
    (target / "equilibria.json").write_text(
        json.dumps({"run_id": manifest["run_id"], "equilibria": equilibria}, indent=2,
                   allow_nan=False) + "\n", encoding="utf-8")
    (target / "spectra.json").write_text(
        json.dumps({"run_id": manifest["run_id"], "spectral_equilibrium_count": len(equilibria),
                    "events": [root for root in equilibria if abs(root["trace"]) < 1e-5
                               or abs(root["determinant"]) < 1e-5
                               or abs(root["discriminant"]) < 1e-5]},
                   indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (target / "manifest.json").write_text(
        json.dumps({"run_id": manifest["run_id"], "parameters": p,
                    "nullcline_grid": [n_c, n_i], "spectral_I_grid": len(i_spectral),
                    "equilibrium_C_bracket_resolution": 1600,
                    "notes": "Dense-grid sign-change enumeration; tangent roots may require refinement."},
                   indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"prepared: {manifest['run_id']} ({len(equilibria)} equilibrium samples)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["baseline", "threshold_reduction", "parameter_sweep"])
    parser.add_argument("--case", choices=["FM", "AM"])
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if not args.all and args.stage is None:
        parser.error("choose --all or specify --stage")
    stages = read_json(PACKAGE / "config" / "protocol.json")["optimization"]["stage_order"] if args.all else [args.stage]
    cases = [args.case] if args.case else ["FM", "AM"]
    protocol = read_json(PACKAGE / "config" / "protocol.json")
    plan_path = OUTPUT / "plan" / "index.json"
    if not plan_path.exists():
        parser.error("no plan found; run scripts/plan.py first")
    run_ids = set(read_json(plan_path)["runs"])
    for stage in stages:
        for case in cases:
            prefix = f"{stage}/{case}_"
            for run_id in sorted(item for item in run_ids if item.startswith(prefix)):
                relative = Path("runs/fm_am_reduction") / run_id
                prepare_run(ROOT / relative, protocol)


if __name__ == "__main__":
    main()
