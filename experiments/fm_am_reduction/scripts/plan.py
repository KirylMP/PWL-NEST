#!/usr/bin/env python3
"""Expand the FM/AM protocol into immutable, isolated run configurations."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "experiments" / "fm_am_reduction"
OUTPUT = ROOT / "runs" / "fm_am_reduction"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def slug(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def expand(stage: str, cases: dict, protocol: dict, selected_cases: list[str]) -> list[dict]:
    jobs = []
    stage_config = protocol["stages"][stage]
    for case in selected_cases:
        case_parameters = cases["cases"][case]
        if stage_config["K_policy"] == "canonical":
            k_values = [case_parameters["K_ER"]]
        elif stage_config["K_policy"] == "case_sweep":
            k_values = case_parameters["K_ER_sweep"]
        else:
            raise ValueError(f"Unknown K policy for stage {stage!r}")

        for m, k_er in itertools.product(stage_config["m_values"], k_values):
            name = f"{case}_m{int(m):02d}_K{slug(float(k_er))}"
            relative_dir = Path(stage) / name
            jobs.append({
                "run_id": f"{stage}/{name}",
                "stage": stage,
                "case": case,
                "m": int(m),
                "regions": int(m) + 1,
                "K_ER": float(k_er),
                "parameters": {**cases["shared_parameters"], "K_ER": float(k_er)},
                "seed": int(protocol["optimization"]["seed"]),
                "restarts": int(protocol["optimization"]["restarts"]),
                "protocol_id": protocol["protocol_id"],
                "output_dir": str(Path("runs/fm_am_reduction") / relative_dir),
                "reference_dir": "reference",
                "warm_start_from_previous_m": bool(stage_config.get("warm_start_from_previous_m", False)),
            })
    return jobs


def persist(jobs: list[dict], cases_path: Path, protocol_path: Path) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    index = {
        "schema_version": 1,
        "protocol_id": read_json(protocol_path)["protocol_id"],
        "source_hashes": {
            "cases.json": sha256(cases_path),
            "protocol.json": sha256(protocol_path),
            "FM_REDUCTION_PROTOCOL.md": sha256(PACKAGE / "protocol" / "FM_REDUCTION_PROTOCOL.md"),
        },
        "runs": [],
    }
    for job in jobs:
        run_dir = ROOT / job["output_dir"]
        run_dir.mkdir(parents=True, exist_ok=True)
        run_path = run_dir / "run.json"
        manifest = {
            **job,
            "config_hashes": index["source_hashes"],
            "status": "planned",
            "warm_start_policy": (
                "select the best independently validated candidate at the next larger m;"
                " keep at least one independent restart"
                if job["warm_start_from_previous_m"] else "independent configured restarts"
            ),
        }
        encoded = json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if run_path.exists() and run_path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError(f"Refusing to overwrite a different run manifest: {run_path}")
        run_path.write_text(encoded, encoding="utf-8")
        index["runs"].append(job["run_id"])

    plan_dir = OUTPUT / "plan"
    plan_dir.mkdir(parents=True, exist_ok=True)
    index_path = plan_dir / "index.json"
    encoded_index = json.dumps(index, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if index_path.exists():
        previous = read_json(index_path)
        old_runs = set(previous.get("runs", []))
        new_runs = set(index["runs"])
        previous["runs"] = sorted(old_runs | new_runs)
        previous["source_hashes"] = index["source_hashes"]
        encoded_index = json.dumps(previous, indent=2, sort_keys=True, allow_nan=False) + "\n"
    index_path.write_text(encoded_index, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["baseline", "threshold_reduction", "parameter_sweep"])
    parser.add_argument("--case", choices=["FM", "AM"])
    parser.add_argument("--all", action="store_true", help="plan every stage for both cases")
    args = parser.parse_args()
    if not args.all and args.stage is None:
        parser.error("choose --all or specify --stage")

    cases_path = PACKAGE / "config" / "cases.json"
    protocol_path = PACKAGE / "config" / "protocol.json"
    cases = read_json(cases_path)
    protocol = read_json(protocol_path)
    stages = protocol["optimization"]["stage_order"] if args.all else [args.stage]
    selected_cases = [args.case] if args.case else ["FM", "AM"]
    jobs = [job for stage in stages for job in expand(stage, cases, protocol, selected_cases)]
    persist(jobs, cases_path, protocol_path)
    print(f"Planned {len(jobs)} run(s):")
    for job in jobs:
        print(f"  {job['run_id']}  K_ER={job['K_ER']:g}  output={job['output_dir']}")


if __name__ == "__main__":
    main()
