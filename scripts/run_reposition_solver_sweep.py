from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SweepParam:
    solver_method: int
    solver_crossover: int
    time_limit_sec: int
    numeric_focus: int


def _parse_int_list(value: str) -> list[int]:
    values = [v.strip() for v in str(value).split(",") if v.strip()]
    if not values:
        raise ValueError(f"empty integer list: {value!r}")
    return [int(v) for v in values]


def _dedupe_keep_order(items: Iterable[SweepParam]) -> list[SweepParam]:
    seen: set[SweepParam] = set()
    out: list[SweepParam] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _build_sweep_plan(
    *,
    methods: list[int],
    crossovers: list[int],
    time_limits: list[int],
    numeric_focuses: list[int],
    max_combos: int,
) -> list[SweepParam]:
    preferred_method = 2 if 2 in methods else methods[0]
    alt_crossover = 1 if 1 in crossovers else crossovers[0]
    alt_numeric = 1 if 1 in numeric_focuses else numeric_focuses[0]

    candidates: list[SweepParam] = []

    # Baseline rail: method x time with stable crossover=0, numeric=0.
    base_crossover = 0 if 0 in crossovers else crossovers[0]
    base_numeric = 0 if 0 in numeric_focuses else numeric_focuses[0]
    for method in methods:
        for tl in time_limits:
            candidates.append(
                SweepParam(
                    solver_method=int(method),
                    solver_crossover=int(base_crossover),
                    time_limit_sec=int(tl),
                    numeric_focus=int(base_numeric),
                )
            )

    # Sensitivity rail A: crossover impact on preferred method.
    if alt_crossover != base_crossover:
        for tl in time_limits:
            candidates.append(
                SweepParam(
                    solver_method=int(preferred_method),
                    solver_crossover=int(alt_crossover),
                    time_limit_sec=int(tl),
                    numeric_focus=int(base_numeric),
                )
            )

    # Sensitivity rail B: numeric focus impact on preferred method.
    if alt_numeric != base_numeric:
        for tl in time_limits:
            candidates.append(
                SweepParam(
                    solver_method=int(preferred_method),
                    solver_crossover=int(base_crossover),
                    time_limit_sec=int(tl),
                    numeric_focus=int(alt_numeric),
                )
            )

    deduped = _dedupe_keep_order(candidates)
    if len(deduped) > max_combos:
        return deduped[:max_combos]
    return deduped


def _load_first_row(csv_path: Path) -> dict[str, str]:
    if not csv_path.exists():
        return {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            return {k: str(v) for k, v in row.items()}
    return {}


def _safe_int(value: str | None) -> int | None:
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() == "none":
        return None
    try:
        return int(float(s))
    except Exception:
        return None


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() == "none":
        return None
    try:
        return float(s)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ev_yellow_2025_11_eval_6_22_fcfs.yaml")
    parser.add_argument("--python-bin", default="/opt/anaconda3/envs/bsm/bin/python")
    parser.add_argument("--output-csv", default="results/diagnostics/reposition_solver_param_sweep.csv")
    parser.add_argument(
        "--output-summary-json",
        default="results/diagnostics/reposition_solver_param_sweep.summary.json",
    )
    parser.add_argument(
        "--output-run-dir",
        default="results/diagnostics/reposition_solver_param_sweep_runs",
    )
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--max-run-seconds-per-run", type=float, default=240.0)
    parser.add_argument("--inventory", type=int, default=5)
    parser.add_argument("--chargers", type=int, default=5)
    parser.add_argument("--swap-capacity", type=int, default=6)
    parser.add_argument("--methods", default="2,1")
    parser.add_argument("--crossovers", default="0,1")
    parser.add_argument("--time-limits", default="10,15,20")
    parser.add_argument("--numeric-focuses", default="0,1")
    parser.add_argument("--max-combos", type=int, default=12)
    args = parser.parse_args()

    if args.max_combos <= 0:
        raise ValueError("--max-combos must be > 0")

    methods = _parse_int_list(args.methods)
    crossovers = _parse_int_list(args.crossovers)
    time_limits = _parse_int_list(args.time_limits)
    numeric_focuses = _parse_int_list(args.numeric_focuses)
    plan = _build_sweep_plan(
        methods=methods,
        crossovers=crossovers,
        time_limits=time_limits,
        numeric_focuses=numeric_focuses,
        max_combos=int(args.max_combos),
    )
    if not plan:
        raise RuntimeError("empty sweep plan")

    run_dir = ROOT / str(args.output_run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, int | float | str | None]] = []

    for idx, param in enumerate(plan, start=1):
        param_id = f"p{idx:02d}"
        run_csv = run_dir / f"{param_id}.csv"
        run_json = run_dir / f"{param_id}.summary.json"

        cmd = [
            str(args.python_bin),
            str(ROOT / "scripts" / "run_reposition_diagnostic.py"),
            "--config",
            str(args.config),
            "--output-csv",
            str(run_csv),
            "--output-summary-json",
            str(run_json),
            "--max-steps",
            str(int(args.max_steps)),
            "--max-run-seconds",
            str(float(args.max_run_seconds_per_run)),
            "--reposition-solver",
            "gurobi",
            "--charging-solver",
            "fcfs",
            "--inventory",
            str(int(args.inventory)),
            "--chargers",
            str(int(args.chargers)),
            "--swap-capacity",
            str(int(args.swap_capacity)),
            "--solver-time-limit-sec",
            str(int(param.time_limit_sec)),
            "--reposition-solver-method",
            str(int(param.solver_method)),
            "--reposition-solver-crossover",
            str(int(param.solver_crossover)),
            "--reposition-numeric-focus",
            str(int(param.numeric_focus)),
        ]
        completed = subprocess.run(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        diag_row = _load_first_row(run_csv)
        summary_payload: dict[str, object] = {}
        if run_json.exists():
            with run_json.open("r", encoding="utf-8") as f:
                summary_payload = json.load(f)

        sol_count = _safe_int(diag_row.get("reposition_sol_count"))
        status = _safe_int(diag_row.get("reposition_status"))
        outcome = diag_row.get("reposition_outcome")
        row = {
            "param_id": param_id,
            "solver_method": int(param.solver_method),
            "solver_crossover": int(param.solver_crossover),
            "time_limit_sec": int(param.time_limit_sec),
            "numeric_focus": int(param.numeric_focus),
            "process_return_code": int(completed.returncode),
            "sol_count": sol_count,
            "status": status,
            "outcome": outcome,
            "error": diag_row.get("error"),
            "run_wall_time_sec": _safe_float(str(summary_payload.get("run_wall_time_sec"))),
            "step_build_time_sec": _safe_float(diag_row.get("reposition_build_time_sec")),
            "step_optimize_time_sec": _safe_float(diag_row.get("reposition_optimize_time_sec")),
            "num_vars": _safe_int(diag_row.get("reposition_num_vars")),
            "num_constrs": _safe_int(diag_row.get("reposition_num_constrs")),
            "num_nz": _safe_int(diag_row.get("reposition_num_nz")),
            "row_csv": str(run_csv),
            "summary_json": str(run_json),
            "stdout_tail": completed.stdout.strip().splitlines()[-1] if completed.stdout.strip() else "",
            "stderr_tail": completed.stderr.strip().splitlines()[-1] if completed.stderr.strip() else "",
            "recommended": 0,
        }
        rows.append(row)
        print(
            f"[{param_id}] method={param.solver_method} crossover={param.solver_crossover} "
            f"tl={param.time_limit_sec} nf={param.numeric_focus} "
            f"sol={sol_count} status={status} wall={row['run_wall_time_sec']}"
        )

    feasible = [
        r
        for r in rows
        if (r.get("sol_count") is not None and int(r["sol_count"]) > 0)
        and (r.get("run_wall_time_sec") is not None)
    ]
    best: dict[str, int | float | str | None] | None = None
    if feasible:
        best = min(feasible, key=lambda r: float(r["run_wall_time_sec"]))  # type: ignore[arg-type]
        best["recommended"] = 1

    out_csv = ROOT / str(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "param_id",
        "solver_method",
        "solver_crossover",
        "time_limit_sec",
        "numeric_focus",
        "process_return_code",
        "sol_count",
        "status",
        "outcome",
        "error",
        "run_wall_time_sec",
        "step_build_time_sec",
        "step_optimize_time_sec",
        "num_vars",
        "num_constrs",
        "num_nz",
        "row_csv",
        "summary_json",
        "stdout_tail",
        "stderr_tail",
        "recommended",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    failure_reasons = Counter()
    if not feasible:
        for row in rows:
            reason = str(row.get("error") or row.get("outcome") or row.get("status") or "unknown")
            failure_reasons[reason] += 1

    summary = {
        "config": str(args.config),
        "python_bin": str(args.python_bin),
        "fixed_case": {
            "inventory": int(args.inventory),
            "chargers": int(args.chargers),
            "swap_capacity": int(args.swap_capacity),
            "reposition_solver": "gurobi",
            "charging_solver": "fcfs",
            "strict_mode": True,
        },
        "max_steps": int(args.max_steps),
        "max_run_seconds_per_run": float(args.max_run_seconds_per_run),
        "plan_size": len(plan),
        "selected_combo_count": len(rows),
        "at_least_one_incumbent": bool(feasible),
        "best_param_id": None if best is None else best.get("param_id"),
        "best_candidate": None
        if best is None
        else {
            "param_id": best.get("param_id"),
            "solver_method": best.get("solver_method"),
            "solver_crossover": best.get("solver_crossover"),
            "time_limit_sec": best.get("time_limit_sec"),
            "numeric_focus": best.get("numeric_focus"),
            "sol_count": best.get("sol_count"),
            "status": best.get("status"),
            "run_wall_time_sec": best.get("run_wall_time_sec"),
            "step_build_time_sec": best.get("step_build_time_sec"),
            "step_optimize_time_sec": best.get("step_optimize_time_sec"),
        },
        "failure_reason_counts": dict(failure_reasons),
        "output_csv": str(out_csv),
    }

    out_summary = ROOT / str(args.output_summary_json)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    with out_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Sweep CSV: {out_csv}")
    print(f"Sweep summary JSON: {out_summary}")
    print("Summary:", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
