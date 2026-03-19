from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ResourceGroup:
    group_id: str
    inventory_per_station: int
    chargers_per_station: int
    swap_capacity_per_station: int


@dataclass(frozen=True)
class Combo:
    name: str
    reposition_solver: str
    charging_solver: str


COMBOS: tuple[Combo, ...] = (
    Combo("heuristic_plus_fcfs", "heuristic", "fcfs"),
    Combo("ideal_plus_fcfs", "ideal", "fcfs"),
    Combo("heuristic_plus_gurobi", "heuristic", "gurobi"),
    Combo("ideal_plus_gurobi", "ideal", "gurobi"),
    Combo("algorithm_plus_fcfs", "gurobi", "fcfs"),
    Combo("algorithm_plus_gurobi", "gurobi", "gurobi"),
)


def _parse_group_ids(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def _load_groups(path: Path) -> list[ResourceGroup]:
    if not path.exists():
        raise FileNotFoundError(f"Group CSV not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "group_id",
            "inventory_per_station",
            "chargers_per_station",
            "swap_capacity_per_station",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
        rows = list(reader)

    groups: list[ResourceGroup] = []
    seen: set[str] = set()
    for row in rows:
        gid = str(row["group_id"]).strip()
        if not gid:
            raise ValueError("group_id cannot be empty")
        if gid in seen:
            raise ValueError(f"Duplicate group_id: {gid}")
        seen.add(gid)
        inv = int(row["inventory_per_station"])
        chg = int(row["chargers_per_station"])
        swp = int(row["swap_capacity_per_station"])
        if inv <= 0 or chg <= 0 or swp <= 0:
            raise ValueError(f"Non-positive values in group {gid}: inv={inv}, chargers={chg}, swap={swp}")
        groups.append(
            ResourceGroup(
                group_id=gid,
                inventory_per_station=inv,
                chargers_per_station=chg,
                swap_capacity_per_station=swp,
            )
        )
    if not groups:
        raise ValueError(f"No rows found in group CSV: {path}")
    return groups


def _select_groups(groups: list[ResourceGroup], selected_group_ids: Iterable[str]) -> list[ResourceGroup]:
    selected = [g for g in groups if g.group_id in set(selected_group_ids)]
    missing = set(selected_group_ids) - {g.group_id for g in groups}
    if missing:
        raise ValueError(f"Requested group_id(s) not found: {sorted(missing)}")
    if not selected:
        raise ValueError("No groups selected")
    return selected


def _run_case(
    *,
    python_bin: str,
    config_path: Path,
    out_file: Path,
    group: ResourceGroup,
    combo: Combo,
    resume: bool,
) -> None:
    cmd = [
        python_bin,
        str(ROOT / "scripts" / "run_param_sweep.py"),
        "--config",
        str(config_path),
        "--inventories",
        str(group.inventory_per_station),
        "--chargers",
        str(group.chargers_per_station),
        "--swap-capacities",
        str(group.swap_capacity_per_station),
        "--reposition-solver",
        combo.reposition_solver,
        "--charging-solver",
        combo.charging_solver,
        "--output-csv",
        str(out_file),
        "--resume" if resume else "--no-resume",
    ]
    env = os.environ.copy()
    python_env_root = Path(python_bin).resolve().parent.parent
    proj_dir = python_env_root / "share" / "proj"
    if proj_dir.exists():
        proj_dir_str = str(proj_dir)
        env["PROJ_DATA"] = proj_dir_str
        env["PROJ_LIB"] = proj_dir_str
    else:
        env["PROJ_DATA"] = "/opt/anaconda3/envs/bsm/share/proj"
        env["PROJ_LIB"] = "/opt/anaconda3/envs/bsm/share/proj"
    subprocess.run(cmd, cwd=str(ROOT), check=True, env=env)


def _merge_rows(
    *,
    expected_cases: list[tuple[ResourceGroup, Combo, Path]],
    output_csv: Path,
    require_all: bool,
) -> tuple[int, int]:
    rows: list[dict[str, str]] = []
    missing_files = 0
    for group, combo, path in expected_cases:
        if not path.exists() or path.stat().st_size == 0:
            missing_files += 1
            continue
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["group_id"] = group.group_id
                row["combo"] = combo.name
                row["resource_inventory_per_station"] = str(group.inventory_per_station)
                row["resource_chargers_per_station"] = str(group.chargers_per_station)
                row["resource_swap_capacity_per_station"] = str(group.swap_capacity_per_station)
                rows.append(row)

    if require_all and missing_files > 0:
        raise RuntimeError(f"Missing output files for {missing_files} case(s)")
    if not rows:
        raise RuntimeError("No rows available to merge")

    fieldnames = sorted({k for row in rows for k in row.keys()})
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows), missing_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups-csv", default="configs/eval_critical3_groups.csv")
    parser.add_argument("--config", default="configs/ev_yellow_2025_11_eval_6_22_strict30.yaml")
    parser.add_argument("--python-bin", default="/opt/anaconda3/envs/bsm/bin/python")
    parser.add_argument(
        "--tmp-dir",
        default="results/baseline_combinations/_tmp_eval_critical3_6combo",
    )
    parser.add_argument(
        "--output-csv",
        default="results/baseline_combinations/full_matrix_6combo_critical3_6_22.csv",
    )
    parser.add_argument(
        "--group-ids",
        default="",
        help="comma-separated subset (e.g. G2) for smoke/preflight",
    )
    parser.add_argument("--merge-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-all", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    groups = _load_groups(ROOT / str(args.groups_csv))
    if args.group_ids.strip():
        groups = _select_groups(groups, _parse_group_ids(args.group_ids))

    tmp_dir = ROOT / str(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    output_csv = ROOT / str(args.output_csv)

    expected_cases: list[tuple[ResourceGroup, Combo, Path]] = []
    for group in groups:
        for combo in COMBOS:
            out_file = tmp_dir / f"{group.group_id}__{combo.name}.csv"
            expected_cases.append((group, combo, out_file))

    failures: list[str] = []
    if not args.merge_only:
        for group, combo, out_file in expected_cases:
            print(
                f"Running {group.group_id}/{combo.name} "
                f"(inv={group.inventory_per_station}, chargers={group.chargers_per_station}, "
                f"swap={group.swap_capacity_per_station})"
            )
            try:
                _run_case(
                    python_bin=args.python_bin,
                    config_path=ROOT / str(args.config),
                    out_file=out_file,
                    group=group,
                    combo=combo,
                    resume=bool(args.resume),
                )
            except subprocess.CalledProcessError as exc:
                msg = f"{group.group_id}/{combo.name}: exit={exc.returncode}"
                failures.append(msg)
                print(f"[ERROR] {msg}")
                if not args.continue_on_error:
                    break

    rows, missing_files = _merge_rows(
        expected_cases=expected_cases,
        output_csv=output_csv,
        require_all=bool(args.require_all),
    )
    print(f"Saved merged matrix: {output_csv} (rows={rows}, missing_files={missing_files})")
    if failures:
        print("Failures:")
        for item in failures:
            print(f"- {item}")
        if not args.continue_on_error:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
