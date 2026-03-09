from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ev_yellow_2025_11.yaml")
    parser.add_argument("--inventory", type=int, default=50)
    parser.add_argument("--chargers", type=int, default=5)
    parser.add_argument("--swap-capacity", type=int, default=6)
    parser.add_argument("--output-csv", default="results/baseline_combinations/summary.csv")
    args = parser.parse_args()

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_csv.parent / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    combos = [
        ("algorithm_plus_fcfs", "gurobi", "fcfs"),
        ("heuristic_plus_fcfs", "heuristic", "fcfs"),
        ("ideal_plus_fcfs", "ideal", "fcfs"),
    ]

    rows: list[dict] = []
    for name, reposition_solver, charging_solver in combos:
        tmp_csv = tmp_dir / f"{name}.csv"
        cmd = [
            sys.executable,
            "scripts/run_param_sweep.py",
            "--config",
            args.config,
            "--inventories",
            str(args.inventory),
            "--chargers",
            str(args.chargers),
            "--swap-capacities",
            str(args.swap_capacity),
            "--reposition-solver",
            reposition_solver,
            "--charging-solver",
            charging_solver,
            "--output-csv",
            str(tmp_csv),
        ]
        print(f"Running {name}: reposition={reposition_solver}, charging={charging_solver}")
        subprocess.run(cmd, check=True)

        with tmp_csv.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["combo"] = name
                rows.append(row)

    if rows:
        fieldnames = sorted({k for row in rows for k in row.keys()})
        with output_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"Saved baseline combinations to: {output_csv}")


if __name__ == "__main__":
    main()

