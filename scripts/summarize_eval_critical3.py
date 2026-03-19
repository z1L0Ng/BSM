from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


COMBO_ORDER = [
    "heuristic_plus_fcfs",
    "ideal_plus_fcfs",
    "heuristic_plus_gurobi",
    "ideal_plus_gurobi",
    "algorithm_plus_fcfs",
    "algorithm_plus_gurobi",
]
KPI_COLUMNS = [
    "total_served",
    "battery_swap_success_ratio",
    "deadline_miss_ratio",
    "charging_deadline_miss_ratio",
    "total_waiting_vehicles",
    "max_station_total_power_kw",
]
AUDIT_PREFIXES = ("reposition_step_", "reposition_event_")


def _load_groups(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "group_id" not in (reader.fieldnames or []):
            raise ValueError(f"Missing group_id column in {path}")
        return [str(row["group_id"]).strip() for row in reader if str(row["group_id"]).strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        default="results/baseline_combinations/full_matrix_6combo_critical3_6_22.csv",
    )
    parser.add_argument(
        "--groups-csv",
        default="configs/eval_critical3_groups.csv",
    )
    parser.add_argument(
        "--output-md",
        default="results/baseline_combinations/eval_summary_critical3_6combo.md",
    )
    parser.add_argument(
        "--output-csv",
        default="results/baseline_combinations/eval_summary_critical3_6combo.csv",
    )
    parser.add_argument("--strict-validate", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    df = pd.read_csv(input_csv)

    required_cols = ["group_id", "combo", *KPI_COLUMNS]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {input_csv}: {missing}")

    for c in KPI_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if bool(args.strict_validate) and df[KPI_COLUMNS].isna().any().any():
        bad_cols = [c for c in KPI_COLUMNS if df[c].isna().any()]
        raise ValueError(f"Found null KPI values in columns: {bad_cols}")

    dup_count = int(df.duplicated(subset=["group_id", "combo"]).sum())
    if bool(args.strict_validate) and dup_count > 0:
        raise ValueError(f"Duplicate keys found for (group_id, combo): {dup_count}")

    expected_groups = _load_groups(Path(args.groups_csv))
    expected_rows = len(expected_groups) * len(COMBO_ORDER)
    row_count = int(len(df))
    if bool(args.strict_validate) and row_count != expected_rows:
        raise ValueError(f"Row count mismatch: got {row_count}, expected {expected_rows}")

    if bool(args.strict_validate):
        if "reposition_solver" in df.columns:
            gurobi_mask = df["reposition_solver"].astype(str).eq("gurobi")
        else:
            gurobi_mask = df["combo"].astype(str).str.startswith("algorithm_plus_")
        gurobi_rows = int(gurobi_mask.sum())
        if gurobi_rows == 0:
            raise ValueError("No gurobi reposition rows found for strict audit validation")
        audit_cols = [c for c in df.columns if c.startswith(AUDIT_PREFIXES)]
        missing_prefix = [p for p in AUDIT_PREFIXES if not any(c.startswith(p) for c in audit_cols)]
        if missing_prefix:
            raise ValueError(f"Missing gurobi audit fields with prefix(es): {missing_prefix}")
        null_audit_cols = [c for c in audit_cols if df.loc[gurobi_mask, c].isna().any()]
        if null_audit_cols:
            raise ValueError(f"Found null gurobi audit values in columns: {null_audit_cols}")

    df["combo"] = pd.Categorical(df["combo"], categories=COMBO_ORDER, ordered=True)
    group_order = [g for g in expected_groups if g in set(df["group_id"])]
    df["group_id"] = pd.Categorical(df["group_id"], categories=group_order, ordered=True)
    out_df = df[["group_id", "combo", *KPI_COLUMNS]].sort_values(["group_id", "combo"]).copy()
    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    lines: list[str] = []
    lines.append("# Evaluation Summary (Critical3 x 6 Combos)")
    lines.append("")
    lines.append(f"- Source: `{input_csv}`")
    lines.append(f"- Rows: {row_count} (expected {expected_rows})")
    lines.append(f"- Duplicate key count (group_id+combo): {dup_count}")
    lines.append("")
    lines.append(
        "| Group | Combo | Served | Swap Success | Swap Miss | Charging Miss | Waiting | Peak Station Power |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for _, row in out_df.iterrows():
        lines.append(
            f"| {row['group_id']} | {row['combo']} | {row['total_served']:.0f} | "
            f"{row['battery_swap_success_ratio']:.6f} | {row['deadline_miss_ratio']:.6f} | "
            f"{row['charging_deadline_miss_ratio']:.6f} | {row['total_waiting_vehicles']:.0f} | "
            f"{row['max_station_total_power_kw']:.1f} |"
        )

    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved summary CSV: {out_csv}")
    print(f"Saved summary MD: {out_md}")


if __name__ == "__main__":
    main()
