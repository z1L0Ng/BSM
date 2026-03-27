from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = ["COP", "FDB", "HDB", "CCB"]
METHOD_COLORS = {
    "COP": "#004f9f",
    "FDB": "#c44e52",
    "HDB": "#7f7f7f",
    "CCB": "#6aa84f",
}


def _save_pdf(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def _require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _to_numeric(df: pd.DataFrame, cols: list[str], name: str) -> None:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].isna().any():
            raise ValueError(f"{name} has non-numeric values in {c}")


def _load_method_means(primary_df: pd.DataFrame, ablation_df: pd.DataFrame, ccb_df: pd.DataFrame) -> pd.DataFrame:
    cop = primary_df[primary_df["combo"] == "algorithm_plus_gurobi"].copy()
    hdb = primary_df[primary_df["combo"] == "heuristic_plus_fcfs"].copy()
    fdb = ablation_df[ablation_df["combo"] == "algorithm_plus_fcfs"].copy()
    ccb = ccb_df.copy()

    key_cols = ["sim_date", "group_id"]
    for name, df in [("COP", cop), ("HDB", hdb), ("FDB", fdb), ("CCB", ccb)]:
        if len(df) != 9:
            raise ValueError(f"{name} rows mismatch: {len(df)} != 9")
        if df.duplicated(subset=key_cols).any():
            raise ValueError(f"{name} has duplicated (sim_date, group_id) keys")

    metrics = [
        "total_served",
        "max_charging_power_demand_kw",
        "max_station_total_power_kw",
        "total_idle_driving_distance",
        "total_number_of_swaps",
    ]
    for name, df in [("COP", cop), ("HDB", hdb), ("FDB", fdb), ("CCB", ccb)]:
        _require_columns(df, metrics, name)
        _to_numeric(df, metrics, name)

    out = pd.DataFrame(
        [
            {"method": "COP", **{m: float(cop[m].mean()) for m in metrics}},
            {"method": "FDB", **{m: float(fdb[m].mean()) for m in metrics}},
            {"method": "HDB", **{m: float(hdb[m].mean()) for m in metrics}},
            {"method": "CCB", **{m: float(ccb[m].mean()) for m in metrics}},
        ]
    )
    out["method"] = pd.Categorical(out["method"], categories=METHOD_ORDER, ordered=True)
    return out.sort_values("method").reset_index(drop=True)


def _plot_bar_single(means_df: pd.DataFrame, metric: str, ylabel: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.55, 2.60))
    x = np.arange(len(METHOD_ORDER))
    vals = [float(means_df.loc[means_df["method"] == m, metric].iloc[0]) for m in METHOD_ORDER]
    colors = [METHOD_COLORS[m] for m in METHOD_ORDER]
    ax.bar(x, vals, color=colors, width=0.64)
    ax.set_xticks(x)
    ax.set_xticklabels(METHOD_ORDER)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    _save_pdf(fig, out_path)


def _plot_bar_double(
    means_df: pd.DataFrame,
    metric_left: str,
    metric_right: str,
    ylabel_left: str,
    ylabel_right: str,
    title_left: str,
    title_right: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.10, 2.60))
    x = np.arange(len(METHOD_ORDER))
    colors = [METHOD_COLORS[m] for m in METHOD_ORDER]

    left_vals = [float(means_df.loc[means_df["method"] == m, metric_left].iloc[0]) for m in METHOD_ORDER]
    axes[0].bar(x, left_vals, color=colors, width=0.64)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(METHOD_ORDER)
    axes[0].set_ylabel(ylabel_left)
    axes[0].set_title(title_left)
    axes[0].grid(axis="y", alpha=0.25)

    right_vals = [float(means_df.loc[means_df["method"] == m, metric_right].iloc[0]) for m in METHOD_ORDER]
    axes[1].bar(x, right_vals, color=colors, width=0.64)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(METHOD_ORDER)
    axes[1].set_ylabel(ylabel_right)
    axes[1].set_title(title_right)
    axes[1].grid(axis="y", alpha=0.25)

    _save_pdf(fig, out_path)


def _build_ablation_rows(primary_df: pd.DataFrame, ablation_df: pd.DataFrame) -> pd.DataFrame:
    def _mean(df: pd.DataFrame, combo: str, metric: str) -> float:
        return float(df.loc[df["combo"] == combo, metric].mean())

    m = "max_charging_power_demand_kw"
    s = "total_served"
    rows = [
        {
            "contrast": "Variant A (COP vs FDB)",
            "delta_served": _mean(primary_df, "algorithm_plus_gurobi", s) - _mean(ablation_df, "algorithm_plus_fcfs", s),
            "delta_peak_demand_kw": _mean(primary_df, "algorithm_plus_gurobi", m)
            - _mean(ablation_df, "algorithm_plus_fcfs", m),
        },
        {
            "contrast": "Variant B (Ideal dispatch: optimized vs FCFS charging)",
            "delta_served": _mean(ablation_df, "ideal_plus_gurobi", s) - _mean(primary_df, "ideal_plus_fcfs", s),
            "delta_peak_demand_kw": _mean(ablation_df, "ideal_plus_gurobi", m)
            - _mean(primary_df, "ideal_plus_fcfs", m),
        },
        {
            "contrast": "Dispatch effect under optimized charging",
            "delta_served": _mean(primary_df, "algorithm_plus_gurobi", s)
            - _mean(ablation_df, "heuristic_plus_gurobi", s),
            "delta_peak_demand_kw": _mean(primary_df, "algorithm_plus_gurobi", m)
            - _mean(ablation_df, "heuristic_plus_gurobi", m),
        },
        {
            "contrast": "Dispatch effect under FCFS charging",
            "delta_served": _mean(ablation_df, "algorithm_plus_fcfs", s) - _mean(primary_df, "heuristic_plus_fcfs", s),
            "delta_peak_demand_kw": _mean(ablation_df, "algorithm_plus_fcfs", m)
            - _mean(primary_df, "heuristic_plus_fcfs", m),
        },
    ]
    return pd.DataFrame(rows)


def _plot_ablation(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.20, 3.20))
    y = np.arange(len(df))

    axes[0].barh(y, df["delta_served"].to_numpy(dtype=float), color="#1f77b4", height=0.60)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(df["contrast"].tolist(), fontsize=7.2)
    axes[0].set_xlabel("Delta Served")
    axes[0].set_title("Ablation: Served")
    axes[0].grid(axis="x", alpha=0.25)
    axes[0].invert_yaxis()

    axes[1].barh(y, df["delta_peak_demand_kw"].to_numpy(dtype=float), color="#ff7f0e", height=0.60)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].set_xlabel("Delta Peak Demand (kW)")
    axes[1].set_title("Ablation: Peak Charging Demand")
    axes[1].grid(axis="x", alpha=0.25)
    axes[1].invert_yaxis()

    _save_pdf(fig, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-B teacher-style evaluation figures (PDF only).")
    parser.add_argument(
        "--primary-csv",
        default="results/evaluation_runs/20260320_214439_main_only_3day/agg/main_matrix_raw.csv",
    )
    parser.add_argument(
        "--ablation-csv",
        default="results/evaluation_runs/20260321_223203_incremental_fix_maintext/agg/ablation_6combo_3day_all.csv",
    )
    parser.add_argument("--ccb-csv", required=True)
    parser.add_argument(
        "--output-dir",
        default="results/evaluation_runs/phaseb_eval_figures_pdf",
    )
    parser.add_argument("--expected-primary-rows", type=int, default=27)
    parser.add_argument("--expected-ablation-rows", type=int, default=54)
    parser.add_argument("--expected-ccb-rows", type=int, default=9)
    args = parser.parse_args()

    primary_df = pd.read_csv(Path(args.primary_csv))
    ablation_df = pd.read_csv(Path(args.ablation_csv))
    ccb_df = pd.read_csv(Path(args.ccb_csv))
    out_dir = Path(args.output_dir)

    if len(primary_df) != args.expected_primary_rows:
        raise ValueError(f"Primary rows mismatch: {len(primary_df)} != {args.expected_primary_rows}")
    if len(ablation_df) != args.expected_ablation_rows:
        raise ValueError(f"Ablation rows mismatch: {len(ablation_df)} != {args.expected_ablation_rows}")
    if len(ccb_df) != args.expected_ccb_rows:
        raise ValueError(f"CCB rows mismatch: {len(ccb_df)} != {args.expected_ccb_rows}")

    _require_columns(primary_df, ["combo", "sim_date", "group_id"], "primary")
    _require_columns(ablation_df, ["combo", "sim_date", "group_id"], "ablation")
    _require_columns(ccb_df, ["sim_date", "group_id", "reposition_solver"], "ccb")
    if ccb_df.duplicated(subset=["sim_date", "group_id"]).any():
        raise ValueError("CCB has duplicate (sim_date, group_id) keys")
    solver_values = sorted(set(ccb_df["reposition_solver"].astype(str).str.lower().tolist()))
    if solver_values != ["gurobi"]:
        raise ValueError(f"CCB gate failed: reposition_solver values = {solver_values}, expected ['gurobi']")

    means_df = _load_method_means(primary_df, ablation_df, ccb_df)
    ablation_rows = _build_ablation_rows(primary_df, ablation_df)

    _plot_bar_single(
        means_df,
        metric="total_served",
        ylabel="Served Passengers",
        title="Performance: Served Passengers",
        out_path=out_dir / "fig1_performance_served.pdf",
    )
    _plot_bar_double(
        means_df,
        metric_left="max_charging_power_demand_kw",
        metric_right="max_station_total_power_kw",
        ylabel_left="Peak Charging Demand (kW)",
        ylabel_right="Peak Station Total Power (kW)",
        title_left="Performance Metric 2a",
        title_right="Performance Metric 2b",
        out_path=out_dir / "fig2_performance_peak_power.pdf",
    )
    _plot_bar_double(
        means_df,
        metric_left="total_idle_driving_distance",
        metric_right="total_number_of_swaps",
        ylabel_left="Idle Driving Distance",
        ylabel_right="Number of Swaps",
        title_left="Overhead: Idle Distance",
        title_right="Overhead: Battery Handling",
        out_path=out_dir / "fig3_overhead.pdf",
    )
    _plot_ablation(ablation_rows, out_dir / "fig4_ablation_variants.pdf")

    summary = {
        "primary_rows": int(len(primary_df)),
        "ablation_rows": int(len(ablation_df)),
        "ccb_rows": int(len(ccb_df)),
        "ccb_reposition_solver_values": solver_values,
        "means": means_df.to_dict(orient="records"),
        "ablation_rows_table": ablation_rows.to_dict(orient="records"),
        "files": [
            "fig1_performance_served.pdf",
            "fig2_performance_peak_power.pdf",
            "fig3_overhead.pdf",
            "fig4_ablation_variants.pdf",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "phaseb_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved phase-B figures to: {out_dir}")
    print(f"Saved summary: {out_dir / 'phaseb_summary.json'}")


if __name__ == "__main__":
    main()
