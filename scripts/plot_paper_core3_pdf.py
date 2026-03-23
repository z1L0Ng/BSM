from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Internal combo ids are kept for reproducibility and script I/O.
PRIMARY_COMBO_ORDER = [
    "heuristic_plus_fcfs",
    "ideal_plus_fcfs",
    "algorithm_plus_gurobi",
]

ABLATION_COMBO_ORDER = [
    "heuristic_plus_fcfs",
    "ideal_plus_fcfs",
    "heuristic_plus_gurobi",
    "ideal_plus_gurobi",
    "algorithm_plus_fcfs",
    "algorithm_plus_gurobi",
]

METHOD_LABEL = {
    "algorithm_plus_gurobi": "COP",
    "heuristic_plus_fcfs": "HDB",
    "ideal_plus_fcfs": "ORB",
}

ABLATION_LABEL = {
    "heuristic_plus_fcfs": "Heuristic Dispatch x Rule Charging",
    "ideal_plus_fcfs": "Oracle Dispatch x Rule Charging",
    "heuristic_plus_gurobi": "Heuristic Dispatch x Optimized Charging",
    "ideal_plus_gurobi": "Oracle Dispatch x Optimized Charging",
    "algorithm_plus_fcfs": "Coordinated Dispatch x Rule Charging",
    "algorithm_plus_gurobi": "Coordinated Dispatch x Optimized Charging (COP)",
}

PALETTE = {
    "algorithm_plus_gurobi": "#004f9f",
    "heuristic_plus_fcfs": "#7f7f7f",
    "ideal_plus_fcfs": "#2f8f4e",
    "heuristic_plus_gurobi": "#8c6bb1",
    "ideal_plus_gurobi": "#d98e04",
    "algorithm_plus_fcfs": "#c44e52",
}

TS_COLORS = {
    "alg": "#004f9f",
    "heu": "#7f7f7f",
}


def _must_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{name} missing required columns: {miss}")


def _must_nonempty(df: pd.DataFrame, cols: list[str], name: str) -> None:
    for c in cols:
        if df[c].isna().any():
            raise ValueError(f"{name} contains NaN in column: {c}")


def _cast_numeric(df: pd.DataFrame, cols: list[str], name: str) -> None:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].isna().any():
            raise ValueError(f"{name} has non-numeric values in column: {c}")


def _save_pdf(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_fig1_primary_served(primary_df: pd.DataFrame, out_dir: Path) -> dict[str, float]:
    df = primary_df[primary_df["combo"].isin(PRIMARY_COMBO_ORDER)].copy()
    agg = df.groupby("combo", as_index=True)["total_served"].mean().reindex(PRIMARY_COMBO_ORDER)

    labels = [METHOD_LABEL[c] for c in PRIMARY_COMBO_ORDER]
    values = [float(agg[c]) for c in PRIMARY_COMBO_ORDER]
    colors = [PALETTE[c] for c in PRIMARY_COMBO_ORDER]

    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    x = np.arange(len(labels))
    ax.bar(x, values, color=colors, width=0.62)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Total Served")
    ax.set_title("Primary Performance")
    ax.grid(axis="y", alpha=0.25)

    _save_pdf(fig, out_dir / "fig1_primary_served.pdf")

    summary = {}
    for c in PRIMARY_COMBO_ORDER:
        summary[f"fig1_mean_total_served_{METHOD_LABEL[c]}"] = float(agg[c])
    return summary


def plot_fig2_ablation_served(ablation_df: pd.DataFrame, out_dir: Path) -> dict[str, float]:
    agg = (
        ablation_df.groupby("combo", as_index=True)["total_served"]
        .mean()
        .reindex(ABLATION_COMBO_ORDER)
    )
    labels = [ABLATION_LABEL[c] for c in ABLATION_COMBO_ORDER]
    values = [float(agg[c]) for c in ABLATION_COMBO_ORDER]
    colors = [PALETTE[c] for c in ABLATION_COMBO_ORDER]

    fig, ax = plt.subplots(figsize=(3.45, 4.20))
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.2)
    ax.set_xlabel("Total Served")
    ax.set_title("Ablation: Dispatch x Charging")
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()

    _save_pdf(fig, out_dir / "fig2_ablation_served.pdf")

    return {f"fig2_mean_total_served_{c}": float(agg[c]) for c in ABLATION_COMBO_ORDER}


def _plot_ts_pair(
    x: np.ndarray,
    y_alg: np.ndarray,
    y_heu: np.ndarray,
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    ax.plot(x, y_alg, color=TS_COLORS["alg"], linewidth=1.8, label="COP")
    ax.plot(x, y_heu, color=TS_COLORS["heu"], linewidth=1.8, label="HDB")
    ax.set_xlabel("Time Slot (15 min)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    _save_pdf(fig, out_path)


def plot_fig3_timeseries_set(ts_df: pd.DataFrame, out_dir: Path) -> dict[str, float]:
    x = ts_df["time_slot"].to_numpy(dtype=float)

    _plot_ts_pair(
        x,
        ts_df["served_alg"].to_numpy(dtype=float),
        ts_df["served_heu"].to_numpy(dtype=float),
        "Served", "Time-Series Check: Served", out_dir / "fig3a_timeseries_served.pdf"
    )
    _plot_ts_pair(
        x,
        ts_df["waiting_vehicles_alg"].to_numpy(dtype=float),
        ts_df["waiting_vehicles_heu"].to_numpy(dtype=float),
        "Waiting Vehicles", "Time-Series Check: Waiting", out_dir / "fig3b_timeseries_waiting.pdf"
    )
    _plot_ts_pair(
        x,
        ts_df["station_total_power_kw_alg"].to_numpy(dtype=float),
        ts_df["station_total_power_kw_heu"].to_numpy(dtype=float),
        "Station Power (kW)", "Time-Series Check: Station Power", out_dir / "fig3c_timeseries_station_power.pdf"
    )
    _plot_ts_pair(
        x,
        ts_df["swap_success_ratio_alg"].to_numpy(dtype=float),
        ts_df["swap_success_ratio_heu"].to_numpy(dtype=float),
        "Swap Success Ratio", "Time-Series Check: Swap Success", out_dir / "fig3d_timeseries_swap_success_ratio.pdf"
    )
    _plot_ts_pair(
        x,
        ts_df["charging_deadline_miss_ratio_alg"].to_numpy(dtype=float),
        ts_df["charging_deadline_miss_ratio_heu"].to_numpy(dtype=float),
        "Charging Miss Ratio", "Time-Series Check: Charging Miss", out_dir / "fig3e_timeseries_charging_miss_ratio.pdf"
    )

    served_delta = float((ts_df["served_alg"] - ts_df["served_heu"]).sum())
    waiting_delta = float((ts_df["waiting_vehicles_alg"] - ts_df["waiting_vehicles_heu"]).sum())
    power_delta = float((ts_df["station_total_power_kw_alg"] - ts_df["station_total_power_kw_heu"]).sum())

    return {
        "timeseries_delta_served_sum_COP_minus_HDB": served_delta,
        "timeseries_delta_waiting_sum_COP_minus_HDB": waiting_delta,
        "timeseries_delta_station_power_sum_COP_minus_HDB": power_delta,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CDC-style core-3 paper figures in PDF.")
    parser.add_argument(
        "--primary-csv",
        default="results/evaluation_runs/20260320_214439_main_only_3day/agg/main_matrix_raw.csv",
    )
    parser.add_argument(
        "--ablation-csv",
        default="results/evaluation_runs/20260321_223203_incremental_fix_maintext/agg/ablation_6combo_3day_all.csv",
    )
    parser.add_argument(
        "--timeseries-csv",
        default=(
            "results/evaluation_runs/20260321_223203_incremental_fix_maintext/agg/"
            "timeseries_compare_alg_vs_heu_G2_2025-11-05.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="results/evaluation_runs/20260321_223203_incremental_fix_maintext/figures_pdf",
    )
    parser.add_argument("--expected-primary-rows", type=int, default=27)
    parser.add_argument("--expected-ablation-rows", type=int, default=54)
    parser.add_argument("--expected-timeseries-rows", type=int, default=64)
    parser.add_argument("--expected-timeseries-served-delta", type=float, default=44947.0)
    args = parser.parse_args()

    primary_path = Path(args.primary_csv)
    ablation_path = Path(args.ablation_csv)
    timeseries_path = Path(args.timeseries_csv)
    out_dir = Path(args.output_dir)

    primary_df = pd.read_csv(primary_path)
    ablation_df = pd.read_csv(ablation_path)
    ts_df = pd.read_csv(timeseries_path)

    if len(primary_df) != args.expected_primary_rows:
        raise ValueError(f"Primary rows mismatch: {len(primary_df)} != {args.expected_primary_rows}")
    if len(ablation_df) != args.expected_ablation_rows:
        raise ValueError(f"Ablation rows mismatch: {len(ablation_df)} != {args.expected_ablation_rows}")
    if len(ts_df) != args.expected_timeseries_rows:
        raise ValueError(f"Time-series rows mismatch: {len(ts_df)} != {args.expected_timeseries_rows}")

    _must_columns(primary_df, ["combo", "sim_date", "group_id", "total_served"], "primary")
    _must_columns(ablation_df, ["combo", "sim_date", "group_id", "total_served"], "ablation")
    _must_columns(
        ts_df,
        [
            "time_slot",
            "served_alg",
            "served_heu",
            "waiting_vehicles_alg",
            "waiting_vehicles_heu",
            "station_total_power_kw_alg",
            "station_total_power_kw_heu",
            "swap_success_ratio_alg",
            "swap_success_ratio_heu",
            "charging_deadline_miss_ratio_alg",
            "charging_deadline_miss_ratio_heu",
        ],
        "timeseries",
    )

    _cast_numeric(primary_df, ["total_served"], "primary")
    _cast_numeric(ablation_df, ["total_served"], "ablation")
    _cast_numeric(
        ts_df,
        [
            "time_slot",
            "served_alg",
            "served_heu",
            "waiting_vehicles_alg",
            "waiting_vehicles_heu",
            "station_total_power_kw_alg",
            "station_total_power_kw_heu",
            "swap_success_ratio_alg",
            "swap_success_ratio_heu",
            "charging_deadline_miss_ratio_alg",
            "charging_deadline_miss_ratio_heu",
        ],
        "timeseries",
    )

    _must_nonempty(primary_df, ["combo", "sim_date", "group_id", "total_served"], "primary")
    _must_nonempty(ablation_df, ["combo", "sim_date", "group_id", "total_served"], "ablation")

    summary: dict[str, float | int | str | list[str]] = {}
    summary.update(plot_fig1_primary_served(primary_df, out_dir))
    summary.update(plot_fig2_ablation_served(ablation_df, out_dir))
    summary.update(plot_fig3_timeseries_set(ts_df, out_dir))

    delta_served = float(summary["timeseries_delta_served_sum_COP_minus_HDB"])
    if abs(delta_served - args.expected_timeseries_served_delta) > 1e-6:
        raise ValueError(
            "Time-series served delta mismatch: "
            f"{delta_served} != {args.expected_timeseries_served_delta}"
        )

    summary["primary_rows"] = int(len(primary_df))
    summary["ablation_rows"] = int(len(ablation_df))
    summary["timeseries_rows"] = int(len(ts_df))
    summary["output_dir"] = str(out_dir)
    summary["files"] = [
        "fig1_primary_served.pdf",
        "fig2_ablation_served.pdf",
        "fig3a_timeseries_served.pdf",
        "fig3b_timeseries_waiting.pdf",
        "fig3c_timeseries_station_power.pdf",
        "fig3d_timeseries_swap_success_ratio.pdf",
        "fig3e_timeseries_charging_miss_ratio.pdf",
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "core3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved PDF figures to: {out_dir}")
    print(f"Saved summary JSON: {out_dir / 'core3_summary.json'}")


if __name__ == "__main__":
    main()
