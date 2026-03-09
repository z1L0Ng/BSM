from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _to_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in columns:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        default="results/baseline_combinations/stress_full_matrix.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="results/baseline_combinations/figures",
    )
    parser.add_argument(
        "--report-md",
        default="results/baseline_combinations/figures/README.md",
    )
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    report_md = Path(args.report_md)

    df = pd.read_csv(input_csv)
    numeric_cols = [
        "inventory_per_station",
        "chargers_per_station",
        "swap_capacity_per_station",
        "total_served",
        "battery_swap_success_ratio",
        "deadline_miss_ratio",
        "charging_deadline_miss_ratio",
        "total_waiting_vehicles",
        "peak_station_power_kw",
        "total_idle_moves",
    ]
    df = _to_numeric(df, numeric_cols)

    sns.set_theme(style="whitegrid")
    palette = {
        "algorithm_plus_fcfs": "#0072B2",
        "heuristic_plus_fcfs": "#009E73",
        "ideal_plus_fcfs": "#D55E00",
    }
    combo_order = ["algorithm_plus_fcfs", "heuristic_plus_fcfs", "ideal_plus_fcfs"]

    # 1) average KPI panel by combo
    avg = (
        df.groupby("combo", as_index=False)[
            [
                "total_served",
                "battery_swap_success_ratio",
                "deadline_miss_ratio",
                "charging_deadline_miss_ratio",
                "total_waiting_vehicles",
                "peak_station_power_kw",
            ]
        ]
        .mean()
        .set_index("combo")
        .reindex(combo_order)
        .reset_index()
    )
    kpi_map = [
        ("total_served", "Avg Served"),
        ("battery_swap_success_ratio", "Avg Swap Success"),
        ("deadline_miss_ratio", "Avg Swap Miss"),
        ("charging_deadline_miss_ratio", "Avg Charging Miss"),
        ("total_waiting_vehicles", "Avg Waiting Vehicles"),
        ("peak_station_power_kw", "Avg Peak Station Power (kW)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, (col, title) in zip(axes.flat, kpi_map):
        sns.barplot(
            data=avg,
            x="combo",
            y=col,
            hue="combo",
            order=combo_order,
            palette=palette,
            legend=False,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)
    _save(fig, output_dir / "01_combo_avg_kpis.png")

    # 2) served vs swap capacity
    g = (
        df.groupby(["combo", "swap_capacity_per_station"], as_index=False)["total_served"]
        .mean()
        .sort_values(["combo", "swap_capacity_per_station"])
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    sns.lineplot(
        data=g,
        x="swap_capacity_per_station",
        y="total_served",
        hue="combo",
        hue_order=combo_order,
        marker="o",
        palette=palette,
        ax=ax,
    )
    ax.set_title("Served vs Swap Capacity")
    ax.set_xlabel("Swap Capacity per Station")
    ax.set_ylabel("Avg Served")
    _save(fig, output_dir / "02_served_vs_swap_capacity.png")

    # 3) swap miss vs swap capacity
    g = (
        df.groupby(["combo", "swap_capacity_per_station"], as_index=False)["deadline_miss_ratio"]
        .mean()
        .sort_values(["combo", "swap_capacity_per_station"])
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    sns.lineplot(
        data=g,
        x="swap_capacity_per_station",
        y="deadline_miss_ratio",
        hue="combo",
        hue_order=combo_order,
        marker="o",
        palette=palette,
        ax=ax,
    )
    ax.set_title("Swap Miss Ratio vs Swap Capacity")
    ax.set_xlabel("Swap Capacity per Station")
    ax.set_ylabel("Avg Swap Miss Ratio")
    _save(fig, output_dir / "03_swap_miss_vs_swap_capacity.png")

    # 4) charging miss vs chargers
    g = (
        df.groupby(["combo", "chargers_per_station"], as_index=False)["charging_deadline_miss_ratio"]
        .mean()
        .sort_values(["combo", "chargers_per_station"])
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    sns.lineplot(
        data=g,
        x="chargers_per_station",
        y="charging_deadline_miss_ratio",
        hue="combo",
        hue_order=combo_order,
        marker="o",
        palette=palette,
        ax=ax,
    )
    ax.set_title("Charging Miss Ratio vs Chargers")
    ax.set_xlabel("Chargers per Station")
    ax.set_ylabel("Avg Charging Miss Ratio")
    _save(fig, output_dir / "04_charging_miss_vs_chargers.png")

    # 5) waiting vs swap capacity
    g = (
        df.groupby(["combo", "swap_capacity_per_station"], as_index=False)["total_waiting_vehicles"]
        .mean()
        .sort_values(["combo", "swap_capacity_per_station"])
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    sns.lineplot(
        data=g,
        x="swap_capacity_per_station",
        y="total_waiting_vehicles",
        hue="combo",
        hue_order=combo_order,
        marker="o",
        palette=palette,
        ax=ax,
    )
    ax.set_title("Waiting Vehicles vs Swap Capacity")
    ax.set_xlabel("Swap Capacity per Station")
    ax.set_ylabel("Avg Waiting Vehicles")
    _save(fig, output_dir / "05_waiting_vs_swap_capacity.png")

    # 6) peak station power vs chargers
    g = (
        df.groupby(["combo", "chargers_per_station"], as_index=False)["peak_station_power_kw"]
        .mean()
        .sort_values(["combo", "chargers_per_station"])
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    sns.lineplot(
        data=g,
        x="chargers_per_station",
        y="peak_station_power_kw",
        hue="combo",
        hue_order=combo_order,
        marker="o",
        palette=palette,
        ax=ax,
    )
    ax.set_title("Peak Station Power vs Chargers")
    ax.set_xlabel("Chargers per Station")
    ax.set_ylabel("Avg Peak Station Power (kW)")
    _save(fig, output_dir / "06_peak_power_vs_chargers.png")

    # Report markdown
    hard = df[
        (df["inventory_per_station"] == 5)
        & (df["chargers_per_station"] == 3)
        & (df["swap_capacity_per_station"] == 3)
    ][
        [
            "combo",
            "total_served",
            "battery_swap_success_ratio",
            "deadline_miss_ratio",
            "charging_deadline_miss_ratio",
            "total_waiting_vehicles",
            "peak_station_power_kw",
        ]
    ].sort_values("combo")

    lines: list[str] = []
    lines.append("# Stress Figures")
    lines.append("")
    lines.append(f"Source CSV: `{input_csv}`")
    lines.append("")
    lines.append("## Generated Figures")
    lines.append("")
    lines.append("- `01_combo_avg_kpis.png`")
    lines.append("- `02_served_vs_swap_capacity.png`")
    lines.append("- `03_swap_miss_vs_swap_capacity.png`")
    lines.append("- `04_charging_miss_vs_chargers.png`")
    lines.append("- `05_waiting_vs_swap_capacity.png`")
    lines.append("- `06_peak_power_vs_chargers.png`")
    lines.append("")
    lines.append("## Hard Setting Slice (inventory=5, chargers=3, swap_capacity=3)")
    lines.append("")
    lines.append(
        "| Combo | Served | Swap Success | Swap Miss | Charging Miss | Waiting Vehicles | Peak Station Power |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for _, row in hard.iterrows():
        lines.append(
            f"| {row['combo']} | {row['total_served']:.0f} | "
            f"{row['battery_swap_success_ratio']:.6f} | {row['deadline_miss_ratio']:.6f} | "
            f"{row['charging_deadline_miss_ratio']:.6f} | {row['total_waiting_vehicles']:.0f} | "
            f"{row['peak_station_power_kw']:.1f} |"
        )
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved figures to: {output_dir}")
    print(f"Saved report to: {report_md}")


if __name__ == "__main__":
    main()
