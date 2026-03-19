from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


COMBO_ORDER = [
    "heuristic_plus_fcfs",
    "ideal_plus_fcfs",
    "heuristic_plus_gurobi",
    "ideal_plus_gurobi",
    "algorithm_plus_fcfs",
    "algorithm_plus_gurobi",
]
PALETTE = {
    "heuristic_plus_fcfs": "#1b9e77",
    "ideal_plus_fcfs": "#d95f02",
    "heuristic_plus_gurobi": "#7570b3",
    "ideal_plus_gurobi": "#e7298a",
    "algorithm_plus_fcfs": "#66a61e",
    "algorithm_plus_gurobi": "#e6ab02",
}
METRICS = [
    ("total_served", "Served", "01_served_by_group_combo.png"),
    ("battery_swap_success_ratio", "Swap Success Ratio", "02_swap_success_by_group_combo.png"),
    ("deadline_miss_ratio", "Swap Miss Ratio", "03_swap_miss_by_group_combo.png"),
    ("charging_deadline_miss_ratio", "Charging Miss Ratio", "04_charging_miss_by_group_combo.png"),
    ("total_waiting_vehicles", "Waiting Vehicles", "05_waiting_by_group_combo.png"),
    ("max_station_total_power_kw", "Peak Station Power (kW)", "06_peak_power_by_group_combo.png"),
]


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-csv",
        default="results/baseline_combinations/full_matrix_6combo_critical3_6_22.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="results/baseline_combinations/figures_eval_critical3_6combo",
    )
    parser.add_argument(
        "--report-md",
        default="results/baseline_combinations/figures_eval_critical3_6combo/README.md",
    )
    args = parser.parse_args()

    df = pd.read_csv(Path(args.input_csv))
    if "group_id" not in df.columns or "combo" not in df.columns:
        raise ValueError("Input CSV must contain group_id and combo columns")

    for metric, _, _ in METRICS:
        if metric not in df.columns:
            raise ValueError(f"Missing metric column in input CSV: {metric}")
        df[metric] = pd.to_numeric(df[metric], errors="coerce")

    group_order = sorted(df["group_id"].dropna().astype(str).unique().tolist())
    df["combo"] = pd.Categorical(df["combo"], categories=COMBO_ORDER, ordered=True)

    sns.set_theme(style="whitegrid")
    output_dir = Path(args.output_dir)

    generated: list[str] = []
    for metric, ylabel, filename in METRICS:
        fig, ax = plt.subplots(figsize=(9.0, 5.2))
        sns.barplot(
            data=df,
            x="group_id",
            y=metric,
            hue="combo",
            order=group_order,
            hue_order=COMBO_ORDER,
            palette=PALETTE,
            ax=ax,
        )
        ax.set_title(f"{ylabel} by Group and Combo")
        ax.set_xlabel("Resource Group")
        ax.set_ylabel(ylabel)
        ax.legend(title="Combo", fontsize=8, title_fontsize=9, loc="best")
        path = output_dir / filename
        _save(fig, path)
        generated.append(filename)

    report = Path(args.report_md)
    lines: list[str] = []
    lines.append("# Evaluation Figures (Critical3 x 6 Combos)")
    lines.append("")
    lines.append(f"- Source CSV: `{args.input_csv}`")
    lines.append("")
    lines.append("## Generated Figures")
    lines.append("")
    for name in generated:
        lines.append(f"- `{name}`")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved figures to: {output_dir}")
    print(f"Saved report to: {report}")


if __name__ == "__main__":
    main()
