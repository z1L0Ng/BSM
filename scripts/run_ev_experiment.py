from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from bsm.io import ensure_dirs, load_config
from bsm.evsim.data import make_demand_matrix, make_od_tensor
from bsm.evsim.optimization import ChargingScheduler, FleetOptimizer
from bsm.evsim.simulation import simulate_baseline, simulate_proposed
from bsm.evsim.visualization import plot_demand_supply, plot_inventory, plot_power_profile


def _infer_station_zones(cfg, demand_df, zones):
    if cfg.data["stations"]["zones"]:
        return cfg.data["stations"]["zones"]
    top = (
        demand_df.groupby("pu_zone", as_index=False)["demand"].sum()
        .sort_values("demand", ascending=False)
        .head(3)["pu_zone"]
        .astype(int)
        .tolist()
    )
    cfg.data["stations"]["zones"] = top
    cfg.data["stations"]["monitor_zone_id"] = top[0]
    return top


def _run_single_window(cfg, demand_df, od_df, travel_stats, zones, time_bins):
    demand = make_demand_matrix(demand_df, zones, time_bins)
    od_tensor = make_od_tensor(od_df, zones, time_bins)

    optimizer = FleetOptimizer(
        demand=demand,
        od_tensor=od_tensor,
        travel_stats=travel_stats,
        zones=zones,
        config=cfg.data,
    )
    result = optimizer.solve()

    distance_matrix = optimizer._distance_matrix()
    baseline_supply, baseline_power, baseline_inventory = simulate_baseline(
        demand=demand,
        od_tensor=od_tensor,
        distance_matrix=distance_matrix,
        zones=zones,
        config=cfg.data,
    )

    scheduler = ChargingScheduler(cfg.data)
    arrivals = result.Y.sum(axis=(1, 2, 3))
    schedule = scheduler.schedule(arrivals)

    slot_hours = float(cfg.data["processing"]["time_bin_minutes"]) / 60
    battery_kwh = float(cfg.data["ev"]["battery_kwh"])
    charged_batteries = np.cumsum(schedule.power_kw * slot_hours / battery_kwh)
    proposed_inventory = np.cumsum(arrivals) - charged_batteries

    proposed_supply, proposed_power, proposed_inventory = simulate_proposed(
        demand=demand,
        optimization_served=result.served,
        charging_power=schedule.power_kw,
        inventory=proposed_inventory,
    )

    return (
        baseline_supply,
        baseline_power,
        baseline_inventory,
        proposed_supply,
        proposed_power,
        proposed_inventory,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    processed_dir = Path(cfg.paths["processed_dir"])
    demand_df = pd.read_parquet(processed_dir / "ev_demand.parquet")
    od_df = pd.read_parquet(processed_dir / "ev_od.parquet")
    travel_stats = pd.read_parquet(processed_dir / "ev_travel_stats.parquet")
    zones = pd.read_csv(processed_dir / "ev_zones.csv")["zone_id"].astype(int).tolist()

    station_zones = _infer_station_zones(cfg, demand_df, zones)

    optimize_by_day = bool(cfg.data["processing"].get("optimize_by_day"))
    if optimize_by_day:
        demand_df["date"] = demand_df["time_bin"].dt.date
        od_df["date"] = od_df["time_bin"].dt.date

        dates = sorted(demand_df["date"].unique())
        start_date = cfg.data["processing"].get("start_date")
        end_date = cfg.data["processing"].get("end_date")
        if start_date:
            dates = [d for d in dates if str(d) >= str(start_date)]
        if end_date:
            dates = [d for d in dates if str(d) <= str(end_date)]
        max_days = cfg.data["processing"].get("max_days")
        if max_days:
            dates = dates[: int(max_days)]
        baseline_supply_list = []
        baseline_power_list = []
        baseline_inventory_list = []
        proposed_supply_list = []
        proposed_power_list = []
        proposed_inventory_list = []
        time_bins_all = []

        for date in dates:
            print(f"Optimizing day: {date}")
            day_demand = demand_df[demand_df["date"] == date].drop(columns=["date"])
            day_od = od_df[od_df["date"] == date].drop(columns=["date"])
            if day_demand.empty:
                continue

            time_bins = sorted(day_demand["time_bin"].unique())
            outputs = _run_single_window(cfg, day_demand, day_od, travel_stats, zones, time_bins)

            baseline_supply_list.append(outputs[0])
            baseline_power_list.append(outputs[1])
            baseline_inventory_list.append(outputs[2])
            proposed_supply_list.append(outputs[3])
            proposed_power_list.append(outputs[4])
            proposed_inventory_list.append(outputs[5])
            time_bins_all.extend(time_bins)

        baseline_supply = np.concatenate(baseline_supply_list) if baseline_supply_list else np.array([])
        baseline_power = np.concatenate(baseline_power_list) if baseline_power_list else np.array([])
        baseline_inventory = np.concatenate(baseline_inventory_list) if baseline_inventory_list else np.array([])
        proposed_supply = np.concatenate(proposed_supply_list) if proposed_supply_list else np.array([])
        proposed_power = np.concatenate(proposed_power_list) if proposed_power_list else np.array([])
        proposed_inventory = np.concatenate(proposed_inventory_list) if proposed_inventory_list else np.array([])
        time_bins = time_bins_all
        demand_agg = demand_df.groupby("time_bin", as_index=False)["demand"].sum()
        demand_map = dict(zip(demand_agg["time_bin"], demand_agg["demand"]))
        demand_curve = np.array([demand_map.get(tb, 0.0) for tb in time_bins])
    else:
        time_bins = sorted(demand_df["time_bin"].unique())
        demand = make_demand_matrix(demand_df, zones, time_bins)
        od_tensor = make_od_tensor(od_df, zones, time_bins)

        optimizer = FleetOptimizer(
            demand=demand,
            od_tensor=od_tensor,
            travel_stats=travel_stats,
            zones=zones,
            config=cfg.data,
        )
        result = optimizer.solve()

        distance_matrix = optimizer._distance_matrix()
        baseline_supply, baseline_power, baseline_inventory = simulate_baseline(
            demand=demand,
            od_tensor=od_tensor,
            distance_matrix=distance_matrix,
            zones=zones,
            config=cfg.data,
        )

        scheduler = ChargingScheduler(cfg.data)
        arrivals = result.Y.sum(axis=(1, 2, 3))
        schedule = scheduler.schedule(arrivals)

        slot_hours = float(cfg.data["processing"]["time_bin_minutes"]) / 60
        battery_kwh = float(cfg.data["ev"]["battery_kwh"])
        charged_batteries = np.cumsum(schedule.power_kw * slot_hours / battery_kwh)
        proposed_inventory = np.cumsum(arrivals) - charged_batteries

        proposed_supply, proposed_power, proposed_inventory = simulate_proposed(
            demand=demand,
            optimization_served=result.served,
            charging_power=schedule.power_kw,
            inventory=proposed_inventory,
        )
        demand_curve = demand.sum(axis=1)

    reports_dir = Path(cfg.paths["reports_dir"])
    fig1 = plot_demand_supply(time_bins, demand_curve, baseline_supply, proposed_supply, reports_dir / "fig1_demand_supply.png")
    fig2 = plot_power_profile(time_bins, baseline_power, proposed_power, reports_dir / "fig2_power_profile.png")
    fig3 = plot_inventory(time_bins, baseline_inventory, proposed_inventory, reports_dir / "fig3_battery_inventory.png")

    metrics = {
        "baseline_served": float(baseline_supply.sum()),
        "proposed_served": float(proposed_supply.sum()),
        "baseline_unserved_rate": float(1 - baseline_supply.sum() / max(demand_curve.sum(), 1.0)),
        "proposed_unserved_rate": float(1 - proposed_supply.sum() / max(demand_curve.sum(), 1.0)),
        "baseline_peak_power_kw": float(np.max(baseline_power)),
        "proposed_peak_power_kw": float(np.max(proposed_power)),
    }

    metrics_path = reports_dir / "ev_metrics.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    print(f"Saved {fig1}")
    print(f"Saved {fig2}")
    print(f"Saved {fig3}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
