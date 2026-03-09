from __future__ import annotations

import sys

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from etaxi_sim.config import load_config
from etaxi_sim.data.loaders import (
    estimate_peak_concurrent_trips,
    load_zone_map,
    load_yellow_trip_demand,
    make_synthetic_demand,
)
from etaxi_sim.data.preprocess import (
    energy_consumption_matrix,
    energy_consumption_from_reachability,
    make_distance_matrix,
    make_reachability,
    make_transition_from_tripdata,
    make_uniform_transition,
    reachability_from_taxi_zones,
)
from etaxi_sim.models.fleet import initialize_fleet
from etaxi_sim.models.station import Station
from etaxi_sim.sim.core import Simulation
from etaxi_sim.sim.metrics import MetricsRecorder


def save_episode_logs(
    metrics: MetricsRecorder,
    summary: dict,
    output_root: Path,
    config_path: str,
) -> Path:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=False)

    default_fields = [
        "time_slot",
        "charging_demand",
        "number_of_swaps",
        "unmet_battery_demand",
        "served",
        "idle_moves",
        "swap_arrivals",
        "swap_success_ratio",
        "deadline_misses",
        "deadline_miss_ratio",
        "charging_power_kw",
        "total_charging_demand_kw",
        "peak_station_power_kw",
        "waiting_time_for_battery_slots",
        "idle_driving_distance",
        "vacant",
        "occupied",
        "full_batteries",
        "t",
    ]
    extras = []
    if metrics.steps:
        extras = sorted({k for step in metrics.steps for k in step.keys()} - set(default_fields))
    fieldnames = default_fields + extras

    with (output_dir / "timeseries.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for step in metrics.steps:
            writer.writerow(step)

    summary_payload = {
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": config_path,
        "summary": summary,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    return output_dir


def build_stations(
    m: int,
    levels: int,
    swapping_capacity: int,
    chargers: int,
    full_batteries: int,
    base_load_kw: float,
) -> list[Station]:
    stations = []
    for i in range(m):
        stations.append(
            Station(
                station_id=i,
                swapping_capacity=swapping_capacity,
                chargers=chargers,
                battery_levels=levels,
                full_batteries=full_batteries,
                partial_batteries=np.zeros(levels, dtype=int),
                base_load_kw=base_load_kw,
            )
        )
    return stations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ev_yellow_2025_11.yaml")
    parser.add_argument("--episode-log-dir", default="results/episodes")
    parser.add_argument(
        "--save-episode-log",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    zone_map = load_zone_map(cfg.data.taxi_zone_lookup_path)
    m = len(zone_map.zone_ids)

    if cfg.data.source == "yellow_tripdata":
        demand_data = load_yellow_trip_demand(
            cfg.data.yellow_tripdata_path,
            zone_map,
            cfg.sim.time_bin_minutes,
            cfg.sim.horizon,
            cfg.sim.sim_date,
        )
    else:
        demand_data = make_synthetic_demand(cfg.sim.horizon, m, cfg.sim.seed)

    if cfg.model.transition_mode == "historical_tripdata" and cfg.data.source == "yellow_tripdata":
        transition = make_transition_from_tripdata(
            yellow_tripdata_path=cfg.data.yellow_tripdata_path,
            zone_ids=zone_map.zone_ids,
            zone_index=zone_map.zone_index,
            time_bin_minutes=cfg.sim.time_bin_minutes,
            horizon=cfg.sim.horizon,
            sim_date=cfg.sim.sim_date,
            pickup_prob_floor=cfg.model.transition_pickup_floor,
            pickup_prob_ceiling=cfg.model.transition_pickup_ceiling,
            pickup_smoothing_window=cfg.model.transition_pickup_smoothing_window,
        )
    else:
        transition = make_uniform_transition(cfg.sim.horizon, m)

    if cfg.model.distance_mode == "taxi_zones":
        reachability = reachability_from_taxi_zones(cfg.data.taxi_zones_shp_path, zone_map.zone_ids)
        energy_consumption = energy_consumption_from_reachability(reachability)
    else:
        distance_matrix = make_distance_matrix(m, cfg.sim.seed)
        reachability = make_reachability(distance_matrix, cfg.model.max_reachable_distance)
        energy_consumption = energy_consumption_matrix(distance_matrix, cfg.sim.battery_levels)

    initial_vehicles = cfg.sim.initial_vehicles
    if cfg.sim.initial_vehicles_mode == "from_data_peak_overlap" and cfg.data.source == "yellow_tripdata":
        peak_overlap = estimate_peak_concurrent_trips(cfg.data.yellow_tripdata_path, cfg.sim.sim_date)
        initial_vehicles = max(1, int(np.ceil(peak_overlap * cfg.sim.initial_vehicles_scale)))
        print(
            f"Initial vehicles from data peak overlap: peak={peak_overlap}, "
            f"scale={cfg.sim.initial_vehicles_scale}, used={initial_vehicles}"
        )

    fleet = initialize_fleet(m, cfg.sim.battery_levels, cfg.sim.seed, initial_vehicles)
    stations = build_stations(
        m=m,
        levels=cfg.sim.battery_levels,
        swapping_capacity=cfg.station.swapping_capacity,
        chargers=cfg.station.chargers,
        full_batteries=cfg.sim.min_full_batteries_per_station,
        base_load_kw=cfg.sim.base_load_kw,
    )

    metrics = MetricsRecorder()
    sim = Simulation(
        fleet=fleet,
        stations=stations,
        transition=transition,
        energy_consumption=energy_consumption,
        reachability=reachability,
        levels=cfg.sim.battery_levels,
        horizon=cfg.sim.horizon,
        demand_forecast=demand_data.demand,
        charge_rate_levels_per_slot=cfg.sim.charge_rate_levels_per_slot,
        deadline_horizon=cfg.sim.swap_deadline_horizon,
        charge_power_kw=cfg.sim.charge_power_kw,
        reposition_solver=cfg.model.reposition_solver,
        reposition_planning_horizon_slots=cfg.model.reposition_planning_horizon_slots,
        charging_solver=cfg.model.charging_solver,
        reposition_idle_cost_weight=cfg.model.reposition_idle_cost_weight,
        reposition_top_demand_targets=cfg.model.reposition_top_demand_targets,
        reposition_top_swap_targets=cfg.model.reposition_top_swap_targets,
        reposition_low_energy_swap_bonus=cfg.model.reposition_low_energy_swap_bonus,
        reposition_transition_topk=cfg.model.reposition_transition_topk,
        charging_miss_penalty=cfg.model.charging_miss_penalty,
        solver_time_limit_sec=cfg.model.solver_time_limit_sec,
        metrics=metrics,
        charging_tasks=[],
        waiting_queue=np.zeros((m, cfg.sim.battery_levels + 1), dtype=int),
        active_charging_task_ids_by_station={},
        task_counter=0,
        rng=np.random.default_rng(cfg.sim.seed),
    )

    for t in range(cfg.sim.horizon):
        sim.step(t, demand_data.demand[t], cfg.sim.swap_low_energy_threshold)

    summary = metrics.to_summary()
    if metrics.steps:
        totals = [step["vacant"] + step["occupied"] + step["waiting_vehicles"] for step in metrics.steps]
        summary["vehicle_total_min"] = int(min(totals))
        summary["vehicle_total_max"] = int(max(totals))
        summary["initial_vehicles"] = int(initial_vehicles)

    if args.save_episode_log:
        output_dir = save_episode_logs(
            metrics=metrics,
            summary=summary,
            output_root=Path(args.episode_log_dir),
            config_path=str(args.config),
        )
        summary["episode_log_dir"] = str(output_dir)

    print("Simulation complete")
    print(summary)


if __name__ == "__main__":
    main()
