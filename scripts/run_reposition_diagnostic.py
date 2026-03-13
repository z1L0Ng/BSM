from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from etaxi_sim.config import load_config
from etaxi_sim.data.loaders import (
    estimate_peak_concurrent_trips,
    load_yellow_trip_demand,
    load_zone_map,
    make_synthetic_demand,
)
from etaxi_sim.data.preprocess import (
    energy_consumption_from_reachability,
    energy_consumption_matrix,
    make_distance_matrix,
    make_reachability,
    make_transition_from_tripdata,
    make_uniform_transition,
    reachability_from_taxi_zones,
)
from etaxi_sim.models.fleet import initialize_fleet
from etaxi_sim.models.station import Station
from etaxi_sim.policies.reposition import get_last_gurobi_reposition_trace
from etaxi_sim.sim.core import Simulation
from etaxi_sim.sim.metrics import MetricsRecorder


def build_stations(
    m: int,
    levels: int,
    swapping_capacity: int,
    chargers: int,
    full_batteries: int,
    base_load_kw: float,
) -> list[Station]:
    stations: list[Station] = []
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
    parser.add_argument("--output-csv", default="results/diagnostics/reposition_trace.csv")
    parser.add_argument("--max-steps", type=int, default=0, help="0 means full horizon")
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="continue loop after step exceptions (best effort)",
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

    limit = cfg.sim.horizon if args.max_steps <= 0 else min(cfg.sim.horizon, args.max_steps)
    rows: list[dict[str, int | float | str | None]] = []

    for t in range(limit):
        err: str | None = None
        try:
            sim.step(t, demand_data.demand[t], cfg.sim.swap_low_energy_threshold)
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"

        trace = get_last_gurobi_reposition_trace()
        step = metrics.steps[-1] if metrics.steps and int(metrics.steps[-1]["t"]) == t else None
        row = {
            "t": int(t),
            "trace_t_start": trace.get("t_start"),
            "reposition_status": trace.get("status"),
            "reposition_sol_count": trace.get("sol_count"),
            "reposition_outcome": trace.get("outcome"),
            "reposition_note": trace.get("note"),
            "error": err,
            "swap_requests": None if step is None else int(step["swap_requests"]),
            "successful_swaps": None if step is None else int(step["number_of_swaps"]),
            "charging_deadline_misses": None if step is None else int(step["charging_deadline_misses"]),
            "waiting_vehicles": None if step is None else int(step["waiting_vehicles"]),
        }
        rows.append(row)

        if err is not None and not args.continue_on_error:
            break

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "t",
        "trace_t_start",
        "reposition_status",
        "reposition_sol_count",
        "reposition_outcome",
        "reposition_note",
        "error",
        "swap_requests",
        "successful_swaps",
        "charging_deadline_misses",
        "waiting_vehicles",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    outcome_counter = Counter(str(row["reposition_outcome"]) for row in rows)
    status_counter = Counter(str(row["reposition_status"]) for row in rows)
    failed_steps = [int(row["t"]) for row in rows if row["error"]]

    print(f"Diagnostic rows: {len(rows)}")
    print(f"Trace CSV: {output_path}")
    print("Outcome counts:", dict(outcome_counter))
    print("Status counts:", dict(status_counter))
    print("Failed steps:", failed_steps)
    print(
        "Summary JSON:",
        json.dumps(
            {
                "rows": len(rows),
                "outcome_counts": dict(outcome_counter),
                "status_counts": dict(status_counter),
                "failed_steps": failed_steps,
                "output_csv": str(output_path),
            },
            ensure_ascii=False,
        ),
    )


if __name__ == "__main__":
    main()
