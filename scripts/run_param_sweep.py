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
    load_zone_map,
    load_yellow_trip_demand,
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
from etaxi_sim.policies.reposition import (
    get_gurobi_reposition_trace_stats,
    get_last_gurobi_reposition_trace,
    reset_gurobi_reposition_trace_stats,
)
from etaxi_sim.sim.core import Simulation
from etaxi_sim.sim.metrics import MetricsRecorder


def parse_int_list(raw: str) -> list[int]:
    values = [x.strip() for x in raw.split(",") if x.strip()]
    return [int(v) for v in values]


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


def compute_effective_horizon(time_bin_minutes: int, horizon: int, start_hour: int, end_hour: int) -> int:
    if not (0 <= start_hour < end_hour <= 24):
        raise ValueError("Invalid time window: require 0 <= sim_start_hour < sim_end_hour <= 24")
    window_slots = int((end_hour - start_hour) * 60 / time_bin_minutes)
    return max(1, min(horizon, window_slots))


def case_key(
    inventory: int,
    chargers: int,
    swap_capacity: int,
    reposition_solver: str,
    charging_solver: str,
) -> tuple[str, str, str, str, str]:
    return (
        str(inventory),
        str(chargers),
        str(swap_capacity),
        str(reposition_solver),
        str(charging_solver),
    )


def load_existing_rows(output_csv: Path) -> list[dict[str, str]]:
    if not output_csv.exists() or output_csv.stat().st_size == 0:
        return []
    with output_csv.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(output_csv: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    tmp_csv = output_csv.with_suffix(f"{output_csv.suffix}.tmp")
    with tmp_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
    tmp_csv.replace(output_csv)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ev_yellow_2025_11.yaml")
    parser.add_argument("--inventories", default="50,100,150,200")
    parser.add_argument("--chargers", default="5,10,15")
    parser.add_argument("--swap-capacities", default="6,12,18")
    parser.add_argument("--output-csv", default="results/param_sweep/summary.csv")
    parser.add_argument("--reposition-solver", default=None)
    parser.add_argument("--charging-solver", default=None)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="resume from existing output-csv and skip completed cases",
    )
    args = parser.parse_args()

    inventories = parse_int_list(args.inventories)
    chargers_grid = parse_int_list(args.chargers)
    swap_capacities = parse_int_list(args.swap_capacities)

    cfg = load_config(args.config)
    effective_horizon = compute_effective_horizon(
        time_bin_minutes=cfg.sim.time_bin_minutes,
        horizon=cfg.sim.horizon,
        start_hour=cfg.sim.sim_start_hour,
        end_hour=cfg.sim.sim_end_hour,
    )
    zone_map = load_zone_map(cfg.data.taxi_zone_lookup_path)
    m = len(zone_map.zone_ids)
    reposition_solver = args.reposition_solver or cfg.model.reposition_solver
    charging_solver = args.charging_solver or cfg.model.charging_solver

    if cfg.data.source == "yellow_tripdata":
        demand_data = load_yellow_trip_demand(
            cfg.data.yellow_tripdata_path,
            zone_map,
            cfg.sim.time_bin_minutes,
            effective_horizon,
            cfg.sim.sim_date,
            cfg.sim.sim_start_hour,
            cfg.sim.sim_end_hour,
        )
    else:
        demand_data = make_synthetic_demand(effective_horizon, m, cfg.sim.seed)

    if cfg.model.transition_mode == "historical_tripdata" and cfg.data.source == "yellow_tripdata":
        transition = make_transition_from_tripdata(
            yellow_tripdata_path=cfg.data.yellow_tripdata_path,
            zone_ids=zone_map.zone_ids,
            zone_index=zone_map.zone_index,
            time_bin_minutes=cfg.sim.time_bin_minutes,
            horizon=effective_horizon,
            sim_date=cfg.sim.sim_date,
            sim_start_hour=cfg.sim.sim_start_hour,
            sim_end_hour=cfg.sim.sim_end_hour,
            pickup_prob_floor=cfg.model.transition_pickup_floor,
            pickup_prob_ceiling=cfg.model.transition_pickup_ceiling,
            pickup_smoothing_window=cfg.model.transition_pickup_smoothing_window,
        )
    else:
        transition = make_uniform_transition(effective_horizon, m)

    if cfg.model.distance_mode == "taxi_zones":
        reachability = reachability_from_taxi_zones(cfg.data.taxi_zones_shp_path, zone_map.zone_ids)
        energy_consumption = energy_consumption_from_reachability(reachability)
    else:
        distance_matrix = make_distance_matrix(m, cfg.sim.seed)
        reachability = make_reachability(distance_matrix, cfg.model.max_reachable_distance)
        energy_consumption = energy_consumption_matrix(distance_matrix, cfg.sim.battery_levels)

    initial_vehicles = cfg.sim.initial_vehicles
    if cfg.sim.initial_vehicles_mode == "from_data_peak_overlap" and cfg.data.source == "yellow_tripdata":
        peak_overlap = estimate_peak_concurrent_trips(
            cfg.data.yellow_tripdata_path,
            cfg.sim.sim_date,
            cfg.sim.sim_start_hour,
            cfg.sim.sim_end_hour,
        )
        initial_vehicles = max(1, int(np.ceil(peak_overlap * cfg.sim.initial_vehicles_scale)))

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not args.resume and output_csv.exists():
        output_csv.unlink()

    rows: list[dict] = []
    if args.resume:
        rows = load_existing_rows(output_csv)
    existing_keys: set[tuple[str, str, str, str, str]] = set()
    for row in rows:
        try:
            inventory = int(float(row.get("inventory_per_station", "")))
            chargers = int(float(row.get("chargers_per_station", "")))
            swap_capacity = int(float(row.get("swap_capacity_per_station", "")))
            rep_solver = str(row.get("reposition_solver", ""))
            chg_solver = str(row.get("charging_solver", ""))
        except (TypeError, ValueError):
            continue
        if not rep_solver or not chg_solver:
            continue
        existing_keys.add(case_key(inventory, chargers, swap_capacity, rep_solver, chg_solver))

    case_idx = 0
    total_cases = len(inventories) * len(chargers_grid) * len(swap_capacities)
    for inventory in inventories:
        for chargers in chargers_grid:
            for swap_capacity in swap_capacities:
                case_idx += 1
                key = case_key(
                    inventory=inventory,
                    chargers=chargers,
                    swap_capacity=swap_capacity,
                    reposition_solver=reposition_solver,
                    charging_solver=charging_solver,
                )
                if args.resume and key in existing_keys:
                    print(
                        f"[{case_idx}/{total_cases}] skip completed "
                        f"inventory={inventory}, chargers={chargers}, swap_capacity={swap_capacity}"
                    )
                    continue

                print(
                    f"[{case_idx}/{total_cases}] inventory={inventory}, "
                    f"chargers={chargers}, swap_capacity={swap_capacity}"
                )

                fleet = initialize_fleet(m, cfg.sim.battery_levels, cfg.sim.seed, initial_vehicles)
                stations = build_stations(
                    m=m,
                    levels=cfg.sim.battery_levels,
                    swapping_capacity=swap_capacity,
                    chargers=chargers,
                    full_batteries=inventory,
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
                    horizon=effective_horizon,
                    demand_forecast=demand_data.demand,
                    charge_rate_levels_per_slot=cfg.sim.charge_rate_levels_per_slot,
                    deadline_horizon=cfg.sim.swap_deadline_horizon,
                    charge_power_kw=cfg.sim.charge_power_kw,
                    reposition_solver=reposition_solver,
                    reposition_planning_horizon_slots=cfg.model.reposition_planning_horizon_slots,
                    charging_solver=charging_solver,
                    reposition_idle_cost_weight=cfg.model.reposition_idle_cost_weight,
                    reposition_top_demand_targets=cfg.model.reposition_top_demand_targets,
                    reposition_top_swap_targets=cfg.model.reposition_top_swap_targets,
                    reposition_low_energy_swap_bonus=cfg.model.reposition_low_energy_swap_bonus,
                    reposition_transition_topk=cfg.model.reposition_transition_topk,
                    reposition_solver_method=cfg.model.reposition_solver_method,
                    reposition_solver_crossover=cfg.model.reposition_solver_crossover,
                    reposition_numeric_focus=cfg.model.reposition_numeric_focus,
                    reposition_eliminate_auxiliary_vars=cfg.model.reposition_eliminate_auxiliary_vars,
                    reposition_preaggregate_transitions=cfg.model.reposition_preaggregate_transitions,
                    charging_miss_penalty=cfg.model.charging_miss_penalty,
                    solver_time_limit_sec=cfg.model.solver_time_limit_sec,
                    metrics=metrics,
                    charging_tasks=[],
                    waiting_queue=np.zeros((m, cfg.sim.battery_levels + 1), dtype=int),
                    active_charging_task_ids_by_station={},
                    task_counter=0,
                    rng=np.random.default_rng(cfg.sim.seed),
                )

                trace_outcomes = Counter()
                trace_statuses = Counter()
                trace_runtime_samples: list[float] = []
                trace_no_solution_steps = 0
                use_gurobi_reposition = reposition_solver in {"gurobi", "ideal"}
                if use_gurobi_reposition:
                    reset_gurobi_reposition_trace_stats()

                for t in range(effective_horizon):
                    sim.step(t, demand_data.demand[t], cfg.sim.swap_low_energy_threshold)
                    if use_gurobi_reposition:
                        trace = get_last_gurobi_reposition_trace()
                        if int(trace.get("t_start", -1) or -1) != t:
                            continue
                        outcome = str(trace.get("outcome", "unknown"))
                        trace_outcomes[outcome] += 1
                        status_val = trace.get("status")
                        if status_val is not None:
                            trace_statuses[str(int(status_val))] += 1
                        sol_count = trace.get("sol_count")
                        if sol_count is None or int(sol_count) <= 0:
                            trace_no_solution_steps += 1
                        runtime_sec = trace.get("runtime_sec")
                        if runtime_sec is not None:
                            trace_runtime_samples.append(float(runtime_sec))

                summary = metrics.to_summary()
                trace_event_stats = (
                    get_gurobi_reposition_trace_stats() if use_gurobi_reposition else {"outcome_counts": {}, "status_counts": {}}
                )
                trace_event_outcomes = trace_event_stats["outcome_counts"]
                trace_retry_events = int(
                    sum(
                        count
                        for outcome, count in trace_event_outcomes.items()
                        if outcome.startswith("retry_") or outcome.startswith("fallback_")
                    )
                )
                trace_no_incumbent_events = int(
                    sum(
                        trace_event_outcomes.get(name, 0)
                        for name in ("no_incumbent", "retry_h1_no_incumbent", "retry_t20_no_incumbent")
                    )
                )
                row = {
                    "inventory_per_station": inventory,
                    "chargers_per_station": chargers,
                    "swap_capacity_per_station": swap_capacity,
                    "reposition_solver": reposition_solver,
                    "charging_solver": charging_solver,
                    "sim_start_hour": cfg.sim.sim_start_hour,
                    "sim_end_hour": cfg.sim.sim_end_hour,
                    "effective_horizon": effective_horizon,
                    "reposition_step_outcome_counts": json.dumps(
                        dict(trace_outcomes), ensure_ascii=False, sort_keys=True
                    ),
                    "reposition_step_status_counts": json.dumps(
                        dict(trace_statuses), ensure_ascii=False, sort_keys=True
                    ),
                    "reposition_step_no_solution_steps": trace_no_solution_steps,
                    "reposition_step_runtime_sec_avg": (
                        float(np.mean(trace_runtime_samples)) if trace_runtime_samples else 0.0
                    ),
                    "reposition_step_runtime_sec_p95": (
                        float(np.quantile(np.array(trace_runtime_samples, dtype=float), 0.95))
                        if trace_runtime_samples
                        else 0.0
                    ),
                    "reposition_step_runtime_sec_max": (
                        float(max(trace_runtime_samples)) if trace_runtime_samples else 0.0
                    ),
                    "reposition_event_outcome_counts": json.dumps(
                        dict(trace_event_outcomes), ensure_ascii=False, sort_keys=True
                    ),
                    "reposition_event_status_counts": json.dumps(
                        dict(trace_event_stats["status_counts"]), ensure_ascii=False, sort_keys=True
                    ),
                    "reposition_event_retry_count": trace_retry_events,
                    "reposition_event_no_incumbent_count": trace_no_incumbent_events,
                }
                row.update(summary)
                rows.append(row)
                existing_keys.add(key)
                write_rows(output_csv, rows)
                print(
                    f"  saved case -> {output_csv} "
                    f"(rows={len(rows)}, retry_events={trace_retry_events}, no_solution_steps={trace_no_solution_steps})"
                )

    if rows:
        print(f"Saved sweep summary to: {output_csv} (rows={len(rows)})")
    else:
        print(f"No rows written. Output path: {output_csv}")


if __name__ == "__main__":
    main()
