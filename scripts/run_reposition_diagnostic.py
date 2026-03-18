from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

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
    parser.add_argument("--output-summary-json", default="")
    parser.add_argument("--max-steps", type=int, default=0, help="0 means full horizon")
    parser.add_argument(
        "--max-run-seconds",
        type=float,
        default=0.0,
        help="hard stop for total diagnostic wall time; 0 means unlimited",
    )
    parser.add_argument("--reposition-solver", default=None)
    parser.add_argument("--charging-solver", default=None)
    parser.add_argument(
        "--solver-time-limit-sec",
        type=float,
        default=None,
        help="override solver time limit for reposition/charging models",
    )
    parser.add_argument(
        "--reposition-solver-method",
        type=int,
        default=None,
        help="override Gurobi Method for reposition model",
    )
    parser.add_argument(
        "--reposition-solver-crossover",
        type=int,
        default=None,
        help="override Gurobi Crossover for reposition model",
    )
    parser.add_argument(
        "--reposition-numeric-focus",
        type=int,
        default=None,
        help="override Gurobi NumericFocus for reposition model",
    )
    parser.add_argument(
        "--reposition-presolve",
        type=int,
        default=None,
        help="override Gurobi Presolve for reposition model",
    )
    parser.add_argument(
        "--reposition-use-lp-primal-start",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="toggle LP primal warm start construction for reposition model",
    )
    parser.add_argument(
        "--reposition-lp-warm-start-mode",
        type=int,
        default=None,
        help="override Gurobi LPWarmStart for reposition model",
    )
    parser.add_argument("--inventory", type=int, default=None)
    parser.add_argument("--chargers", type=int, default=None)
    parser.add_argument("--swap-capacity", type=int, default=None)
    parser.add_argument(
        "--legacy-dense",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="disable auxiliary-variable elimination for before/after comparison",
    )
    parser.add_argument(
        "--disable-preaggregation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="disable transition/service preaggregation for before baseline",
    )
    parser.add_argument(
        "--continue-on-error",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="continue loop after step exceptions (best effort)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not (0 <= cfg.sim.sim_start_hour < cfg.sim.sim_end_hour <= 24):
        raise ValueError("Invalid sim window: require 0 <= sim_start_hour < sim_end_hour <= 24")
    window_slots = int((cfg.sim.sim_end_hour - cfg.sim.sim_start_hour) * 60 / cfg.sim.time_bin_minutes)
    effective_horizon = max(1, min(cfg.sim.horizon, window_slots))
    zone_map = load_zone_map(cfg.data.taxi_zone_lookup_path)
    m = len(zone_map.zone_ids)

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
        print(
            f"Initial vehicles from data peak overlap: peak={peak_overlap}, "
            f"scale={cfg.sim.initial_vehicles_scale}, used={initial_vehicles}"
        )

    fleet = initialize_fleet(m, cfg.sim.battery_levels, cfg.sim.seed, initial_vehicles)
    stations = build_stations(
        m=m,
        levels=cfg.sim.battery_levels,
        swapping_capacity=int(args.swap_capacity if args.swap_capacity is not None else cfg.station.swapping_capacity),
        chargers=int(args.chargers if args.chargers is not None else cfg.station.chargers),
        full_batteries=int(
            args.inventory if args.inventory is not None else cfg.sim.min_full_batteries_per_station
        ),
        base_load_kw=cfg.sim.base_load_kw,
    )

    metrics = MetricsRecorder()
    reposition_solver = str(args.reposition_solver or cfg.model.reposition_solver)
    charging_solver = str(args.charging_solver or cfg.model.charging_solver)
    eliminate_auxiliary_vars = (
        False if args.legacy_dense else bool(cfg.model.reposition_eliminate_auxiliary_vars)
    )
    preaggregate_transitions = (
        False if args.disable_preaggregation else bool(cfg.model.reposition_preaggregate_transitions)
    )
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
        reposition_solver_method=(
            int(args.reposition_solver_method)
            if args.reposition_solver_method is not None
            else cfg.model.reposition_solver_method
        ),
        reposition_solver_crossover=(
            int(args.reposition_solver_crossover)
            if args.reposition_solver_crossover is not None
            else cfg.model.reposition_solver_crossover
        ),
        reposition_numeric_focus=(
            int(args.reposition_numeric_focus)
            if args.reposition_numeric_focus is not None
            else cfg.model.reposition_numeric_focus
        ),
        reposition_presolve=(
            int(args.reposition_presolve)
            if args.reposition_presolve is not None
            else cfg.model.reposition_presolve
        ),
        reposition_use_lp_primal_start=(
            bool(args.reposition_use_lp_primal_start)
            if args.reposition_use_lp_primal_start is not None
            else bool(cfg.model.reposition_use_lp_primal_start)
        ),
        reposition_lp_warm_start_mode=(
            int(args.reposition_lp_warm_start_mode)
            if args.reposition_lp_warm_start_mode is not None
            else int(cfg.model.reposition_lp_warm_start_mode)
        ),
        reposition_eliminate_auxiliary_vars=eliminate_auxiliary_vars,
        reposition_preaggregate_transitions=preaggregate_transitions,
        charging_miss_penalty=cfg.model.charging_miss_penalty,
        solver_time_limit_sec=(
            float(args.solver_time_limit_sec)
            if args.solver_time_limit_sec is not None
            else cfg.model.solver_time_limit_sec
        ),
        metrics=metrics,
        charging_tasks=[],
        waiting_queue=np.zeros((m, cfg.sim.battery_levels + 1), dtype=int),
        active_charging_task_ids_by_station={},
        task_counter=0,
        rng=np.random.default_rng(cfg.sim.seed),
    )

    limit = effective_horizon if args.max_steps <= 0 else min(effective_horizon, args.max_steps)
    rows: list[dict[str, int | float | str | None]] = []
    run_start = perf_counter()
    timed_out = False

    for t in range(limit):
        if args.max_run_seconds > 0 and (perf_counter() - run_start) >= float(args.max_run_seconds):
            timed_out = True
            break
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
            "reposition_runtime_sec": trace.get("runtime_sec"),
            "reposition_build_time_sec": trace.get("build_time_sec"),
            "reposition_candidate_prep_time_sec": trace.get("candidate_prep_time_sec"),
            "reposition_vars_build_time_sec": trace.get("vars_build_time_sec"),
            "reposition_expressions_build_time_sec": trace.get("expressions_build_time_sec"),
            "reposition_constraints_build_time_sec": trace.get("constraints_build_time_sec"),
            "reposition_objective_build_time_sec": trace.get("objective_build_time_sec"),
            "reposition_optimize_time_sec": trace.get("optimize_time_sec"),
            "reposition_wall_time_sec": trace.get("wall_time_sec"),
            "reposition_outcome": trace.get("outcome"),
            "reposition_note": trace.get("note"),
            "reposition_num_vars": trace.get("num_vars"),
            "reposition_num_constrs": trace.get("num_constrs"),
            "reposition_num_nz": trace.get("num_nz"),
            "reposition_iter_count": trace.get("iter_count"),
            "reposition_bar_iter_count": trace.get("bar_iter_count"),
            "reposition_obj_val": trace.get("obj_val"),
            "reposition_obj_bound": trace.get("obj_bound"),
            "reposition_solver_method": trace.get("solver_method"),
            "reposition_solver_crossover": trace.get("solver_crossover"),
            "reposition_numeric_focus": trace.get("numeric_focus"),
            "reposition_presolve": trace.get("presolve"),
            "reposition_lp_warm_start_enabled": trace.get("lp_warm_start_enabled"),
            "reposition_lp_warm_start_mode": trace.get("lp_warm_start_mode"),
            "reposition_lp_warm_start_applied": trace.get("lp_warm_start_applied"),
            "reposition_lp_warm_start_var_count": trace.get("lp_warm_start_var_count"),
            "reposition_lp_warm_start_nonzero_count": trace.get("lp_warm_start_nonzero_count"),
            "reposition_lp_warm_start_note": trace.get("lp_warm_start_note"),
            "reposition_use_preaggregation": trace.get("use_preaggregation"),
            "reposition_use_aux_elimination": trace.get("use_aux_elimination"),
            "x_var_count": trace.get("x_var_count"),
            "y_var_count": trace.get("y_var_count"),
            "v_var_count": trace.get("v_var_count"),
            "o_var_count": trace.get("o_var_count"),
            "r_var_count": trace.get("r_var_count"),
            "b_var_count": trace.get("b_var_count"),
            "mu_var_count": trace.get("mu_var_count"),
            "served_var_count": trace.get("served_var_count"),
            "full_stock_var_count": trace.get("full_stock_var_count"),
            "vacant_split_constr_count": trace.get("vacant_split_constr_count"),
            "b_def_constr_count": trace.get("b_def_constr_count"),
            "mu_le_b_constr_count": trace.get("mu_le_b_constr_count"),
            "swap_cap_constr_count": trace.get("swap_cap_constr_count"),
            "full_dyn_constr_count": trace.get("full_dyn_constr_count"),
            "served_by_supply_constr_count": trace.get("served_by_supply_constr_count"),
            "o_dyn_constr_count": trace.get("o_dyn_constr_count"),
            "v_dyn_constr_count": trace.get("v_dyn_constr_count"),
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
        "reposition_runtime_sec",
        "reposition_build_time_sec",
        "reposition_candidate_prep_time_sec",
        "reposition_vars_build_time_sec",
        "reposition_expressions_build_time_sec",
        "reposition_constraints_build_time_sec",
        "reposition_objective_build_time_sec",
        "reposition_optimize_time_sec",
        "reposition_wall_time_sec",
        "reposition_outcome",
        "reposition_note",
        "reposition_num_vars",
        "reposition_num_constrs",
        "reposition_num_nz",
        "reposition_iter_count",
        "reposition_bar_iter_count",
        "reposition_obj_val",
        "reposition_obj_bound",
        "reposition_solver_method",
        "reposition_solver_crossover",
        "reposition_numeric_focus",
        "reposition_presolve",
        "reposition_lp_warm_start_enabled",
        "reposition_lp_warm_start_mode",
        "reposition_lp_warm_start_applied",
        "reposition_lp_warm_start_var_count",
        "reposition_lp_warm_start_nonzero_count",
        "reposition_lp_warm_start_note",
        "reposition_use_preaggregation",
        "reposition_use_aux_elimination",
        "x_var_count",
        "y_var_count",
        "v_var_count",
        "o_var_count",
        "r_var_count",
        "b_var_count",
        "mu_var_count",
        "served_var_count",
        "full_stock_var_count",
        "vacant_split_constr_count",
        "b_def_constr_count",
        "mu_le_b_constr_count",
        "swap_cap_constr_count",
        "full_dyn_constr_count",
        "served_by_supply_constr_count",
        "o_dyn_constr_count",
        "v_dyn_constr_count",
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
    run_wall_time_sec = float(perf_counter() - run_start)
    summary_payload = {
        "rows": len(rows),
        "effective_horizon": int(effective_horizon),
        "requested_limit": int(limit),
        "executed_steps": len(rows),
        "timed_out": bool(timed_out),
        "max_run_seconds": float(args.max_run_seconds),
        "run_wall_time_sec": run_wall_time_sec,
        "reposition_solver": reposition_solver,
        "charging_solver": charging_solver,
        "inventory_per_station": int(
            args.inventory if args.inventory is not None else cfg.sim.min_full_batteries_per_station
        ),
        "chargers_per_station": int(args.chargers if args.chargers is not None else cfg.station.chargers),
        "swap_capacity_per_station": int(
            args.swap_capacity if args.swap_capacity is not None else cfg.station.swapping_capacity
        ),
        "solver_time_limit_sec": float(
            args.solver_time_limit_sec if args.solver_time_limit_sec is not None else cfg.model.solver_time_limit_sec
        ),
        "reposition_solver_method": int(
            args.reposition_solver_method
            if args.reposition_solver_method is not None
            else cfg.model.reposition_solver_method
        ),
        "reposition_solver_crossover": int(
            args.reposition_solver_crossover
            if args.reposition_solver_crossover is not None
            else cfg.model.reposition_solver_crossover
        ),
        "reposition_numeric_focus": int(
            args.reposition_numeric_focus
            if args.reposition_numeric_focus is not None
            else cfg.model.reposition_numeric_focus
        ),
        "reposition_presolve": int(
            args.reposition_presolve
            if args.reposition_presolve is not None
            else cfg.model.reposition_presolve
        ),
        "reposition_use_lp_primal_start": bool(
            args.reposition_use_lp_primal_start
            if args.reposition_use_lp_primal_start is not None
            else cfg.model.reposition_use_lp_primal_start
        ),
        "reposition_lp_warm_start_mode": int(
            args.reposition_lp_warm_start_mode
            if args.reposition_lp_warm_start_mode is not None
            else cfg.model.reposition_lp_warm_start_mode
        ),
        "reposition_eliminate_auxiliary_vars": bool(eliminate_auxiliary_vars),
        "reposition_preaggregate_transitions": bool(preaggregate_transitions),
        "outcome_counts": dict(outcome_counter),
        "status_counts": dict(status_counter),
        "failed_steps": failed_steps,
        "output_csv": str(output_path),
        "metrics_summary": metrics.to_summary(),
    }
    summary_path = (
        Path(args.output_summary_json)
        if args.output_summary_json
        else output_path.with_suffix(".summary.json")
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    print(f"Diagnostic rows: {len(rows)}")
    print(f"Trace CSV: {output_path}")
    print(f"Summary JSON path: {summary_path}")
    print("Outcome counts:", dict(outcome_counter))
    print("Status counts:", dict(status_counter))
    print("Failed steps:", failed_steps)
    print(
        "Summary JSON:",
        json.dumps(summary_payload, ensure_ascii=False),
    )


if __name__ == "__main__":
    main()
