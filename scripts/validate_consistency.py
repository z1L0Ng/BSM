from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

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
from etaxi_sim.policies.charging import (
    ChargingPolicyConfig,
    edf_charging_policy,
    fcfs_nonpreemptive_charging_policy,
    gurobi_peak_charging_policy,
)
from etaxi_sim.policies.reposition import (
    RepositionPolicyConfig,
    greedy_same_zone_policy,
    gurobi_reposition_policy,
    heuristic_battery_aware_policy,
)
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

    init_battery_total = int(sum(s.full_batteries + int(s.partial_batteries.sum()) for s in sim.stations))

    # Diagnostics accumulators
    eq9_unreachable_viol = 0
    eq9_energy_viol = 0
    dispatch_overflow = 0
    eq4_vehicle_viol = 0
    eq4_full_viol = 0
    eq4_capacity_viol = 0
    eq5_charger_viol = 0
    trans_occ_max_err = 0.0
    trans_vac_max_err = 0.0
    vehicle_min = 10**18
    vehicle_max = -1
    battery_min = 10**18
    battery_max = -1

    for t in range(cfg.sim.horizon):
        demand_t = demand_data.demand[t]
        window_end = min(cfg.sim.horizon, t + max(1, cfg.model.reposition_planning_horizon_slots))
        demand_window = demand_data.demand[t:window_end]

        rep_cfg = RepositionPolicyConfig(
            swap_low_energy_threshold=cfg.sim.swap_low_energy_threshold,
            planning_horizon_slots=cfg.model.reposition_planning_horizon_slots,
            top_demand_targets=cfg.model.reposition_top_demand_targets,
            top_swap_targets=cfg.model.reposition_top_swap_targets,
            idle_cost_weight=cfg.model.reposition_idle_cost_weight,
            service_reward=1.0,
            low_energy_swap_bonus=cfg.model.reposition_low_energy_swap_bonus,
            transition_topk=cfg.model.reposition_transition_topk,
            time_limit_sec=cfg.model.solver_time_limit_sec,
        )
        if cfg.model.reposition_solver == "gurobi":
            X, Y = gurobi_reposition_policy(
                fleet=sim.fleet,
                demand_window=demand_window,
                reachability=reachability,
                energy_consumption=energy_consumption,
                station_full_batteries=np.array([s.full_batteries for s in sim.stations], dtype=int),
                station_swapping_capacity=np.array([s.swapping_capacity for s in sim.stations], dtype=int),
                transition=transition,
                t_start=t,
                config=rep_cfg,
            )
        elif cfg.model.reposition_solver == "heuristic":
            X, Y = heuristic_battery_aware_policy(
                fleet=sim.fleet,
                demand=demand_window[0],
                station_full_batteries=np.array([s.full_batteries for s in sim.stations], dtype=int),
                config=rep_cfg,
            )
        elif cfg.model.reposition_solver == "ideal":
            ideal_cap = int(max(1, sim.fleet.vacant.sum()))
            X, Y = gurobi_reposition_policy(
                fleet=sim.fleet,
                demand_window=demand_window,
                reachability=reachability,
                energy_consumption=energy_consumption,
                station_full_batteries=np.full(len(sim.stations), ideal_cap, dtype=int),
                station_swapping_capacity=np.full(len(sim.stations), ideal_cap, dtype=int),
                transition=transition,
                t_start=t,
                config=rep_cfg,
            )
        else:
            X, Y = greedy_same_zone_policy(sim.fleet, demand_window[0], rep_cfg)

        # Eq. (9): reachability and energy feasibility.
        unreachable_mask = (reachability == 1)[:, :, None]
        eq9_unreachable_viol += int((X * unreachable_mask).sum() + (Y * unreachable_mask).sum())
        levels = np.arange(cfg.sim.battery_levels + 1)[None, None, :]
        bad_energy = levels < energy_consumption[:, :, None]
        eq9_energy_viol += int((X * bad_energy).sum() + (Y * bad_energy).sum())

        dispatched = X.sum(axis=1) + Y.sum(axis=1)
        dispatch_overflow += int(np.maximum(dispatched - sim.fleet.vacant, 0).sum())

        B = Y.sum(axis=0)
        chargers_by_station = {s.station_id: s.chargers for s in sim.stations}

        for i, station in enumerate(sim.stations):
            st = copy.deepcopy(station)
            swapped, _ = st.perform_swapping(B[i].copy())
            mu = swapped
            eq4_vehicle_viol += int(np.maximum(mu - B[i], 0).sum())
            eq4_full_viol += int(max(0, int(mu.sum()) - int(station.full_batteries)))
            eq4_capacity_viol += int(max(0, int(mu.sum()) - int(station.swapping_capacity)))

        if cfg.model.charging_solver == "gurobi":
            charge_cfg = ChargingPolicyConfig(
                planning_horizon_slots=cfg.sim.swap_deadline_horizon,
                charge_power_kw=cfg.sim.charge_power_kw,
                miss_penalty=cfg.model.charging_miss_penalty,
                time_limit_sec=cfg.model.solver_time_limit_sec,
            )
            charged = gurobi_peak_charging_policy(
                tasks=sim.charging_tasks,
                chargers_by_station=chargers_by_station,
                base_load_by_station={s.station_id: s.base_load_kw for s in sim.stations},
                current_time=t,
                config=charge_cfg,
            )
        elif cfg.model.charging_solver == "fcfs":
            charged, _ = fcfs_nonpreemptive_charging_policy(
                tasks=sim.charging_tasks,
                chargers_by_station=chargers_by_station,
                current_time=t,
                active_task_ids_by_station=sim.active_charging_task_ids_by_station,
            )
        else:
            charged = edf_charging_policy(sim.charging_tasks, chargers_by_station, current_time=t)
        for sid, cap in chargers_by_station.items():
            used = sum(1 for task in charged if task.station_id == sid)
            eq5_charger_viol += max(0, used - cap)

        occ_err = np.max(np.abs((transition.P[t] + transition.Q[t]).sum(axis=1) - 1.0))
        vac_err = np.max(np.abs((transition.P_tilde[t] + transition.Q_tilde[t]).sum(axis=1) - 1.0))
        trans_occ_max_err = max(trans_occ_max_err, float(occ_err))
        trans_vac_max_err = max(trans_vac_max_err, float(vac_err))

        sim.step(t, demand_t, cfg.sim.swap_low_energy_threshold)

        total_vehicles = int(sim.fleet.vacant.sum() + sim.fleet.occupied.sum() + sim.waiting_queue.sum())
        vehicle_min = min(vehicle_min, total_vehicles)
        vehicle_max = max(vehicle_max, total_vehicles)

        total_battery = int(sum(s.full_batteries + int(s.partial_batteries.sum()) for s in sim.stations))
        battery_min = min(battery_min, total_battery)
        battery_max = max(battery_max, total_battery)

    result = {
        "config": str(args.config),
        "initial_vehicles": int(initial_vehicles),
        "vehicle_total_min": int(vehicle_min),
        "vehicle_total_max": int(vehicle_max),
        "battery_total_min": int(battery_min),
        "battery_total_max": int(battery_max),
        "initial_battery_total": int(init_battery_total),
        "eq9_unreachable_violations": int(eq9_unreachable_viol),
        "eq9_energy_violations": int(eq9_energy_viol),
        "dispatch_overflow_violations": int(dispatch_overflow),
        "eq4_mu_gt_B_violations": int(eq4_vehicle_viol),
        "eq4_mu_gt_full_violations": int(eq4_full_viol),
        "eq4_mu_gt_swap_capacity_violations": int(eq4_capacity_viol),
        "eq5_charger_capacity_violations": int(eq5_charger_viol),
        "transition_occ_row_max_abs_error": float(trans_occ_max_err),
        "transition_vac_row_max_abs_error": float(trans_vac_max_err),
        "metrics_summary": metrics.to_summary(),
    }
    print(result)


if __name__ == "__main__":
    main()
