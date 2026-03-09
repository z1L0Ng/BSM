from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from etaxi_sim.data.preprocess import TransitionProbabilities
from etaxi_sim.models.charging import ChargingTask
from etaxi_sim.models.fleet import FleetState
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
from etaxi_sim.sim.metrics import MetricsRecorder


@dataclass
class Simulation:
    fleet: FleetState
    stations: List[Station]
    transition: TransitionProbabilities
    energy_consumption: np.ndarray
    reachability: np.ndarray
    levels: int
    horizon: int
    demand_forecast: np.ndarray
    charge_rate_levels_per_slot: int
    deadline_horizon: int
    charge_power_kw: float
    reposition_solver: str
    reposition_planning_horizon_slots: int
    charging_solver: str
    reposition_idle_cost_weight: float
    reposition_top_demand_targets: int
    reposition_top_swap_targets: int
    reposition_low_energy_swap_bonus: float
    reposition_transition_topk: int
    charging_miss_penalty: float
    solver_time_limit_sec: float
    metrics: MetricsRecorder

    charging_tasks: List[ChargingTask]
    waiting_queue: np.ndarray
    active_charging_task_ids_by_station: dict[int, list[int]]
    task_counter: int
    rng: np.random.Generator

    def step(self, t: int, demand: np.ndarray, swap_threshold: int) -> None:
        window_end = min(self.horizon, t + max(1, self.reposition_planning_horizon_slots))
        demand_window = self.demand_forecast[t:window_end]
        if demand_window.size == 0:
            demand_window = demand[None, :]

        # 1) Reposition policy
        reposition_cfg = RepositionPolicyConfig(
            swap_low_energy_threshold=swap_threshold,
            planning_horizon_slots=self.reposition_planning_horizon_slots,
            top_demand_targets=self.reposition_top_demand_targets,
            top_swap_targets=self.reposition_top_swap_targets,
            idle_cost_weight=self.reposition_idle_cost_weight,
            service_reward=1.0,
            low_energy_swap_bonus=self.reposition_low_energy_swap_bonus,
            transition_topk=self.reposition_transition_topk,
            time_limit_sec=self.solver_time_limit_sec,
        )
        if self.reposition_solver == "gurobi":
            X, Y = gurobi_reposition_policy(
                fleet=self.fleet,
                demand_window=demand_window,
                reachability=self.reachability,
                energy_consumption=self.energy_consumption,
                station_full_batteries=np.array([s.full_batteries for s in self.stations], dtype=int),
                station_swapping_capacity=np.array([s.swapping_capacity for s in self.stations], dtype=int),
                transition=self.transition,
                t_start=t,
                config=reposition_cfg,
            )
        elif self.reposition_solver == "heuristic":
            X, Y = heuristic_battery_aware_policy(
                fleet=self.fleet,
                demand=demand_window[0],
                station_full_batteries=np.array([s.full_batteries for s in self.stations], dtype=int),
                config=reposition_cfg,
            )
        elif self.reposition_solver == "ideal":
            ideal_cap = int(max(1, self.fleet.vacant.sum()))
            X, Y = gurobi_reposition_policy(
                fleet=self.fleet,
                demand_window=demand_window,
                reachability=self.reachability,
                energy_consumption=self.energy_consumption,
                station_full_batteries=np.full(len(self.stations), ideal_cap, dtype=int),
                station_swapping_capacity=np.full(len(self.stations), ideal_cap, dtype=int),
                transition=self.transition,
                t_start=t,
                config=reposition_cfg,
            )
        else:
            X, Y = greedy_same_zone_policy(self.fleet, demand_window[0], reposition_cfg)
        # Enforce reachability: nu[i,i'] = 1 means unreachable within one slot
        reachable = (self.reachability == 0)[:, :, None]
        X = (X * reachable).astype(int, copy=False)
        Y = (Y * reachable).astype(int, copy=False)

        # Enforce energy feasibility: l >= E_{i,i'}
        levels = np.arange(self.levels + 1, dtype=int)[None, None, :]
        enough_energy = levels >= self.energy_consumption[:, :, None]
        X = (X * enough_energy).astype(int, copy=False)
        Y = (Y * enough_energy).astype(int, copy=False)
        move_counts = (X + Y).sum(axis=2)
        idle_driving_distance = float(np.sum(move_counts * self.energy_consumption))

        dispatched = X.sum(axis=1) + Y.sum(axis=1)
        residual_vacant = self.fleet.vacant - dispatched
        if (residual_vacant < 0).any():
            raise ValueError("Dispatch exceeds available vacant vehicles")

        # 2) Compute arriving vehicles
        B = Y.sum(axis=0)  # (m, L+1)
        S = X.sum(axis=0)  # (m, L+1)
        U = self.fleet.occupied.copy()
        swap_arrivals = int(B[:, : self.levels].sum())

        # 3) Swapping at stations
        H = np.zeros_like(B)
        next_waiting_queue = np.zeros_like(self.waiting_queue)
        successful_swaps = 0
        swap_requests = 0
        waiting_time_for_battery_slots = 0.0
        ideal_swap_assumption = self.reposition_solver == "ideal"
        for i, station in enumerate(self.stations):
            station_requests = self.waiting_queue[i].copy()
            station_requests[: self.levels] += B[i, : self.levels]
            station_requests[self.levels] = 0

            if ideal_swap_assumption:
                swapped = station_requests.copy()
                swapped[self.levels] = 0
                not_swapped = np.zeros_like(station_requests)
                for l in range(self.levels):
                    if swapped[l] > 0:
                        station.pending_charge[l] += int(swapped[l])
                        station.partial_batteries[l] += int(swapped[l])
            else:
                swapped, not_swapped = station.perform_swapping(station_requests)
            station_swapped = int(swapped[: self.levels].sum())
            successful_swaps += station_swapped
            swap_requests += int(station_requests[: self.levels].sum())

            next_waiting_queue[i, : self.levels] = not_swapped[: self.levels]
            waiting_time_for_battery_slots += float(next_waiting_queue[i, : self.levels].sum())

            # Swapped vehicles plus direct full-battery arrivals re-enter fleet as full.
            H[i, self.levels] = int(station_swapped + B[i, self.levels])
            # 4) Generate charging tasks
            new_tasks = station.generate_charging_tasks(
                swapped=swapped,
                current_time=t,
                charge_rate_levels_per_slot=self.charge_rate_levels_per_slot,
                deadline_horizon=self.deadline_horizon,
                task_id_start=self.task_counter,
            )
            self.task_counter += len(new_tasks)
            self.charging_tasks.extend(new_tasks)
        self.waiting_queue = next_waiting_queue
        waiting_by_station = self.waiting_queue[:, : self.levels].sum(axis=1)
        waiting_vehicles = int(self.waiting_queue[:, : self.levels].sum())
        unmet_battery_demand = waiting_vehicles

        # 5) Charging schedule
        charging_demand = sum(
            1 for task in self.charging_tasks if (not task.is_completed()) and task.is_available(t)
        )
        total_charging_demand_kw = float(self.charge_power_kw * charging_demand)

        chargers_by_station = {s.station_id: s.chargers for s in self.stations}
        if self.charging_solver == "gurobi":
            charging_cfg = ChargingPolicyConfig(
                planning_horizon_slots=self.deadline_horizon,
                charge_power_kw=self.charge_power_kw,
                miss_penalty=self.charging_miss_penalty,
                time_limit_sec=self.solver_time_limit_sec,
            )
            charged = gurobi_peak_charging_policy(
                tasks=self.charging_tasks,
                chargers_by_station=chargers_by_station,
                base_load_by_station={s.station_id: s.base_load_kw for s in self.stations},
                current_time=t,
                config=charging_cfg,
            )
        elif self.charging_solver == "fcfs":
            charged, self.active_charging_task_ids_by_station = fcfs_nonpreemptive_charging_policy(
                tasks=self.charging_tasks,
                chargers_by_station=chargers_by_station,
                current_time=t,
                active_task_ids_by_station=self.active_charging_task_ids_by_station,
            )
        else:
            charged = edf_charging_policy(self.charging_tasks, chargers_by_station, current_time=t)
            self.active_charging_task_ids_by_station = {}
        charged_by_station = {station.station_id: 0 for station in self.stations}
        for station in self.stations:
            station_tasks = [task for task in charged if task.station_id == station.station_id]
            station.apply_charging(station_tasks, self.charge_rate_levels_per_slot)
            charged_by_station[station.station_id] = len(station_tasks)

        # Remove completed or expired tasks, and track deadline misses.
        deadline_misses = sum(
            1 for task in self.charging_tasks if (t + 1 >= task.deadline and not task.is_completed())
        )
        self.charging_tasks = [
            task for task in self.charging_tasks if not task.is_completed() and (t + 1 < task.deadline)
        ]
        active_task_ids = {task.task_id for task in self.charging_tasks}
        self.active_charging_task_ids_by_station = {
            sid: [tid for tid in task_ids if tid in active_task_ids]
            for sid, task_ids in self.active_charging_task_ids_by_station.items()
        }

        # 6) State transition
        self.fleet = _state_transition(
            fleet=self.fleet,
            S=S,
            U=U,
            H=H,
            residual_vacant=residual_vacant,
            transition=self.transition,
            energy_consumption=self.energy_consumption,
            t=t,
            rng=self.rng,
        )

        # 7) Record metrics
        self.metrics.record_step(
            t=t,
            demand=demand,
            X=X,
            Y=Y,
            fleet=self.fleet,
            stations=self.stations,
            charged_by_station=charged_by_station,
            deadline_misses=deadline_misses,
            charge_power_kw=self.charge_power_kw,
            swap_arrivals=swap_arrivals,
            swap_requests=swap_requests,
            successful_swaps=successful_swaps,
            unmet_battery_demand=unmet_battery_demand,
            charging_demand=charging_demand,
            total_charging_demand_kw=total_charging_demand_kw,
            waiting_time_for_battery_slots=waiting_time_for_battery_slots,
            idle_driving_distance=idle_driving_distance,
            waiting_vehicles=waiting_vehicles,
            waiting_vehicles_by_station=waiting_by_station,
        )



def _state_transition(
    fleet: FleetState,
    S: np.ndarray,
    U: np.ndarray,
    H: np.ndarray,
    residual_vacant: np.ndarray,
    transition: TransitionProbabilities,
    energy_consumption: np.ndarray,
    t: int,
    rng: np.random.Generator,
) -> FleetState:
    m, levels_plus = fleet.vacant.shape
    levels = levels_plus - 1

    V_next = np.zeros_like(fleet.vacant)
    O_next = np.zeros_like(fleet.occupied)

    P = transition.P[t]
    Q = transition.Q[t]
    P_tilde = transition.P_tilde[t]
    Q_tilde = transition.Q_tilde[t]

    for i_prev in range(m):
        for l_prev in range(levels + 1):
            s_count = int(S[i_prev, l_prev])
            u_count = int(U[i_prev, l_prev])
            if s_count <= 0 and u_count <= 0:
                continue

            feasible = l_prev >= energy_consumption[i_prev]

            if s_count > 0:
                s_occ = P_tilde[i_prev].copy()
                s_vac = Q_tilde[i_prev].copy()
                s_occ[~feasible] = 0.0
                s_vac[~feasible] = 0.0
                _assign_transitions(
                    count=s_count,
                    origin=i_prev,
                    source_level=l_prev,
                    occ_probs=s_occ,
                    vac_probs=s_vac,
                    energy_consumption=energy_consumption,
                    O_next=O_next,
                    V_next=V_next,
                    rng=rng,
                    fallback_to_occupied=False,
                )

            if u_count > 0:
                u_occ = P[i_prev].copy()
                u_vac = Q[i_prev].copy()
                u_occ[~feasible] = 0.0
                u_vac[~feasible] = 0.0
                _assign_transitions(
                    count=u_count,
                    origin=i_prev,
                    source_level=l_prev,
                    occ_probs=u_occ,
                    vac_probs=u_vac,
                    energy_consumption=energy_consumption,
                    O_next=O_next,
                    V_next=V_next,
                    rng=rng,
                    fallback_to_occupied=True,
                )

    V_next += H + residual_vacant

    return FleetState(vacant=V_next, occupied=O_next)


def _assign_transitions(
    count: int,
    origin: int,
    source_level: int,
    occ_probs: np.ndarray,
    vac_probs: np.ndarray,
    energy_consumption: np.ndarray,
    O_next: np.ndarray,
    V_next: np.ndarray,
    rng: np.random.Generator,
    fallback_to_occupied: bool,
) -> None:
    m = occ_probs.shape[0]
    probs = np.concatenate([occ_probs, vac_probs], axis=0)
    total = float(probs.sum())
    if total <= 0:
        if fallback_to_occupied:
            O_next[origin, source_level] += count
        else:
            V_next[origin, source_level] += count
        return

    probs /= total
    draw = rng.multinomial(count, probs)
    occ_draw = draw[:m]
    vac_draw = draw[m:]

    for dst in range(m):
        e = int(energy_consumption[origin, dst])
        target_level = source_level - e
        if target_level < 0:
            continue
        if occ_draw[dst] > 0:
            O_next[dst, target_level] += int(occ_draw[dst])
        if vac_draw[dst] > 0:
            V_next[dst, target_level] += int(vac_draw[dst])
