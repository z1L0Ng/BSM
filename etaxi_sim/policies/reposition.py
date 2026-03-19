from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter
from typing import Dict, List, Tuple

import numpy as np

from etaxi_sim.data.preprocess import TransitionProbabilities
from etaxi_sim.models.fleet import FleetState

try:
    import gurobipy as gp
    from gurobipy import GRB

    HAS_GUROBI = True
except Exception:
    HAS_GUROBI = False

_GUROBI_ENV = None

_TRACE_DEFAULTS = {
    "t_start": -1,
    "status": None,
    "sol_count": None,
    "runtime_sec": None,
    "build_time_sec": None,
    "candidate_prep_time_sec": None,
    "vars_build_time_sec": None,
    "expressions_build_time_sec": None,
    "constraints_build_time_sec": None,
    "objective_build_time_sec": None,
    "optimize_time_sec": None,
    "wall_time_sec": None,
    "outcome": "unknown",
    "note": "",
    "num_vars": None,
    "num_constrs": None,
    "num_nz": None,
    "iter_count": None,
    "bar_iter_count": None,
    "obj_val": None,
    "obj_bound": None,
    "solver_method": None,
    "solver_crossover": None,
    "numeric_focus": None,
    "presolve": None,
    "lp_warm_start_enabled": None,
    "lp_warm_start_mode": None,
    "lp_warm_start_applied": None,
    "lp_warm_start_var_count": None,
    "lp_warm_start_nonzero_count": None,
    "lp_warm_start_note": "",
    "use_preaggregation": None,
    "use_aux_elimination": None,
    "x_var_count": 0,
    "y_var_count": 0,
    "v_var_count": 0,
    "o_var_count": 0,
    "r_var_count": 0,
    "b_var_count": 0,
    "mu_var_count": 0,
    "served_var_count": 0,
    "full_stock_var_count": 0,
    "vacant_split_constr_count": 0,
    "b_def_constr_count": 0,
    "mu_le_b_constr_count": 0,
    "swap_cap_constr_count": 0,
    "full_dyn_constr_count": 0,
    "served_by_supply_constr_count": 0,
    "o_dyn_constr_count": 0,
    "v_dyn_constr_count": 0,
}
_LAST_GUROBI_REPOSITION_TRACE = dict(_TRACE_DEFAULTS)
_GUROBI_REPOSITION_OUTCOME_COUNTS: Dict[str, int] = {}
_GUROBI_REPOSITION_STATUS_COUNTS: Dict[str, int] = {}


@dataclass(frozen=True)
class RepositionPolicyConfig:
    swap_low_energy_threshold: int
    planning_horizon_slots: int = 4
    top_demand_targets: int = 8
    top_swap_targets: int = 4
    idle_cost_weight: float = 0.05
    service_reward: float = 1.0
    low_energy_swap_bonus: float = 0.15
    transition_topk: int = 6
    time_limit_sec: float = 3.0
    solver_method: int = 2
    solver_crossover: int = 0
    numeric_focus: int = 0
    presolve: int = -1
    use_lp_primal_start: bool = True
    lp_warm_start_mode: int = 2
    eliminate_auxiliary_vars: bool = True
    preaggregate_transitions: bool = True
    allow_no_incumbent_retry: bool = True
    allow_gurobi_error_fallback: bool = False
    # DEBUG-ONLY: set True to fall back to heuristic when solver finds no incumbent.
    # Should remain False in production — fix solver root causes instead.
    allow_heuristic_fallback: bool = False


def _update_reposition_trace(
    *,
    t_start: int,
    status: int | None,
    sol_count: int | None,
    runtime_sec: float | None,
    outcome: str,
    note: str = "",
    extra: Dict[str, int | float | str | None] | None = None,
) -> None:
    _LAST_GUROBI_REPOSITION_TRACE["t_start"] = int(t_start)
    _LAST_GUROBI_REPOSITION_TRACE["status"] = None if status is None else int(status)
    _LAST_GUROBI_REPOSITION_TRACE["sol_count"] = None if sol_count is None else int(sol_count)
    _LAST_GUROBI_REPOSITION_TRACE["runtime_sec"] = (
        None if runtime_sec is None else float(runtime_sec)
    )
    _LAST_GUROBI_REPOSITION_TRACE["outcome"] = str(outcome)
    _LAST_GUROBI_REPOSITION_TRACE["note"] = str(note)
    if extra:
        _LAST_GUROBI_REPOSITION_TRACE.update(extra)
    if outcome:
        _GUROBI_REPOSITION_OUTCOME_COUNTS[outcome] = _GUROBI_REPOSITION_OUTCOME_COUNTS.get(outcome, 0) + 1
    if status is not None:
        key = str(int(status))
        _GUROBI_REPOSITION_STATUS_COUNTS[key] = _GUROBI_REPOSITION_STATUS_COUNTS.get(key, 0) + 1


def get_last_gurobi_reposition_trace() -> Dict[str, int | float | str | None]:
    return dict(_LAST_GUROBI_REPOSITION_TRACE)


def reset_gurobi_reposition_trace_stats() -> None:
    _LAST_GUROBI_REPOSITION_TRACE.clear()
    _LAST_GUROBI_REPOSITION_TRACE.update(_TRACE_DEFAULTS)
    _GUROBI_REPOSITION_OUTCOME_COUNTS.clear()
    _GUROBI_REPOSITION_STATUS_COUNTS.clear()


def _trace_extra_defaults() -> Dict[str, int | float | str | None]:
    base = dict(_TRACE_DEFAULTS)
    for key in ("t_start", "status", "sol_count", "runtime_sec", "outcome", "note"):
        base.pop(key, None)
    return base


def get_gurobi_reposition_trace_stats() -> Dict[str, Dict[str, int]]:
    return {
        "outcome_counts": dict(_GUROBI_REPOSITION_OUTCOME_COUNTS),
        "status_counts": dict(_GUROBI_REPOSITION_STATUS_COUNTS),
    }


def _build_model(name: str) -> gp.Model:
    global _GUROBI_ENV
    if _GUROBI_ENV is None:
        _GUROBI_ENV = gp.Env(empty=True)
        _GUROBI_ENV.setParam("OutputFlag", 0)
        _GUROBI_ENV.start()
    return gp.Model(name, env=_GUROBI_ENV)


def greedy_same_zone_policy(
    fleet: FleetState,
    demand: np.ndarray,
    config: RepositionPolicyConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simple baseline: keep vehicles in same zone.
    Low-energy vehicles are sent to swap, others serve demand in-place.
    """
    m, levels_plus = fleet.vacant.shape

    X = np.zeros((m, m, levels_plus), dtype=int)
    Y = np.zeros((m, m, levels_plus), dtype=int)

    for i in range(m):
        available = fleet.vacant[i].copy()
        # send low-energy to swap
        for l in range(levels_plus):
            if l <= config.swap_low_energy_threshold:
                count = int(available[l])
                if count > 0:
                    Y[i, i, l] = count
                    available[l] -= count

        # serve demand with remaining
        remaining_demand = int(demand[i])
        for l in range(levels_plus - 1, -1, -1):
            if remaining_demand <= 0:
                break
            count = int(available[l])
            if count <= 0:
                continue
            assign = min(count, remaining_demand)
            X[i, i, l] = assign
            available[l] -= assign
            remaining_demand -= assign

    return X, Y


def heuristic_battery_aware_policy(
    fleet: FleetState,
    demand: np.ndarray,
    station_full_batteries: np.ndarray,
    config: RepositionPolicyConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Heuristic baseline:
    - Low-energy vehicles only go to swap when local full-battery stock is available.
    - Remaining low-energy vehicles are delayed (kept idle in-zone).
    - Service is assigned in-zone using higher-energy vehicles first.
    """
    m, levels_plus = fleet.vacant.shape
    X = np.zeros((m, m, levels_plus), dtype=int)
    Y = np.zeros((m, m, levels_plus), dtype=int)

    for i in range(m):
        available = fleet.vacant[i].copy()
        full_budget = max(0, int(station_full_batteries[i]))

        for l in range(levels_plus):
            if l > config.swap_low_energy_threshold:
                continue
            count = int(available[l])
            if count <= 0:
                continue
            send_to_swap = min(count, full_budget)
            if send_to_swap > 0:
                Y[i, i, l] = send_to_swap
                available[l] -= send_to_swap
                full_budget -= send_to_swap

        remaining_demand = int(demand[i])
        for l in range(levels_plus - 1, config.swap_low_energy_threshold, -1):
            if remaining_demand <= 0:
                break
            count = int(available[l])
            if count <= 0:
                continue
            assign = min(count, remaining_demand)
            X[i, i, l] = assign
            available[l] -= assign
            remaining_demand -= assign

    return X, Y


def _topk_dests(row: np.ndarray, k: int) -> np.ndarray:
    if row.size <= k:
        idx = np.arange(row.size)
    else:
        idx = np.argpartition(row, -k)[-k:]
    idx = idx[np.argsort(row[idx])[::-1]]
    return idx


def _allocate_integer_dispatch(
    x_stage: Dict[Tuple[int, int, int], float],
    y_stage: Dict[Tuple[int, int, int], float],
    vacant: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    m, levels_plus = vacant.shape
    X = np.zeros((m, m, levels_plus), dtype=int)
    Y = np.zeros((m, m, levels_plus), dtype=int)

    fractional: Dict[Tuple[int, int], List[Tuple[float, str, Tuple[int, int, int]]]] = {}
    used = np.zeros_like(vacant, dtype=int)

    for key, val in x_stage.items():
        i, j, l = key
        base = int(np.floor(max(0.0, val)))
        X[i, j, l] = base
        used[i, l] += base
        frac = float(max(0.0, val - base))
        if frac > 0:
            fractional.setdefault((i, l), []).append((frac, "x", key))

    for key, val in y_stage.items():
        i, j, l = key
        base = int(np.floor(max(0.0, val)))
        Y[i, j, l] = base
        used[i, l] += base
        frac = float(max(0.0, val - base))
        if frac > 0:
            fractional.setdefault((i, l), []).append((frac, "y", key))

    for i in range(m):
        for l in range(levels_plus):
            cap = int(vacant[i, l])
            if used[i, l] > cap:
                overflow = used[i, l] - cap
                if overflow > 0:
                    # Remove from Y first, then X to prioritize service.
                    for j in range(m):
                        if overflow <= 0:
                            break
                        take = min(overflow, Y[i, j, l])
                        Y[i, j, l] -= take
                        used[i, l] -= take
                        overflow -= take
                    for j in range(m):
                        if overflow <= 0:
                            break
                        take = min(overflow, X[i, j, l])
                        X[i, j, l] -= take
                        used[i, l] -= take
                        overflow -= take

            remain = cap - used[i, l]
            if remain <= 0:
                continue
            cand = fractional.get((i, l), [])
            if not cand:
                continue
            cand.sort(key=lambda x: x[0], reverse=True)
            for _, typ, key in cand:
                if remain <= 0:
                    break
                _, j, _ = key
                if typ == "x":
                    X[i, j, l] += 1
                else:
                    Y[i, j, l] += 1
                used[i, l] += 1
                remain -= 1

    return X, Y


def _set_lp_primal_start(var: gp.Var, value: float) -> bool:
    try:
        var.PStart = float(value)
        return True
    except Exception:
        # Fallback for environments where PStart is unavailable.
        try:
            var.Start = float(value)
            return True
        except Exception:
            return False


def _apply_zero_dispatch_lp_primal_start(
    *,
    x_vars: Dict[Tuple[int, int, int, int], gp.Var],
    y_vars: Dict[Tuple[int, int, int, int], gp.Var],
    r_vars: Dict[Tuple[int, int, int], gp.Var],
    b_vars: Dict[Tuple[int, int, int], gp.Var],
    mu_vars: Dict[Tuple[int, int, int], gp.Var],
    served: Dict[Tuple[int, int], gp.Var],
    full_stock: Dict[Tuple[int, int], gp.Var],
    station_full_batteries: np.ndarray,
) -> tuple[int, int]:
    set_count = 0
    nonzero_count = 0

    zero_blocks = (x_vars, y_vars, r_vars, b_vars, mu_vars, served)
    for block in zero_blocks:
        for var in block.values():
            if _set_lp_primal_start(var, 0.0):
                set_count += 1

    for (_, j), var in full_stock.items():
        value = float(station_full_batteries[j])
        if _set_lp_primal_start(var, value):
            set_count += 1
            if abs(value) > 1e-12:
                nonzero_count += 1

    return set_count, nonzero_count


def gurobi_reposition_policy(
    fleet: FleetState,
    demand_window: np.ndarray,
    reachability: np.ndarray,
    energy_consumption: np.ndarray,
    station_full_batteries: np.ndarray,
    station_swapping_capacity: np.ndarray,
    transition: TransitionProbabilities,
    t_start: int,
    config: RepositionPolicyConfig,
    _retry_stage: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    wall_start = perf_counter()

    if not HAS_GUROBI:
        extra = _trace_extra_defaults()
        extra["wall_time_sec"] = float(perf_counter() - wall_start)
        _update_reposition_trace(
            t_start=t_start,
            status=None,
            sol_count=None,
            runtime_sec=None,
            outcome="no_gurobi",
            note="gurobipy unavailable",
            extra=extra,
        )
        if demand_window.ndim == 1:
            return greedy_same_zone_policy(fleet, demand_window, config)
        return greedy_same_zone_policy(fleet, demand_window[0], config)

    if demand_window.ndim == 1:
        demand_window = demand_window[None, :]

    m, levels_plus = fleet.vacant.shape
    levels = levels_plus - 1
    X = np.zeros((m, m, levels_plus), dtype=int)
    Y = np.zeros((m, m, levels_plus), dtype=int)
    H = int(max(1, min(config.planning_horizon_slots, demand_window.shape[0])))
    demand_window = demand_window[:H]
    use_aux_elimination = bool(config.eliminate_auxiliary_vars)
    use_preaggregation = bool(config.preaggregate_transitions)

    if fleet.vacant.sum() <= 0:
        extra = _trace_extra_defaults()
        extra["wall_time_sec"] = float(perf_counter() - wall_start)
        _update_reposition_trace(
            t_start=t_start,
            status=None,
            sol_count=None,
            runtime_sec=None,
            outcome="skip_no_vacant",
            note="no vacant vehicles",
            extra=extra,
        )
        return X, Y

    all_states = [(i, l) for i in range(m) for l in range(levels_plus)]

    # Candidate dispatch destinations by state/time.
    candidate_prep_start = perf_counter()
    top_demand_by_k = [
        np.argsort(-demand_window[k])[: max(1, config.top_demand_targets)] for k in range(H)
    ]
    demand_rank_by_k = [np.argsort(-demand_window[k]) for k in range(H)]
    swap_priority = np.argsort(-station_full_batteries)
    feasible_by_state: Dict[Tuple[int, int], np.ndarray] = {}
    feasible_set_by_state: Dict[Tuple[int, int], set[int]] = {}
    swap_targets_by_state: Dict[Tuple[int, int], List[int]] = {}
    for i in range(m):
        reachable_targets = np.flatnonzero(reachability[i] == 0)
        if reachable_targets.size == 0:
            for l in range(levels_plus):
                feasible_by_state[(i, l)] = np.empty(0, dtype=int)
                feasible_set_by_state[(i, l)] = set()
                swap_targets_by_state[(i, l)] = []
            continue
        energy_row = energy_consumption[i, reachable_targets]
        for l in range(levels_plus):
            feasible = reachable_targets[energy_row <= l]
            feasible_by_state[(i, l)] = feasible
            feasible_set = set(int(v_) for v_ in feasible.tolist())
            feasible_set_by_state[(i, l)] = feasible_set
            swap_targets = [int(j) for j in swap_priority if int(j) in feasible_set][: config.top_swap_targets]
            if i in feasible_set and i not in swap_targets:
                swap_targets = [i] + swap_targets
                swap_targets = swap_targets[: config.top_swap_targets]
            swap_targets_by_state[(i, l)] = swap_targets

    # Sparse transition support to keep MPC tractable.
    trans_k = max(1, min(config.transition_topk, m))
    trans_targets: Dict[Tuple[int, int], np.ndarray] = {}
    incoming_prev_by_target: Dict[Tuple[int, int], List[int]] = {(k, i): [] for k in range(H) for i in range(m)}
    for k in range(H):
        t_idx = min(t_start + k, transition.P.shape[0] - 1)
        for i_prev in range(m):
            row = transition.P[t_idx, i_prev] + transition.Q[t_idx, i_prev]
            targets = _topk_dests(row, trans_k)
            trans_targets[(k, i_prev)] = targets
            for i in targets:
                incoming_prev_by_target[(k, int(i))].append(i_prev)
    candidate_prep_time_sec = float(perf_counter() - candidate_prep_start)

    try:
        build_start = perf_counter()
        trace_extra = _trace_extra_defaults()
        trace_extra["candidate_prep_time_sec"] = candidate_prep_time_sec

        model = _build_model("fleet_reposition_mpc_vo")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = config.time_limit_sec
        model.Params.Method = int(config.solver_method)
        model.Params.Crossover = int(config.solver_crossover)
        model.Params.NumericFocus = int(config.numeric_focus)
        model.Params.Presolve = int(config.presolve)
        if bool(config.use_lp_primal_start):
            model.Params.LPWarmStart = int(config.lp_warm_start_mode)

        trace_extra.update(
            {
                "solver_method": int(config.solver_method),
                "solver_crossover": int(config.solver_crossover),
                "numeric_focus": int(config.numeric_focus),
                "presolve": int(config.presolve),
                "lp_warm_start_enabled": int(bool(config.use_lp_primal_start)),
                "lp_warm_start_mode": int(config.lp_warm_start_mode),
            }
        )

        vars_build_start = perf_counter()
        # State expressions (equivalent elimination of v/o auxiliary variables).
        v_state: Dict[Tuple[int, int, int], gp.LinExpr | float] = {}
        o_state: Dict[Tuple[int, int, int], gp.LinExpr | float] = {}
        for i in range(m):
            for l in range(levels_plus):
                v_state[(0, i, l)] = float(fleet.vacant[i, l])
                o_state[(0, i, l)] = float(fleet.occupied[i, l])

        # Dispatch/processing variables
        x_vars: Dict[Tuple[int, int, int, int], gp.Var] = {}
        y_vars: Dict[Tuple[int, int, int, int], gp.Var] = {}
        r_vars: Dict[Tuple[int, int, int], gp.Var] = {}
        b_vars: Dict[Tuple[int, int, int], gp.Var] = {}
        mu_vars: Dict[Tuple[int, int, int], gp.Var] = {}

        outbound: Dict[Tuple[int, int, int], List[gp.Var]] = {}
        service_inbound: Dict[Tuple[int, int], List[gp.Var]] = {}
        service_state_inbound: Dict[Tuple[int, int, int], List[gp.Var]] = {}
        swap_inbound: Dict[Tuple[int, int, int], List[gp.Var]] = {}
        low_energy_y: List[gp.Var] = []
        move_cost_terms: List[gp.LinExpr] = []

        for k in range(H):
            top_demand = top_demand_by_k[k]
            demand_rank = demand_rank_by_k[k]
            for i, l in all_states:
                state_key = (k, i, l)
                state_out: List[gp.Var] = []
                if not use_aux_elimination:
                    r_vars[state_key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"r_{k}_{i}_{l}")
                    state_out.append(r_vars[state_key])
                outbound[state_key] = state_out

                feasible = feasible_by_state[(i, l)]
                if feasible.size == 0:
                    continue
                feasible_set = feasible_set_by_state[(i, l)]

                service_targets = [int(j) for j in top_demand if int(j) in feasible_set]
                if i in feasible_set and i not in service_targets:
                    service_targets.append(i)
                if len(service_targets) < config.top_demand_targets:
                    for j in demand_rank:
                        j_idx = int(j)
                        if j_idx in feasible_set and j_idx not in service_targets:
                            service_targets.append(j_idx)
                            if len(service_targets) >= config.top_demand_targets:
                                break

                swap_targets = swap_targets_by_state[(i, l)]

                if l > config.swap_low_energy_threshold:
                    for j in service_targets:
                        key = (k, i, j, l)
                        if key in x_vars:
                            continue
                        var = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"x_{k}_{i}_{j}_{l}")
                        x_vars[key] = var
                        state_out.append(var)
                        service_inbound.setdefault((k, j), []).append(var)
                        service_state_inbound.setdefault((k, j, l), []).append(var)
                        move_cost_terms.append(float(energy_consumption[i, j]) * var)

                for j in swap_targets:
                    key = (k, i, j, l)
                    if key in y_vars:
                        continue
                    var = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"y_{k}_{i}_{j}_{l}")
                    y_vars[key] = var
                    state_out.append(var)
                    swap_inbound.setdefault((k, j, l), []).append(var)
                    move_cost_terms.append(float(energy_consumption[i, j]) * var)
                    if l <= config.swap_low_energy_threshold:
                        low_energy_y.append(var)

        if not x_vars and not y_vars:
            trace_extra.update(
                {
                    "build_time_sec": float(perf_counter() - build_start),
                    "wall_time_sec": float(perf_counter() - wall_start),
                    "v_var_count": 0,
                    "o_var_count": 0,
                    "r_var_count": len(r_vars),
                }
            )
            _update_reposition_trace(
                t_start=t_start,
                status=None,
                sol_count=None,
                runtime_sec=None,
                outcome="skip_no_candidates",
                note="no feasible dispatch vars",
                extra=trace_extra,
            )
            return X, Y

        # Swap station battery stocks and swap execution
        full_stock: Dict[Tuple[int, int], gp.Var] = {}
        for k in range(H + 1):
            for j in range(m):
                if k == 0:
                    full_stock[(k, j)] = model.addVar(
                        vtype=GRB.CONTINUOUS,
                        lb=float(station_full_batteries[j]),
                        ub=float(station_full_batteries[j]),
                        name=f"full_{k}_{j}",
                    )
                else:
                    full_stock[(k, j)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"full_{k}_{j}")

        for k in range(H):
            for j in range(m):
                for l in range(levels_plus):
                    if not use_aux_elimination:
                        b_vars[(k, j, l)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"b_{k}_{j}_{l}")
                    mu_vars[(k, j, l)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"mu_{k}_{j}_{l}")

        served = {
            (k, j): model.addVar(
                vtype=GRB.CONTINUOUS,
                lb=0.0,
                ub=float(demand_window[k, j]),
                name=f"served_{k}_{j}",
            )
            for k in range(H)
            for j in range(m)
        }
        vars_build_time_sec = float(perf_counter() - vars_build_start)

        expr_build_start = perf_counter()
        # Pre-aggregate service flow by destination and level to avoid rebuilding
        # identical quicksum expressions in every transition constraint.
        service_state_expr: Dict[Tuple[int, int, int], gp.LinExpr] = {}
        if use_preaggregation:
            service_state_expr = {
                key: gp.quicksum(vars_) for key, vars_ in service_state_inbound.items()
            }
        swap_inbound_expr: Dict[Tuple[int, int, int], gp.LinExpr] = {
            key: gp.quicksum(vars_) for key, vars_ in swap_inbound.items()
        }
        outbound_expr: Dict[Tuple[int, int, int], gp.LinExpr] = {
            key: gp.quicksum(vars_) for key, vars_ in outbound.items()
        }

        constr_counts: Dict[str, int] = {
            "vacant_split_constr_count": 0,
            "b_def_constr_count": 0,
            "mu_le_b_constr_count": 0,
            "swap_cap_constr_count": 0,
            "full_dyn_constr_count": 0,
            "served_by_supply_constr_count": 0,
            "o_dyn_constr_count": 0,
            "v_dyn_constr_count": 0,
        }

        # Recursively build expected state expressions over the full horizon.
        for k in range(H):
            t_idx = min(t_start + k, transition.P.shape[0] - 1)
            next_v_state: Dict[Tuple[int, int], gp.LinExpr] = {}
            next_o_state: Dict[Tuple[int, int], gp.LinExpr] = {}
            for i in range(m):
                for l in range(levels_plus):
                    o_terms: List[gp.LinExpr] = []
                    if use_aux_elimination:
                        v_terms: List[gp.LinExpr] = [v_state[(k, i, l)] - outbound_expr[(k, i, l)]]
                    else:
                        v_terms = [r_vars[(k, i, l)]]
                    if l == levels:
                        full_arrivals_expr: gp.LinExpr | float
                        if use_aux_elimination:
                            full_arrivals_expr = swap_inbound_expr.get((k, i, levels), 0.0)
                        else:
                            full_arrivals_expr = b_vars[(k, i, levels)]
                        v_terms.append(
                            gp.quicksum(mu_vars[(k, i, ll)] for ll in range(levels_plus))
                            + full_arrivals_expr
                        )
                    if use_preaggregation:
                        prev_candidates = incoming_prev_by_target[(k, i)]
                    else:
                        prev_candidates = range(m)
                    for i_prev in prev_candidates:
                        if not use_preaggregation and i not in trans_targets[(k, i_prev)]:
                            continue
                        e = int(energy_consumption[i_prev, i])
                        l_prev = l + e
                        if l_prev > levels:
                            continue
                        if use_preaggregation:
                            s_prev = service_state_expr.get((k, i_prev, l_prev), 0.0)
                        else:
                            s_prev = gp.quicksum(
                                x_vars.get((k, origin, i_prev, l_prev), 0.0) for origin in range(m)
                            )
                        ptil = float(transition.P_tilde[t_idx, i_prev, i])
                        qtil = float(transition.Q_tilde[t_idx, i_prev, i])
                        pocc = float(transition.P[t_idx, i_prev, i])
                        qocc = float(transition.Q[t_idx, i_prev, i])
                        if ptil != 0.0:
                            o_terms.append(ptil * s_prev)
                        if pocc != 0.0:
                            o_terms.append(pocc * o_state[(k, i_prev, l_prev)])
                        if qtil != 0.0:
                            v_terms.append(qtil * s_prev)
                        if qocc != 0.0:
                            v_terms.append(qocc * o_state[(k, i_prev, l_prev)])
                    next_o_state[(i, l)] = gp.quicksum(o_terms)
                    next_v_state[(i, l)] = gp.quicksum(v_terms)
            for i in range(m):
                for l in range(levels_plus):
                    o_state[(k + 1, i, l)] = next_o_state[(i, l)]
                    v_state[(k + 1, i, l)] = next_v_state[(i, l)]
        expressions_build_time_sec = float(perf_counter() - expr_build_start)

        constraints_build_start = perf_counter()
        # Dispatch feasibility from vacant states.
        for k in range(H):
            for i in range(m):
                for l in range(levels_plus):
                    if use_aux_elimination:
                        model.addConstr(
                            outbound_expr[(k, i, l)] <= v_state[(k, i, l)],
                            name=f"vacant_split_{k}_{i}_{l}",
                        )
                    else:
                        model.addConstr(
                            outbound_expr[(k, i, l)] == v_state[(k, i, l)],
                            name=f"vacant_split_{k}_{i}_{l}",
                        )
                    constr_counts["vacant_split_constr_count"] += 1

        # Swap equations and stock dynamics.
        for k in range(H):
            for j in range(m):
                for l in range(levels_plus):
                    if use_aux_elimination:
                        model.addConstr(
                            mu_vars[(k, j, l)] <= swap_inbound_expr.get((k, j, l), 0.0),
                            name=f"mu_le_b_{k}_{j}_{l}",
                        )
                    else:
                        model.addConstr(
                            b_vars[(k, j, l)] == swap_inbound_expr.get((k, j, l), 0.0),
                            name=f"b_def_{k}_{j}_{l}",
                        )
                        constr_counts["b_def_constr_count"] += 1
                        model.addConstr(
                            mu_vars[(k, j, l)] <= b_vars[(k, j, l)],
                            name=f"mu_le_b_{k}_{j}_{l}",
                        )
                    constr_counts["mu_le_b_constr_count"] += 1

                mu_sum = gp.quicksum(mu_vars[(k, j, l)] for l in range(levels_plus))
                model.addConstr(
                    mu_sum <= float(station_swapping_capacity[j]),
                    name=f"swap_machine_cap_{k}_{j}",
                )
                constr_counts["swap_cap_constr_count"] += 1
                model.addConstr(
                    mu_sum <= full_stock[(k, j)],
                    name=f"swap_full_cap_{k}_{j}",
                )
                constr_counts["swap_cap_constr_count"] += 1
                model.addConstr(
                    full_stock[(k + 1, j)]
                    == full_stock[(k, j)] - mu_sum,
                    name=f"full_dyn_{k}_{j}",
                )
                constr_counts["full_dyn_constr_count"] += 1

                model.addConstr(
                    served[(k, j)] <= gp.quicksum(service_inbound.get((k, j), [])),
                    name=f"served_by_supply_{k}_{j}",
                )
                constr_counts["served_by_supply_constr_count"] += 1
        constraints_build_time_sec = float(perf_counter() - constraints_build_start)

        objective_build_start = perf_counter()
        objective = (
            config.service_reward
            * gp.quicksum((0.95**k) * served[(k, j)] for k in range(H) for j in range(m))
            - config.idle_cost_weight * gp.quicksum(move_cost_terms)
            + config.low_energy_swap_bonus * gp.quicksum(low_energy_y)
        )
        model.setObjective(objective, GRB.MAXIMIZE)
        objective_build_time_sec = float(perf_counter() - objective_build_start)
        model.update()

        lp_warm_start_applied = 0
        lp_warm_start_var_count = 0
        lp_warm_start_nonzero_count = 0
        lp_warm_start_note = ""
        if bool(config.use_lp_primal_start):
            try:
                lp_warm_start_var_count, lp_warm_start_nonzero_count = _apply_zero_dispatch_lp_primal_start(
                    x_vars=x_vars,
                    y_vars=y_vars,
                    r_vars=r_vars,
                    b_vars=b_vars,
                    mu_vars=mu_vars,
                    served=served,
                    full_stock=full_stock,
                    station_full_batteries=station_full_batteries,
                )
                lp_warm_start_applied = int(lp_warm_start_var_count > 0)
            except Exception as exc:
                lp_warm_start_note = f"{type(exc).__name__}: {exc}"

        build_time_sec = float(perf_counter() - build_start)
        num_nz: int | None
        try:
            num_nz = int(model.NumNZs)
        except Exception:
            num_nz = None
        trace_extra.update(
            {
                "build_time_sec": build_time_sec,
                "vars_build_time_sec": vars_build_time_sec,
                "expressions_build_time_sec": expressions_build_time_sec,
                "constraints_build_time_sec": constraints_build_time_sec,
                "objective_build_time_sec": objective_build_time_sec,
                "num_vars": int(model.NumVars),
                "num_constrs": int(model.NumConstrs),
                "num_nz": num_nz,
                "lp_warm_start_applied": int(lp_warm_start_applied),
                "lp_warm_start_var_count": int(lp_warm_start_var_count),
                "lp_warm_start_nonzero_count": int(lp_warm_start_nonzero_count),
                "lp_warm_start_note": lp_warm_start_note,
                "use_preaggregation": int(use_preaggregation),
                "use_aux_elimination": int(use_aux_elimination),
                "x_var_count": len(x_vars),
                "y_var_count": len(y_vars),
                "v_var_count": 0,
                "o_var_count": 0,
                "r_var_count": len(r_vars),
                "b_var_count": len(b_vars),
                "mu_var_count": len(mu_vars),
                "served_var_count": len(served),
                "full_stock_var_count": len(full_stock),
            }
        )
        trace_extra.update(constr_counts)

        optimize_start = perf_counter()
        model.optimize()
        optimize_time_sec = float(perf_counter() - optimize_start)

        status_code = int(model.Status)
        sol_count = int(model.SolCount)
        iter_count: float | None
        bar_iter_count: int | None
        obj_val: float | None
        obj_bound: float | None
        try:
            iter_count = float(model.IterCount)
        except Exception:
            iter_count = None
        try:
            bar_iter_count = int(model.BarIterCount)
        except Exception:
            bar_iter_count = None
        try:
            obj_bound = float(model.ObjBound)
        except Exception:
            obj_bound = None
        if sol_count > 0:
            try:
                obj_val = float(model.ObjVal)
            except Exception:
                obj_val = None
        else:
            obj_val = None
        trace_extra["optimize_time_sec"] = optimize_time_sec
        trace_extra["wall_time_sec"] = float(perf_counter() - wall_start)
        trace_extra["iter_count"] = iter_count
        trace_extra["bar_iter_count"] = bar_iter_count
        trace_extra["obj_val"] = obj_val
        trace_extra["obj_bound"] = obj_bound
        _update_reposition_trace(
            t_start=t_start,
            status=status_code,
            sol_count=sol_count,
            runtime_sec=float(model.Runtime),
            outcome="optimized" if sol_count > 0 else "no_incumbent",
            note="",
            extra=trace_extra,
        )

        if model.SolCount <= 0:
            if config.allow_no_incumbent_retry:
                if _retry_stage == 0 and H > 1:
                    _update_reposition_trace(
                        t_start=t_start,
                        status=status_code,
                        sol_count=sol_count,
                        runtime_sec=float(model.Runtime),
                        outcome="retry_h1_no_incumbent",
                        note=f"h={H}, tl={config.time_limit_sec}",
                        extra=trace_extra,
                    )
                    retry_cfg = replace(config, planning_horizon_slots=1)
                    return gurobi_reposition_policy(
                        fleet=fleet,
                        demand_window=demand_window[:1],
                        reachability=reachability,
                        energy_consumption=energy_consumption,
                        station_full_batteries=station_full_batteries,
                        station_swapping_capacity=station_swapping_capacity,
                        transition=transition,
                        t_start=t_start,
                        config=retry_cfg,
                        _retry_stage=1,
                    )
                if _retry_stage <= 1 and float(config.time_limit_sec) < 20.0:
                    _update_reposition_trace(
                        t_start=t_start,
                        status=status_code,
                        sol_count=sol_count,
                        runtime_sec=float(model.Runtime),
                        outcome="retry_t20_no_incumbent",
                        note=f"h={H}, tl={config.time_limit_sec}",
                        extra=trace_extra,
                    )
                    retry_cfg = replace(config, planning_horizon_slots=1, time_limit_sec=20.0)
                    return gurobi_reposition_policy(
                        fleet=fleet,
                        demand_window=demand_window[:1],
                        reachability=reachability,
                        energy_consumption=energy_consumption,
                        station_full_batteries=station_full_batteries,
                        station_swapping_capacity=station_swapping_capacity,
                        transition=transition,
                        t_start=t_start,
                        config=retry_cfg,
                        _retry_stage=2,
                    )
            if config.allow_heuristic_fallback:
                _update_reposition_trace(
                    t_start=t_start,
                    status=status_code,
                    sol_count=0,
                    runtime_sec=float(model.Runtime),
                    outcome="fallback_heuristic_no_incumbent",
                    note=f"h={H}, tl={config.time_limit_sec}, retry_stage={_retry_stage}",
                    extra=trace_extra,
                )
                return heuristic_battery_aware_policy(
                    fleet=fleet,
                    demand=demand_window[0],
                    station_full_batteries=station_full_batteries,
                    config=config,
                )
            raise RuntimeError(f"reposition model has no solution, status={model.Status}")

        x_stage: Dict[Tuple[int, int, int], float] = {}
        y_stage: Dict[Tuple[int, int, int], float] = {}
        for (k, i, j, l), var in x_vars.items():
            if k != 0:
                continue
            if var.X > 1e-8:
                x_stage[(i, j, l)] = float(var.X)
        for (k, i, j, l), var in y_vars.items():
            if k != 0:
                continue
            if var.X > 1e-8:
                y_stage[(i, j, l)] = float(var.X)

        return _allocate_integer_dispatch(x_stage, y_stage, fleet.vacant)
    except gp.GurobiError as exc:
        extra = _trace_extra_defaults()
        extra["wall_time_sec"] = float(perf_counter() - wall_start)
        outcome = "gurobi_error"
        note = str(exc)
        if config.allow_gurobi_error_fallback:
            outcome = "fallback_greedy_gurobi_error"
        _update_reposition_trace(
            t_start=t_start,
            status=None,
            sol_count=None,
            runtime_sec=None,
            outcome=outcome,
            note=note,
            extra=extra,
        )
        if config.allow_gurobi_error_fallback:
            return greedy_same_zone_policy(fleet, demand_window[0], config)
        raise RuntimeError(f"gurobi_reposition_policy failed: {exc}") from exc
    except Exception as exc:
        prev_note = _LAST_GUROBI_REPOSITION_TRACE.get("note", "")
        extra = _trace_extra_defaults()
        extra.update(
            {
                "build_time_sec": _LAST_GUROBI_REPOSITION_TRACE.get("build_time_sec"),  # type: ignore[dict-item]
                "candidate_prep_time_sec": _LAST_GUROBI_REPOSITION_TRACE.get("candidate_prep_time_sec"),  # type: ignore[dict-item]
                "vars_build_time_sec": _LAST_GUROBI_REPOSITION_TRACE.get("vars_build_time_sec"),  # type: ignore[dict-item]
                "expressions_build_time_sec": _LAST_GUROBI_REPOSITION_TRACE.get("expressions_build_time_sec"),  # type: ignore[dict-item]
                "constraints_build_time_sec": _LAST_GUROBI_REPOSITION_TRACE.get("constraints_build_time_sec"),  # type: ignore[dict-item]
                "objective_build_time_sec": _LAST_GUROBI_REPOSITION_TRACE.get("objective_build_time_sec"),  # type: ignore[dict-item]
                "optimize_time_sec": _LAST_GUROBI_REPOSITION_TRACE.get("optimize_time_sec"),  # type: ignore[dict-item]
                "wall_time_sec": float(perf_counter() - wall_start),
                "num_vars": _LAST_GUROBI_REPOSITION_TRACE.get("num_vars"),  # type: ignore[dict-item]
                "num_constrs": _LAST_GUROBI_REPOSITION_TRACE.get("num_constrs"),  # type: ignore[dict-item]
                "num_nz": _LAST_GUROBI_REPOSITION_TRACE.get("num_nz"),  # type: ignore[dict-item]
                "iter_count": _LAST_GUROBI_REPOSITION_TRACE.get("iter_count"),  # type: ignore[dict-item]
                "bar_iter_count": _LAST_GUROBI_REPOSITION_TRACE.get("bar_iter_count"),  # type: ignore[dict-item]
                "obj_val": _LAST_GUROBI_REPOSITION_TRACE.get("obj_val"),  # type: ignore[dict-item]
                "obj_bound": _LAST_GUROBI_REPOSITION_TRACE.get("obj_bound"),  # type: ignore[dict-item]
                "solver_method": _LAST_GUROBI_REPOSITION_TRACE.get("solver_method"),  # type: ignore[dict-item]
                "solver_crossover": _LAST_GUROBI_REPOSITION_TRACE.get("solver_crossover"),  # type: ignore[dict-item]
                "numeric_focus": _LAST_GUROBI_REPOSITION_TRACE.get("numeric_focus"),  # type: ignore[dict-item]
                "presolve": _LAST_GUROBI_REPOSITION_TRACE.get("presolve"),  # type: ignore[dict-item]
                "lp_warm_start_enabled": _LAST_GUROBI_REPOSITION_TRACE.get("lp_warm_start_enabled"),  # type: ignore[dict-item]
                "lp_warm_start_mode": _LAST_GUROBI_REPOSITION_TRACE.get("lp_warm_start_mode"),  # type: ignore[dict-item]
                "lp_warm_start_applied": _LAST_GUROBI_REPOSITION_TRACE.get("lp_warm_start_applied"),  # type: ignore[dict-item]
                "lp_warm_start_var_count": _LAST_GUROBI_REPOSITION_TRACE.get("lp_warm_start_var_count"),  # type: ignore[dict-item]
                "lp_warm_start_nonzero_count": _LAST_GUROBI_REPOSITION_TRACE.get("lp_warm_start_nonzero_count"),  # type: ignore[dict-item]
                "lp_warm_start_note": _LAST_GUROBI_REPOSITION_TRACE.get("lp_warm_start_note"),  # type: ignore[dict-item]
                "use_preaggregation": _LAST_GUROBI_REPOSITION_TRACE.get("use_preaggregation"),  # type: ignore[dict-item]
                "use_aux_elimination": _LAST_GUROBI_REPOSITION_TRACE.get("use_aux_elimination"),  # type: ignore[dict-item]
                "x_var_count": _LAST_GUROBI_REPOSITION_TRACE.get("x_var_count"),  # type: ignore[dict-item]
                "y_var_count": _LAST_GUROBI_REPOSITION_TRACE.get("y_var_count"),  # type: ignore[dict-item]
                "v_var_count": _LAST_GUROBI_REPOSITION_TRACE.get("v_var_count"),  # type: ignore[dict-item]
                "o_var_count": _LAST_GUROBI_REPOSITION_TRACE.get("o_var_count"),  # type: ignore[dict-item]
                "r_var_count": _LAST_GUROBI_REPOSITION_TRACE.get("r_var_count"),  # type: ignore[dict-item]
                "b_var_count": _LAST_GUROBI_REPOSITION_TRACE.get("b_var_count"),  # type: ignore[dict-item]
                "mu_var_count": _LAST_GUROBI_REPOSITION_TRACE.get("mu_var_count"),  # type: ignore[dict-item]
                "served_var_count": _LAST_GUROBI_REPOSITION_TRACE.get("served_var_count"),  # type: ignore[dict-item]
                "full_stock_var_count": _LAST_GUROBI_REPOSITION_TRACE.get("full_stock_var_count"),  # type: ignore[dict-item]
                "vacant_split_constr_count": _LAST_GUROBI_REPOSITION_TRACE.get("vacant_split_constr_count"),  # type: ignore[dict-item]
                "b_def_constr_count": _LAST_GUROBI_REPOSITION_TRACE.get("b_def_constr_count"),  # type: ignore[dict-item]
                "mu_le_b_constr_count": _LAST_GUROBI_REPOSITION_TRACE.get("mu_le_b_constr_count"),  # type: ignore[dict-item]
                "swap_cap_constr_count": _LAST_GUROBI_REPOSITION_TRACE.get("swap_cap_constr_count"),  # type: ignore[dict-item]
                "full_dyn_constr_count": _LAST_GUROBI_REPOSITION_TRACE.get("full_dyn_constr_count"),  # type: ignore[dict-item]
                "served_by_supply_constr_count": _LAST_GUROBI_REPOSITION_TRACE.get("served_by_supply_constr_count"),  # type: ignore[dict-item]
                "o_dyn_constr_count": _LAST_GUROBI_REPOSITION_TRACE.get("o_dyn_constr_count"),  # type: ignore[dict-item]
                "v_dyn_constr_count": _LAST_GUROBI_REPOSITION_TRACE.get("v_dyn_constr_count"),  # type: ignore[dict-item]
            }
        )
        _update_reposition_trace(
            t_start=t_start,
            status=_LAST_GUROBI_REPOSITION_TRACE.get("status"),  # type: ignore[arg-type]
            sol_count=_LAST_GUROBI_REPOSITION_TRACE.get("sol_count"),  # type: ignore[arg-type]
            runtime_sec=_LAST_GUROBI_REPOSITION_TRACE.get("runtime_sec"),  # type: ignore[arg-type]
            outcome="exception",
            note=f"retry_stage={_retry_stage}; {prev_note}; {type(exc).__name__}: {exc}".strip("; "),
            extra=extra,
        )
        raise RuntimeError(f"gurobi_reposition_policy failed: {exc}") from exc
