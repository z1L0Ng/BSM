from __future__ import annotations

from dataclasses import dataclass, replace
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
_LAST_GUROBI_REPOSITION_TRACE = {
    "t_start": -1,
    "status": None,
    "sol_count": None,
    "runtime_sec": None,
    "outcome": "unknown",
    "note": "",
}
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


def _update_reposition_trace(
    *,
    t_start: int,
    status: int | None,
    sol_count: int | None,
    runtime_sec: float | None,
    outcome: str,
    note: str = "",
) -> None:
    _LAST_GUROBI_REPOSITION_TRACE["t_start"] = int(t_start)
    _LAST_GUROBI_REPOSITION_TRACE["status"] = None if status is None else int(status)
    _LAST_GUROBI_REPOSITION_TRACE["sol_count"] = None if sol_count is None else int(sol_count)
    _LAST_GUROBI_REPOSITION_TRACE["runtime_sec"] = (
        None if runtime_sec is None else float(runtime_sec)
    )
    _LAST_GUROBI_REPOSITION_TRACE["outcome"] = str(outcome)
    _LAST_GUROBI_REPOSITION_TRACE["note"] = str(note)
    if outcome:
        _GUROBI_REPOSITION_OUTCOME_COUNTS[outcome] = _GUROBI_REPOSITION_OUTCOME_COUNTS.get(outcome, 0) + 1
    if status is not None:
        key = str(int(status))
        _GUROBI_REPOSITION_STATUS_COUNTS[key] = _GUROBI_REPOSITION_STATUS_COUNTS.get(key, 0) + 1


def get_last_gurobi_reposition_trace() -> Dict[str, int | float | str | None]:
    return dict(_LAST_GUROBI_REPOSITION_TRACE)


def reset_gurobi_reposition_trace_stats() -> None:
    _LAST_GUROBI_REPOSITION_TRACE["t_start"] = -1
    _LAST_GUROBI_REPOSITION_TRACE["status"] = None
    _LAST_GUROBI_REPOSITION_TRACE["sol_count"] = None
    _LAST_GUROBI_REPOSITION_TRACE["runtime_sec"] = None
    _LAST_GUROBI_REPOSITION_TRACE["outcome"] = "unknown"
    _LAST_GUROBI_REPOSITION_TRACE["note"] = ""
    _GUROBI_REPOSITION_OUTCOME_COUNTS.clear()
    _GUROBI_REPOSITION_STATUS_COUNTS.clear()


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
    if not HAS_GUROBI:
        _update_reposition_trace(
            t_start=t_start,
            status=None,
            sol_count=None,
            runtime_sec=None,
            outcome="no_gurobi",
            note="gurobipy unavailable",
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

    if fleet.vacant.sum() <= 0:
        _update_reposition_trace(
            t_start=t_start,
            status=None,
            sol_count=None,
            runtime_sec=None,
            outcome="skip_no_vacant",
            note="no vacant vehicles",
        )
        return X, Y

    all_states = [(i, l) for i in range(m) for l in range(levels_plus)]

    # Candidate dispatch destinations by state/time.
    top_demand_by_k = [
        np.argsort(-demand_window[k])[: max(1, config.top_demand_targets)] for k in range(H)
    ]
    swap_priority = np.argsort(-station_full_batteries)

    # Sparse transition support to keep MPC tractable.
    trans_k = max(1, min(config.transition_topk, m))
    trans_targets: Dict[Tuple[int, int], np.ndarray] = {}
    for k in range(H):
        t_idx = min(t_start + k, transition.P.shape[0] - 1)
        for i_prev in range(m):
            row = transition.P[t_idx, i_prev] + transition.Q[t_idx, i_prev]
            trans_targets[(k, i_prev)] = _topk_dests(row, trans_k)

    try:
        model = _build_model("fleet_reposition_mpc_vo")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = config.time_limit_sec

        # States
        v: Dict[Tuple[int, int, int], gp.Var] = {}
        o: Dict[Tuple[int, int, int], gp.Var] = {}
        for k in range(H + 1):
            for i in range(m):
                for l in range(levels_plus):
                    if k == 0:
                        v[(k, i, l)] = model.addVar(
                            vtype=GRB.CONTINUOUS,
                            lb=float(fleet.vacant[i, l]),
                            ub=float(fleet.vacant[i, l]),
                            name=f"v_{k}_{i}_{l}",
                        )
                        o[(k, i, l)] = model.addVar(
                            vtype=GRB.CONTINUOUS,
                            lb=float(fleet.occupied[i, l]),
                            ub=float(fleet.occupied[i, l]),
                            name=f"o_{k}_{i}_{l}",
                        )
                    else:
                        v[(k, i, l)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"v_{k}_{i}_{l}")
                        o[(k, i, l)] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"o_{k}_{i}_{l}")

        # Dispatch/processing variables
        x_vars: Dict[Tuple[int, int, int, int], gp.Var] = {}
        y_vars: Dict[Tuple[int, int, int, int], gp.Var] = {}
        r_vars: Dict[Tuple[int, int, int], gp.Var] = {}
        b_vars: Dict[Tuple[int, int, int], gp.Var] = {}
        mu_vars: Dict[Tuple[int, int, int], gp.Var] = {}

        outbound: Dict[Tuple[int, int, int], List[gp.Var]] = {}
        service_inbound: Dict[Tuple[int, int], List[gp.Var]] = {}
        swap_inbound: Dict[Tuple[int, int, int], List[gp.Var]] = {}
        low_energy_y: List[gp.Var] = []
        move_cost_terms: List[gp.LinExpr] = []

        for k in range(H):
            top_demand = top_demand_by_k[k]
            for i, l in all_states:
                state_key = (k, i, l)
                state_out: List[gp.Var] = []
                r_vars[state_key] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"r_{k}_{i}_{l}")
                state_out.append(r_vars[state_key])
                outbound[state_key] = state_out

                feasible = np.where((reachability[i] == 0) & (energy_consumption[i] <= l))[0]
                if feasible.size == 0:
                    continue
                feasible_set = set(int(v_) for v_ in feasible.tolist())

                service_targets = [int(j) for j in top_demand if int(j) in feasible_set]
                if i in feasible_set and i not in service_targets:
                    service_targets.append(i)
                if len(service_targets) < config.top_demand_targets:
                    remain = [int(j) for j in feasible if int(j) not in service_targets]
                    remain.sort(key=lambda z: float(demand_window[k, z]), reverse=True)
                    service_targets.extend(remain[: config.top_demand_targets - len(service_targets)])

                swap_targets = [int(j) for j in swap_priority if int(j) in feasible_set][: config.top_swap_targets]
                if i in feasible_set and i not in swap_targets:
                    swap_targets = [i] + swap_targets
                    swap_targets = swap_targets[: config.top_swap_targets]

                if l > config.swap_low_energy_threshold:
                    for j in service_targets:
                        key = (k, i, j, l)
                        if key in x_vars:
                            continue
                        var = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"x_{k}_{i}_{j}_{l}")
                        x_vars[key] = var
                        state_out.append(var)
                        service_inbound.setdefault((k, j), []).append(var)
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
            _update_reposition_trace(
                t_start=t_start,
                status=None,
                sol_count=None,
                runtime_sec=None,
                outcome="skip_no_candidates",
                note="no feasible dispatch vars",
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

        # Dispatch feasibility from vacant states.
        for k in range(H):
            for i in range(m):
                for l in range(levels_plus):
                    vars_ = outbound.get((k, i, l), [])
                    model.addConstr(gp.quicksum(vars_) == v[(k, i, l)], name=f"vacant_split_{k}_{i}_{l}")

        # Swap equations and stock dynamics.
        for k in range(H):
            for j in range(m):
                for l in range(levels_plus):
                    model.addConstr(
                        b_vars[(k, j, l)] == gp.quicksum(swap_inbound.get((k, j, l), [])),
                        name=f"b_def_{k}_{j}_{l}",
                    )
                    model.addConstr(mu_vars[(k, j, l)] <= b_vars[(k, j, l)], name=f"mu_le_b_{k}_{j}_{l}")

                model.addConstr(
                    gp.quicksum(mu_vars[(k, j, l)] for l in range(levels_plus)) <= float(station_swapping_capacity[j]),
                    name=f"swap_machine_cap_{k}_{j}",
                )
                model.addConstr(
                    gp.quicksum(mu_vars[(k, j, l)] for l in range(levels_plus)) <= full_stock[(k, j)],
                    name=f"swap_full_cap_{k}_{j}",
                )
                model.addConstr(
                    full_stock[(k + 1, j)]
                    == full_stock[(k, j)] - gp.quicksum(mu_vars[(k, j, l)] for l in range(levels_plus)),
                    name=f"full_dyn_{k}_{j}",
                )

                model.addConstr(
                    served[(k, j)] <= gp.quicksum(service_inbound.get((k, j), [])),
                    name=f"served_by_supply_{k}_{j}",
                )

        # Expected V/O transitions over horizon.
        for k in range(H):
            t_idx = min(t_start + k, transition.P.shape[0] - 1)
            for i in range(m):
                for l in range(levels_plus):
                    o_terms: List[gp.LinExpr] = []
                    v_terms: List[gp.LinExpr] = []

                    # Keep undispatched vacant vehicles at the same state.
                    v_terms.append(r_vars[(k, i, l)])

                    # Align MPC with simulator semantics: only successful swaps
                    # and direct full-battery arrivals re-enter dispatchable vacant pool.
                    if l == levels:
                        v_terms.append(
                            gp.quicksum(mu_vars[(k, i, ll)] for ll in range(levels_plus))
                            + b_vars[(k, i, levels)]
                        )

                    for i_prev in range(m):
                        if i not in trans_targets[(k, i_prev)]:
                            continue
                        e = int(energy_consumption[i_prev, i])
                        l_prev = l + e
                        if l_prev > levels:
                            continue

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
                            o_terms.append(pocc * o[(k, i_prev, l_prev)])
                        if qtil != 0.0:
                            v_terms.append(qtil * s_prev)
                        if qocc != 0.0:
                            v_terms.append(qocc * o[(k, i_prev, l_prev)])

                    model.addConstr(o[(k + 1, i, l)] == gp.quicksum(o_terms), name=f"o_dyn_{k}_{i}_{l}")
                    model.addConstr(v[(k + 1, i, l)] == gp.quicksum(v_terms), name=f"v_dyn_{k}_{i}_{l}")

        objective = (
            config.service_reward
            * gp.quicksum((0.95**k) * served[(k, j)] for k in range(H) for j in range(m))
            - config.idle_cost_weight * gp.quicksum(move_cost_terms)
            + config.low_energy_swap_bonus * gp.quicksum(low_energy_y)
        )
        model.setObjective(objective, GRB.MAXIMIZE)
        model.optimize()

        status_code = int(model.Status)
        sol_count = int(model.SolCount)
        _update_reposition_trace(
            t_start=t_start,
            status=status_code,
            sol_count=sol_count,
            runtime_sec=float(model.Runtime),
            outcome="optimized" if sol_count > 0 else "no_incumbent",
            note="",
        )

        if model.SolCount <= 0:
            if _retry_stage == 0 and H > 1:
                _update_reposition_trace(
                    t_start=t_start,
                    status=status_code,
                    sol_count=sol_count,
                    runtime_sec=float(model.Runtime),
                    outcome="retry_h1_no_incumbent",
                    note=f"h={H}, tl={config.time_limit_sec}",
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
        _update_reposition_trace(
            t_start=t_start,
            status=None,
            sol_count=None,
            runtime_sec=None,
            outcome="fallback_greedy_gurobi_error",
            note=str(exc),
        )
        return greedy_same_zone_policy(fleet, demand_window[0], config)
    except Exception as exc:
        prev_note = _LAST_GUROBI_REPOSITION_TRACE.get("note", "")
        _update_reposition_trace(
            t_start=t_start,
            status=_LAST_GUROBI_REPOSITION_TRACE.get("status"),  # type: ignore[arg-type]
            sol_count=_LAST_GUROBI_REPOSITION_TRACE.get("sol_count"),  # type: ignore[arg-type]
            runtime_sec=_LAST_GUROBI_REPOSITION_TRACE.get("runtime_sec"),  # type: ignore[arg-type]
            outcome="exception",
            note=f"retry_stage={_retry_stage}; {prev_note}; {type(exc).__name__}: {exc}".strip("; "),
        )
        raise RuntimeError(f"gurobi_reposition_policy failed: {exc}") from exc
