from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from etaxi_sim.models.charging import ChargingTask

try:
    import gurobipy as gp
    from gurobipy import GRB

    HAS_GUROBI = True
except Exception:
    HAS_GUROBI = False

_GUROBI_ENV = None


@dataclass(frozen=True)
class ChargingPolicyConfig:
    planning_horizon_slots: int
    charge_power_kw: float
    miss_penalty: float = 1000.0
    time_limit_sec: float = 3.0


def _build_model(name: str) -> gp.Model:
    global _GUROBI_ENV
    if _GUROBI_ENV is None:
        _GUROBI_ENV = gp.Env(empty=True)
        _GUROBI_ENV.setParam("OutputFlag", 0)
        _GUROBI_ENV.start()
    return gp.Model(name, env=_GUROBI_ENV)


def edf_charging_policy(
    tasks: List[ChargingTask],
    chargers_by_station: Dict[int, int],
    current_time: int,
) -> List[ChargingTask]:
    charged: List[ChargingTask] = []

    # group by station
    tasks_by_station: Dict[int, List[ChargingTask]] = {k: [] for k in chargers_by_station.keys()}
    for task in tasks:
        if task.is_completed() or not task.is_available(current_time):
            continue
        tasks_by_station[task.station_id].append(task)

    for station_id, station_tasks in tasks_by_station.items():
        station_tasks.sort(key=lambda t: t.deadline)
        capacity = chargers_by_station[station_id]
        for task in station_tasks[:capacity]:
            charged.append(task)

    return charged


def gurobi_peak_charging_policy(
    tasks: List[ChargingTask],
    chargers_by_station: Dict[int, int],
    base_load_by_station: Dict[int, float],
    current_time: int,
    config: ChargingPolicyConfig,
) -> List[ChargingTask]:
    if not HAS_GUROBI:
        return edf_charging_policy(tasks, chargers_by_station, current_time)

    pending = [task for task in tasks if (not task.is_completed()) and task.deadline > current_time]
    if not pending:
        return []

    latest_deadline = max(task.deadline for task in pending)
    window_end = min(latest_deadline, current_time + max(1, config.planning_horizon_slots))
    if window_end <= current_time:
        return []
    times = list(range(current_time, window_end))

    try:
        model = _build_model("charging_peak")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = config.time_limit_sec

        x: Dict[tuple[int, int], gp.Var] = {}
        slack: Dict[int, gp.Var] = {}
        task_by_idx: Dict[int, ChargingTask] = {}

        for idx, task in enumerate(pending):
            task_by_idx[idx] = task
            avail_start = max(current_time, task.arrival_time)
            avail_end = min(task.deadline, window_end)
            feasible_times = [tt for tt in times if avail_start <= tt < avail_end]
            for tt in feasible_times:
                x[(idx, tt)] = model.addVar(vtype=GRB.BINARY, name=f"x_{idx}_{tt}")
            slack[idx] = model.addVar(
                vtype=GRB.INTEGER,
                lb=0,
                ub=int(task.remaining_slots),
                name=f"slack_{idx}",
            )
            model.addConstr(
                gp.quicksum(x[idx, tt] for tt in feasible_times) + slack[idx] == int(task.remaining_slots),
                name=f"task_slots_{idx}",
            )

        for station_id in chargers_by_station:
            cap = int(chargers_by_station[station_id])
            for tt in times:
                expr = gp.quicksum(
                    x[idx, tt]
                    for idx, task in task_by_idx.items()
                    if task.station_id == station_id and (idx, tt) in x
                )
                model.addConstr(expr <= cap, name=f"charger_cap_{station_id}_{tt}")

        z = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="peak_kw")
        for station_id in chargers_by_station:
            base_kw = float(base_load_by_station.get(station_id, 0.0))
            for tt in times:
                load = gp.quicksum(
                    x[idx, tt]
                    for idx, task in task_by_idx.items()
                    if task.station_id == station_id and (idx, tt) in x
                )
                model.addConstr(
                    z >= base_kw + config.charge_power_kw * load,
                    name=f"peak_bind_{station_id}_{tt}",
                )

        model.setObjective(z + config.miss_penalty * gp.quicksum(slack.values()), GRB.MINIMIZE)
        model.optimize()

        if model.SolCount <= 0:
            return edf_charging_policy(tasks, chargers_by_station, current_time)

        charged: List[ChargingTask] = []
        for idx, task in task_by_idx.items():
            var = x.get((idx, current_time))
            if var is not None and var.X > 0.5:
                charged.append(task)
        return charged
    except Exception:
        return edf_charging_policy(tasks, chargers_by_station, current_time)
