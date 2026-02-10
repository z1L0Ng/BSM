from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from .policies import Match, Request, Vehicle, greedy_same_zone_then_any


PolicyFn = Callable[[Dict[int, List[Vehicle]], List[Vehicle], Request, float], Match | None]


@dataclass
class SimulationResult:
    requests: pd.DataFrame
    vehicle_utilization: float
    sim_minutes: float


def _build_idle_index(vehicles: List[Vehicle]) -> Dict[int, List[Vehicle]]:
    idle_by_zone: Dict[int, List[Vehicle]] = {}
    for v in vehicles:
        idle_by_zone.setdefault(v.zone, []).append(v)
    return idle_by_zone


def run_simulation(
    requests_df: pd.DataFrame,
    n_vehicles: int,
    step_minutes: int,
    max_wait_minutes: int,
    pickup_eta_minutes: float,
    policy: str,
    random_seed: int,
) -> SimulationResult:
    rng = np.random.default_rng(random_seed)

    requests_df = requests_df.sort_values("pickup_ts").reset_index(drop=True)
    t0 = requests_df["pickup_ts"].min()
    t1 = requests_df["pickup_ts"].max()

    time_index = pd.date_range(t0, t1, freq=f"{step_minutes}min")

    zone_weights = requests_df["pu_zone"].value_counts(normalize=True)
    zones = zone_weights.index.to_numpy()
    probs = zone_weights.values

    vehicles = [Vehicle(vid=i, zone=int(rng.choice(zones, p=probs)), busy_until=0.0) for i in range(n_vehicles)]
    idle_pool = vehicles.copy()
    idle_by_zone = _build_idle_index(idle_pool)

    pending: List[Request] = []
    served = []
    unserved = []
    busy_time = 0.0

    policy_fn: PolicyFn = greedy_same_zone_then_any if policy == "greedy_same_zone_then_any" else greedy_same_zone_then_any

    req_cursor = 0
    req_rows = requests_df.to_dict("records")

    for current_time in time_index:
        current_min = (current_time - t0).total_seconds() / 60

        # Release vehicles that completed trips
        for v in vehicles:
            if v.busy_until and v.busy_until <= current_min and v not in idle_pool:
                v.busy_until = 0.0
                idle_pool.append(v)
                idle_by_zone.setdefault(v.zone, []).append(v)

        # Add arriving requests for this time window
        while req_cursor < len(req_rows) and req_rows[req_cursor]["pickup_ts"] <= current_time:
            row = req_rows[req_cursor]
            pending.append(
                Request(
                    rid=req_cursor,
                    t_request=(row["pickup_ts"] - t0).total_seconds() / 60,
                    pu_zone=int(row["pu_zone"]),
                    do_zone=int(row["do_zone"]),
                    trip_duration_min=float(row["duration_min"]),
                )
            )
            req_cursor += 1

        still_pending: List[Request] = []
        for req in pending:
            wait_time = current_min - req.t_request
            if wait_time > max_wait_minutes:
                unserved.append(
                    {
                        "rid": req.rid,
                        "served": False,
                        "wait_min": wait_time,
                        "pickup_eta_min": None,
                        "trip_min": None,
                    }
                )
                continue

            match = policy_fn(idle_by_zone, idle_pool, req, pickup_eta_minutes)
            if match is None:
                still_pending.append(req)
                continue

            total_wait = wait_time + match.pickup_eta_min
            trip_time = req.trip_duration_min
            match.vehicle.busy_until = current_min + match.pickup_eta_min + trip_time
            match.vehicle.zone = req.do_zone

            busy_time += match.pickup_eta_min + trip_time
            served.append(
                {
                    "rid": req.rid,
                    "served": True,
                    "wait_min": total_wait,
                    "pickup_eta_min": match.pickup_eta_min,
                    "trip_min": trip_time,
                }
            )

        pending = still_pending

    total_sim_time = (time_index[-1] - time_index[0]).total_seconds() / 60
    utilization = busy_time / max(total_sim_time * n_vehicles, 1.0)

    result_df = pd.DataFrame(served + unserved)
    return SimulationResult(requests=result_df, vehicle_utilization=utilization, sim_minutes=total_sim_time)
