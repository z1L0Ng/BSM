from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cvxpy as cp
import numpy as np
import pandas as pd


@dataclass
class OptimizationResult:
    X: np.ndarray
    Y: np.ndarray
    served: np.ndarray


class FleetOptimizer:
    def __init__(
        self,
        demand: np.ndarray,
        od_tensor: np.ndarray,
        travel_stats: pd.DataFrame,
        zones: List[int],
        config: Dict,
    ) -> None:
        self.demand = demand
        self.od_tensor = od_tensor
        self.travel_stats = travel_stats
        self.zones = zones
        self.cfg = config

        self.n_zones = len(zones)
        self.n_times = demand.shape[0]
        self.n_levels = int(config["ev"]["soc_levels"])
        self.station_zones = [int(z) for z in config["stations"]["zones"]]
        self.station_caps = {int(z): int(config["stations"]["capacity_per_slot"]) for z in self.station_zones}

    def _distance_matrix(self) -> np.ndarray:
        idx = {z: i for i, z in enumerate(self.zones)}
        mat = np.zeros((self.n_zones, self.n_zones), dtype=float)
        for row in self.travel_stats.itertuples(index=False):
            i = idx[int(row.pu_zone)]
            j = idx[int(row.do_zone)]
            mat[i, j] = float(row.mean_distance_mi)
        if np.any(mat == 0):
            fallback = mat[mat > 0].mean() if np.any(mat > 0) else 1.0
            mat[mat == 0] = fallback
        return mat

    def _energy_drop_levels(self, distance_matrix: np.ndarray) -> np.ndarray:
        kwh_per_mile = float(self.cfg["ev"]["kwh_per_mile"])
        battery_kwh = float(self.cfg["ev"]["battery_kwh"])
        level_step = battery_kwh / (self.n_levels - 1)
        energy_kwh = distance_matrix * kwh_per_mile
        drop = np.ceil(energy_kwh / level_step).astype(int)
        return drop

    def solve(self) -> OptimizationResult:
        n, t, l = self.n_zones, self.n_times, self.n_levels
        dist = self._distance_matrix()
        drop = self._energy_drop_levels(dist)

        X = cp.Variable((t, n, n, l), nonneg=True)
        Y = cp.Variable((t, n, n, l), nonneg=True)
        S = cp.Variable((t + 1, n, l), nonneg=True)

        total_vehicles = int(self.cfg["simulation"]["n_vehicles"])
        init_dist = self.demand.sum(axis=0)
        init_dist = init_dist / max(init_dist.sum(), 1.0)

        init_soc = np.zeros(l)
        init_soc[-1] = 1.0
        S0 = total_vehicles * init_dist[:, None] * init_soc[None, :]

        constraints = [S[0] == S0]

        for ti in range(t):
            for i in range(n):
                constraints.append(cp.sum(X[ti, i, :, :]) <= self.demand[ti, i])

        for ti in range(t):
            for i in range(n):
                outflow = cp.sum(X[ti, i, :, :]) + cp.sum(Y[ti, i, :, :])
                constraints.append(outflow <= cp.sum(S[ti, i, :]))

            for j in range(n):
                for lvl in range(l):
                    outflow_lvl = cp.sum(X[ti, j, :, lvl]) + cp.sum(Y[ti, j, :, lvl])

                    inflow_lvl = 0
                    for i in range(n):
                        for k in range(l):
                            new_lvl = max(k - drop[i, j], 0)
                            if new_lvl == lvl:
                                inflow_lvl += X[ti, i, j, k]

                    # Vehicles that swap at station j return full in next slot
                    if self.zones[j] in self.station_zones and lvl == l - 1:
                        inflow_lvl += cp.sum(Y[ti, :, j, :])

                    constraints.append(S[ti + 1, j, lvl] == S[ti, j, lvl] - outflow_lvl + inflow_lvl)

            # Stations capacity
            for sz in self.station_zones:
                if sz in self.zones:
                    si = self.zones.index(sz)
                    constraints.append(cp.sum(Y[ti, :, si, :]) <= self.station_caps[sz])

        beta = float(self.cfg["optimization"]["idle_distance_weight"])
        served = cp.sum(X)
        idle_distance = cp.sum(cp.multiply(dist[None, :, :, None], X + Y))

        objective = cp.Maximize(served - beta * idle_distance)
        problem = cp.Problem(objective, constraints)
        solver_opts = {}
        time_limit = self.cfg.get("optimization", {}).get("time_limit_sec")
        if time_limit:
            solver_opts["TimeLimit"] = float(time_limit)
        try:
            problem.solve(solver=cp.GUROBI, verbose=False, **solver_opts)
        except Exception:
            problem.solve(solver=cp.ECOS, verbose=False)

        return OptimizationResult(
            X=np.maximum(X.value, 0),
            Y=np.maximum(Y.value, 0),
            served=np.sum(X.value, axis=(2, 3)),
        )


@dataclass
class ChargingSchedule:
    power_kw: np.ndarray
    peak_kw: float


class ChargingScheduler:
    def __init__(self, config: Dict) -> None:
        self.cfg = config

    def schedule(self, arrivals: np.ndarray) -> ChargingSchedule:
        slot_hours = float(self.cfg["processing"]["time_bin_minutes"]) / 60
        battery_kwh = float(self.cfg["ev"]["battery_kwh"])
        deadline = int(self.cfg["stations"]["deadline_slots"])
        max_power = float(self.cfg["stations"]["max_power_kw"])

        t = len(arrivals)
        energy_required = arrivals * battery_kwh

        p = cp.Variable(t, nonneg=True)
        peak = cp.Variable(1, nonneg=True)
        constraints = [p <= max_power]

        cumulative_energy = cp.cumsum(p * slot_hours)
        cumulative_arrivals = np.cumsum(energy_required)

        for ti in range(t):
            idx = ti - deadline
            if idx >= 0:
                constraints.append(cumulative_energy[ti] >= cumulative_arrivals[idx])

        constraints.append(cumulative_energy[-1] >= cumulative_arrivals[-1])
        constraints.append(peak >= p)

        objective = cp.Minimize(peak + 1e-4 * cp.sum(p))
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.ECOS, verbose=False)

        return ChargingSchedule(power_kw=p.value, peak_kw=float(peak.value))
