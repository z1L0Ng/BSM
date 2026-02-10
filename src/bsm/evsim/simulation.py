from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class SimulationOutputs:
    demand_curve: np.ndarray
    supply_baseline: np.ndarray
    supply_proposed: np.ndarray
    baseline_power_kw: np.ndarray
    proposed_power_kw: np.ndarray
    baseline_inventory: np.ndarray
    proposed_inventory: np.ndarray


def _energy_drop_levels(distance_matrix: np.ndarray, battery_kwh: float, kwh_per_mile: float, n_levels: int) -> np.ndarray:
    level_step = battery_kwh / (n_levels - 1)
    energy_kwh = distance_matrix * kwh_per_mile
    drop = np.ceil(energy_kwh / level_step).astype(int)
    return drop


def simulate_baseline(
    demand: np.ndarray,
    od_tensor: np.ndarray,
    distance_matrix: np.ndarray,
    zones: List[int],
    config: Dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_times, n_zones = demand.shape
    n_levels = int(config["ev"]["soc_levels"])
    battery_kwh = float(config["ev"]["battery_kwh"])
    kwh_per_mile = float(config["ev"]["kwh_per_mile"])
    soc_threshold = float(config["ev"]["soc_threshold"])
    monitor_zone_id = config["stations"].get("monitor_zone_id")
    station_zone_ids = [int(z) for z in config["stations"]["zones"]]
    station_indices = [zones.index(z) for z in station_zone_ids if z in zones]
    if not station_indices:
        station_indices = [0]

    station_idx = 0
    if monitor_zone_id is not None and int(monitor_zone_id) in zones:
        station_idx = zones.index(int(monitor_zone_id))
    full_level = n_levels - 1

    total_vehicles = int(config["simulation"]["n_vehicles"])
    init_dist = demand.sum(axis=0)
    init_dist = init_dist / max(init_dist.sum(), 1.0)

    fleet = np.zeros((n_zones, n_levels))
    fleet[:, full_level] = total_vehicles * init_dist

    drop_levels = _energy_drop_levels(distance_matrix, battery_kwh, kwh_per_mile, n_levels)
    threshold_level = int(np.floor(soc_threshold * full_level))

    supply = np.zeros(n_times)
    power = np.zeros(n_times)
    inventory = np.zeros(n_times)

    full_batteries = {idx: float(config["stations"]["initial_full_batteries"]) for idx in station_indices}
    for t in range(n_times):
        demand_t = demand[t]
        od_t = od_tensor[t]

        served_t = 0.0
        arrivals_empty = {idx: 0.0 for idx in station_indices}

        for i in range(n_zones):
            available = fleet[i, threshold_level:].sum()
            served = min(available, demand_t[i])
            served_t += served

            if served > 0:
                probs = od_t[i]
                probs = probs / probs.sum() if probs.sum() > 0 else np.ones(n_zones) / n_zones
                dest_counts = served * probs

                for j in range(n_zones):
                    drop = drop_levels[i, j]
                    lvl = max(full_level - drop, 0)
                    fleet[j, lvl] += dest_counts[j]

                fleet[i, threshold_level:] -= served * (fleet[i, threshold_level:] / max(available, 1.0))

        # Low SOC vehicles head to station
        low_soc = fleet[:, :threshold_level].sum(axis=1)
        fleet[:, :threshold_level] = 0

        for i in range(n_zones):
            if low_soc[i] <= 0:
                continue
            # send to nearest station
            distances = distance_matrix[i, station_indices]
            nearest_idx = station_indices[int(np.argmin(distances))]
            arrivals_empty[nearest_idx] += low_soc[i]

        # Swap at stations (baseline)
        total_power = 0.0
        for s_idx in station_indices:
            used_full = min(full_batteries[s_idx], arrivals_empty[s_idx])
            full_batteries[s_idx] -= used_full
            fleet[s_idx, full_level] += used_full

            total_power += arrivals_empty[s_idx] * battery_kwh / (float(config["processing"]["time_bin_minutes"]) / 60)
            full_batteries[s_idx] += arrivals_empty[s_idx]

        power[t] = total_power

        supply[t] = served_t
        inventory[t] = full_batteries.get(station_idx, 0.0)

    return supply, power, inventory


def simulate_proposed(
    demand: np.ndarray,
    optimization_served: np.ndarray,
    charging_power: np.ndarray,
    inventory: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    supply = optimization_served.sum(axis=1)
    return supply, charging_power, inventory
