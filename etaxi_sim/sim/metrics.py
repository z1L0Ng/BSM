from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from etaxi_sim.models.fleet import FleetState
from etaxi_sim.models.station import Station


@dataclass
class MetricsRecorder:
    steps: List[Dict] = field(default_factory=list)
    waiting_station_vehicle_slots: Dict[int, int] = field(default_factory=dict)
    waiting_station_peak: Dict[int, int] = field(default_factory=dict)

    def record_step(
        self,
        t: int,
        demand: np.ndarray,
        X: np.ndarray,
        Y: np.ndarray,
        fleet: FleetState,
        stations: List[Station],
        charged_by_station: Dict[int, int] | None = None,
        deadline_misses: int = 0,
        charge_power_kw: float = 0.0,
        swap_arrivals: int = 0,
        swap_requests: int = 0,
        successful_swaps: int = 0,
        unmet_battery_demand: int = 0,
        charging_demand: int = 0,
        total_charging_demand_kw: float = 0.0,
        waiting_time_for_battery_slots: float = 0.0,
        idle_driving_distance: float | None = None,
        waiting_vehicles: int = 0,
        waiting_vehicles_by_station: np.ndarray | None = None,
    ) -> None:
        served_supply = X.sum(axis=0).sum(axis=1)
        served = np.minimum(served_supply, demand).sum()
        idle_moves = X.sum() + Y.sum()
        full_batteries = sum(s.full_batteries for s in stations)
        charged_by_station = charged_by_station or {}
        station_power = [
            s.base_load_kw + charge_power_kw * charged_by_station.get(s.station_id, 0) for s in stations
        ]
        peak_station_power = max(station_power) if station_power else 0.0
        charging_power_kw = float(charge_power_kw * sum(charged_by_station.values()))
        station_total_power_kw = float(sum(station_power))
        ratio_den = swap_requests if swap_requests > 0 else swap_arrivals
        swap_success_ratio = float(successful_swaps / ratio_den) if ratio_den > 0 else 0.0
        deadline_miss_ratio = float(deadline_misses / ratio_den) if ratio_den > 0 else 0.0

        self.steps.append(
            {
                "t": t,
                "time_slot": int(t),
                "served": int(served),
                "idle_moves": int(idle_moves),
                "idle_driving_distance": idle_driving_distance,
                "vacant": int(fleet.vacant.sum()),
                "occupied": int(fleet.occupied.sum()),
                "full_batteries": int(full_batteries),
                "swap_arrivals": int(swap_arrivals),
                "swap_requests": int(swap_requests),
                "number_of_swaps": int(successful_swaps),
                "swap_success_ratio": swap_success_ratio,
                "unmet_battery_demand": int(unmet_battery_demand),
                "waiting_vehicles": int(waiting_vehicles),
                "deadline_misses": int(deadline_misses),
                "deadline_miss_ratio": deadline_miss_ratio,
                "waiting_time_for_battery_slots": float(waiting_time_for_battery_slots),
                "charging_demand": int(charging_demand),
                "total_charging_demand_kw": float(total_charging_demand_kw),
                "charging_power_kw": charging_power_kw,
                "station_total_power_kw": station_total_power_kw,
                "peak_station_power_kw": float(peak_station_power),
            }
        )
        if waiting_vehicles_by_station is not None:
            for sid, count in enumerate(waiting_vehicles_by_station.astype(int).tolist()):
                self.waiting_station_vehicle_slots[sid] = self.waiting_station_vehicle_slots.get(sid, 0) + count
                self.waiting_station_peak[sid] = max(self.waiting_station_peak.get(sid, 0), count)

    def to_summary(self) -> Dict[str, float | int | None]:
        if not self.steps:
            return {}
        served = sum(s["served"] for s in self.steps)
        idle_moves = sum(s["idle_moves"] for s in self.steps)
        deadline_misses = sum(s["deadline_misses"] for s in self.steps)
        swap_arrivals = sum(s["swap_arrivals"] for s in self.steps)
        swap_requests = sum(s["swap_requests"] for s in self.steps)
        successful_swaps = sum(s["number_of_swaps"] for s in self.steps)
        unmet_battery_demand = sum(s["unmet_battery_demand"] for s in self.steps)
        waiting_vehicles = sum(s["waiting_vehicles"] for s in self.steps)
        charging_demand = sum(s["charging_demand"] for s in self.steps)
        waiting_time_for_battery_slots = sum(s["waiting_time_for_battery_slots"] for s in self.steps)
        waiting_series = np.array([s["waiting_vehicles"] for s in self.steps], dtype=float)
        peak_station_power_kw = max(s["peak_station_power_kw"] for s in self.steps)
        max_charging_power_kw = max(s["charging_power_kw"] for s in self.steps)
        max_total_charging_demand_kw = max(s["total_charging_demand_kw"] for s in self.steps)
        max_station_total_power_kw = max(s["station_total_power_kw"] for s in self.steps)

        has_idle_distance = all(s["idle_driving_distance"] is not None for s in self.steps)
        total_idle_driving_distance = (
            float(sum(float(s["idle_driving_distance"]) for s in self.steps)) if has_idle_distance else None
        )

        ratio_den = swap_requests if swap_requests > 0 else swap_arrivals
        battery_swap_success_ratio = float(successful_swaps / ratio_den) if ratio_den > 0 else 0.0
        deadline_miss_ratio = float(deadline_misses / ratio_den) if ratio_den > 0 else 0.0
        avg_waiting_time_for_battery_slots = (
            float(waiting_time_for_battery_slots / ratio_den) if ratio_den > 0 else 0.0
        )
        waiting_vehicles_p50 = float(np.quantile(waiting_series, 0.50)) if waiting_series.size > 0 else 0.0
        waiting_vehicles_p90 = float(np.quantile(waiting_series, 0.90)) if waiting_series.size > 0 else 0.0
        waiting_vehicles_p99 = float(np.quantile(waiting_series, 0.99)) if waiting_series.size > 0 else 0.0

        station_wait_rank = sorted(
            self.waiting_station_vehicle_slots.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )[:5]
        station_peak_rank = sorted(
            self.waiting_station_peak.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )[:5]
        waiting_vehicle_slots_by_station_top5 = {str(k): int(v) for k, v in station_wait_rank}
        max_waiting_vehicles_by_station_top5 = {str(k): int(v) for k, v in station_peak_rank}

        return {
            "total_served": served,
            "total_idle_moves": idle_moves,
            "total_idle_driving_distance": total_idle_driving_distance,
            "total_swap_arrivals": swap_arrivals,
            "total_swap_requests": swap_requests,
            "total_number_of_swaps": successful_swaps,
            "battery_swap_success_ratio": battery_swap_success_ratio,
            "total_unmet_battery_demand": unmet_battery_demand,
            "total_waiting_vehicles": waiting_vehicles,
            "total_deadline_misses": deadline_misses,
            "deadline_miss_ratio": deadline_miss_ratio,
            "total_charging_demand": charging_demand,
            "max_total_charging_demand_kw": max_total_charging_demand_kw,
            "max_charging_power_demand_kw": max_total_charging_demand_kw,
            "max_charging_power_kw": max_charging_power_kw,
            "max_station_total_power_kw": max_station_total_power_kw,
            "avg_waiting_time_for_battery_slots": avg_waiting_time_for_battery_slots,
            "waiting_vehicles_p50": waiting_vehicles_p50,
            "waiting_vehicles_p90": waiting_vehicles_p90,
            "waiting_vehicles_p99": waiting_vehicles_p99,
            "waiting_vehicle_slots_by_station_top5": waiting_vehicle_slots_by_station_top5,
            "max_waiting_vehicles_by_station_top5": max_waiting_vehicles_by_station_top5,
            "peak_station_power_kw": peak_station_power_kw,
            "steps": len(self.steps),
        }
