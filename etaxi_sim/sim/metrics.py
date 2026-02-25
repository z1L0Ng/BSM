from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from etaxi_sim.models.fleet import FleetState
from etaxi_sim.models.station import Station


@dataclass
class MetricsRecorder:
    steps: List[Dict] = field(default_factory=list)

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

        self.steps.append(
            {
                "t": t,
                "served": int(served),
                "idle_moves": int(idle_moves),
                "vacant": int(fleet.vacant.sum()),
                "occupied": int(fleet.occupied.sum()),
                "full_batteries": int(full_batteries),
                "deadline_misses": int(deadline_misses),
                "peak_station_power_kw": float(peak_station_power),
            }
        )

    def to_summary(self) -> Dict[str, float]:
        if not self.steps:
            return {}
        served = sum(s["served"] for s in self.steps)
        idle_moves = sum(s["idle_moves"] for s in self.steps)
        deadline_misses = sum(s["deadline_misses"] for s in self.steps)
        peak_station_power_kw = max(s["peak_station_power_kw"] for s in self.steps)
        return {
            "total_served": served,
            "total_idle_moves": idle_moves,
            "total_deadline_misses": deadline_misses,
            "peak_station_power_kw": peak_station_power_kw,
            "steps": len(self.steps),
        }
