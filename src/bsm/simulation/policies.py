from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Vehicle:
    vid: int
    zone: int
    busy_until: float


@dataclass
class Request:
    rid: int
    t_request: float
    pu_zone: int
    do_zone: int
    trip_duration_min: float


@dataclass
class Match:
    request: Request
    vehicle: Vehicle
    pickup_eta_min: float


def greedy_same_zone_then_any(
    idle_by_zone: Dict[int, List[Vehicle]],
    idle_pool: List[Vehicle],
    request: Request,
    pickup_eta_min: float,
) -> Optional[Match]:
    zone_pool = idle_by_zone.get(request.pu_zone)
    if zone_pool:
        vehicle = zone_pool.pop()
        idle_pool.remove(vehicle)
        return Match(request=request, vehicle=vehicle, pickup_eta_min=0.0)

    if idle_pool:
        vehicle = idle_pool.pop()
        idle_by_zone[vehicle.zone].remove(vehicle)
        return Match(request=request, vehicle=vehicle, pickup_eta_min=pickup_eta_min)

    return None
