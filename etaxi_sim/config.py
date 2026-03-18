from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(frozen=True)
class Paths:
    data_root: Path
    raw_dir: Path


@dataclass(frozen=True)
class SimConfig:
    seed: int
    time_bin_minutes: int
    horizon: int
    sim_date: str | None
    battery_levels: int
    charge_rate_levels_per_slot: int
    swap_deadline_horizon: int
    swap_low_energy_threshold: int
    min_full_batteries_per_station: int
    initial_vehicles: int
    base_load_kw: float
    initial_vehicles_mode: str = "fixed"
    initial_vehicles_scale: float = 1.0
    sim_start_hour: int = 0
    sim_end_hour: int = 24
    charge_power_kw: float = 30.0


@dataclass(frozen=True)
class StationConfig:
    swapping_capacity: int
    chargers: int


@dataclass(frozen=True)
class DataConfig:
    source: str
    yellow_tripdata_path: Path
    taxi_zone_lookup_path: Path
    taxi_zones_shp_path: Path


@dataclass(frozen=True)
class ModelConfig:
    transition_mode: str
    distance_mode: str
    max_reachable_distance: float
    reposition_solver: str = "gurobi"
    reposition_planning_horizon_slots: int = 4
    charging_solver: str = "gurobi"
    reposition_idle_cost_weight: float = 0.05
    reposition_top_demand_targets: int = 8
    reposition_top_swap_targets: int = 4
    reposition_low_energy_swap_bonus: float = 0.15
    reposition_transition_topk: int = 6
    reposition_solver_method: int = 2
    reposition_solver_crossover: int = 0
    reposition_numeric_focus: int = 0
    reposition_eliminate_auxiliary_vars: bool = True
    reposition_preaggregate_transitions: bool = True
    transition_pickup_floor: float = 0.05
    transition_pickup_ceiling: float = 0.95
    transition_pickup_smoothing_window: int = 4
    charging_miss_penalty: float = 1000.0
    solver_time_limit_sec: float = 3.0


@dataclass(frozen=True)
class Config:
    paths: Paths
    sim: SimConfig
    station: StationConfig
    data: DataConfig
    model: ModelConfig


def _to_path(base: Path, value: str) -> Path:
    p = Path(value)
    if not p.is_absolute():
        p = base / p
    return p


def load_config(path: str | Path) -> Config:
    path = Path(path)
    raw: Dict[str, Any] = yaml.safe_load(path.read_text())

    data_root = _to_path(path.parent, raw["paths"]["data_root"])
    raw_dir = _to_path(path.parent, raw["paths"]["raw_dir"])

    sim_raw = raw["sim"]
    if "sim_date" not in sim_raw:
        sim_raw["sim_date"] = None
    if "sim_start_hour" not in sim_raw:
        sim_raw["sim_start_hour"] = 0
    if "sim_end_hour" not in sim_raw:
        sim_raw["sim_end_hour"] = 24

    cfg = Config(
        paths=Paths(data_root=data_root, raw_dir=raw_dir),
        sim=SimConfig(**sim_raw),
        station=StationConfig(**raw["station"]),
        data=DataConfig(
            source=raw["data"]["source"],
            yellow_tripdata_path=_to_path(path.parent, raw["data"]["yellow_tripdata_path"]),
            taxi_zone_lookup_path=_to_path(path.parent, raw["data"]["taxi_zone_lookup_path"]),
            taxi_zones_shp_path=_to_path(path.parent, raw["data"]["taxi_zones_shp_path"]),
        ),
        model=ModelConfig(**raw["model"]),
    )
    return cfg
