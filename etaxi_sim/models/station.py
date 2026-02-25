from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from etaxi_sim.models.charging import ChargingTask


@dataclass
class Station:
    station_id: int
    swapping_capacity: int
    chargers: int
    battery_levels: int
    full_batteries: int
    partial_batteries: np.ndarray  # shape (L)
    base_load_kw: float = 0.0
    pending_charge: Dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.pending_charge == {}:
            self.pending_charge = {l: 0 for l in range(self.battery_levels)}

    def perform_swapping(
        self,
        arriving_vehicles: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # arriving_vehicles shape (L+1)
        total_demand = int(arriving_vehicles.sum())
        actual_swaps = min(total_demand, self.full_batteries, self.swapping_capacity)

        swapped = np.zeros_like(arriving_vehicles)
        not_swapped = arriving_vehicles.copy()

        remaining = actual_swaps
        # prioritize low energy levels
        for l in range(self.battery_levels):
            if remaining <= 0:
                break
            count = int(arriving_vehicles[l])
            if count <= 0:
                continue
            s = min(count, remaining)
            swapped[l] = s
            not_swapped[l] -= s
            remaining -= s

        self.full_batteries -= actual_swaps
        for l in range(self.battery_levels):
            if swapped[l] > 0:
                self.pending_charge[l] += int(swapped[l])
                self.partial_batteries[l] += int(swapped[l])

        # vehicles after swapping: low levels remain, swapped vehicles become full
        vehicles_after = not_swapped.copy()
        vehicles_after[self.battery_levels] = int(not_swapped[self.battery_levels] + swapped.sum())

        return swapped, vehicles_after

    def generate_charging_tasks(
        self,
        swapped: np.ndarray,
        current_time: int,
        charge_rate_levels_per_slot: int,
        deadline_horizon: int,
        task_id_start: int,
    ) -> List[ChargingTask]:
        tasks: List[ChargingTask] = []
        task_id = task_id_start
        for l in range(self.battery_levels):
            count = int(swapped[l])
            if count <= 0:
                continue
            energy_needed = self.battery_levels - l
            required_slots = int(np.ceil(energy_needed / charge_rate_levels_per_slot))
            for _ in range(count):
                tasks.append(
                    ChargingTask(
                        task_id=task_id,
                        station_id=self.station_id,
                        arrival_time=current_time,
                        required_slots=required_slots,
                        deadline=current_time + deadline_horizon,
                        energy_level=l,
                        remaining_slots=required_slots,
                        current_level=l,
                    )
                )
                task_id += 1
        return tasks

    def apply_charging(self, charged_tasks: List[ChargingTask], charge_rate_levels_per_slot: int) -> None:
        # Eq. (5)(6): move charged batteries across discrete levels each slot.
        for task in charged_tasks:
            if task.is_completed():
                continue

            prev_level = task.current_level
            if 0 <= prev_level < self.battery_levels and self.partial_batteries[prev_level] > 0:
                self.partial_batteries[prev_level] -= 1
            if 0 <= prev_level < self.battery_levels and self.pending_charge[prev_level] > 0:
                self.pending_charge[prev_level] -= 1

            task.remaining_slots -= 1
            task.current_level = min(self.battery_levels, prev_level + charge_rate_levels_per_slot)

            if task.current_level >= self.battery_levels or task.remaining_slots <= 0:
                task.remaining_slots = 0
                self.full_batteries += 1
            else:
                self.partial_batteries[task.current_level] += 1
                self.pending_charge[task.current_level] += 1
