from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChargingTask:
    task_id: int
    station_id: int
    arrival_time: int
    required_slots: int
    deadline: int
    energy_level: int

    remaining_slots: int
    current_level: int

    def is_available(self, current_time: int) -> bool:
        return self.arrival_time <= current_time < self.deadline

    def is_completed(self) -> bool:
        return self.remaining_slots <= 0
