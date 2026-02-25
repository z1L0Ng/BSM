from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FleetState:
    vacant: np.ndarray  # shape (m, L+1)
    occupied: np.ndarray  # shape (m, L+1)

    @property
    def num_regions(self) -> int:
        return self.vacant.shape[0]

    @property
    def levels(self) -> int:
        return self.vacant.shape[1] - 1


def initialize_fleet(m: int, levels: int, seed: int, initial_vehicles: int) -> FleetState:
    rng = np.random.default_rng(seed)
    vacant = np.zeros((m, levels + 1), dtype=int)
    occupied = np.zeros((m, levels + 1), dtype=int)

    # Randomly distribute vehicles across regions and energy levels
    for _ in range(initial_vehicles):
        i = rng.integers(0, m)
        l = rng.integers(1, levels + 1)
        vacant[i, l] += 1

    return FleetState(vacant=vacant, occupied=occupied)
