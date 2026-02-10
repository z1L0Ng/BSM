from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def plot_demand_supply(time_bins, demand_curve, supply_a, supply_b, output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(time_bins, demand_curve, label="Demand", color="#1b4965")
    plt.plot(time_bins, supply_a, label="Baseline Supply", color="#f6aa1c")
    plt.plot(time_bins, supply_b, label="Proposed Supply", color="#1a936f")
    plt.legend()
    plt.ylabel("Trips per slot")
    plt.xlabel("Time")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    return output


def plot_power_profile(time_bins, power_a, power_b, output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(time_bins, power_a, label="Baseline Power", color="#bc4b51")
    plt.plot(time_bins, power_b, label="Proposed Power", color="#5c8001")
    plt.legend()
    plt.ylabel("kW")
    plt.xlabel("Time")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    return output


def plot_inventory(time_bins, inv_a, inv_b, output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(time_bins, inv_a, label="Baseline Inventory", color="#5e548e")
    plt.plot(time_bins, inv_b, label="Proposed Inventory", color="#2f3e46")
    plt.legend()
    plt.ylabel("Full Batteries")
    plt.xlabel("Time")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()
    return output
