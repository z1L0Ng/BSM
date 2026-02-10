from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def compute_metrics(requests: pd.DataFrame, utilization: float, sim_minutes: float) -> Dict[str, float]:
    served = requests[requests["served"] == True]
    unserved = requests[requests["served"] == False]

    metrics: Dict[str, float] = {}
    if not served.empty:
        metrics["avg_wait_min"] = float(served["wait_min"].mean())
        metrics["p95_wait_min"] = float(served["wait_min"].quantile(0.95))
        metrics["avg_trip_min"] = float(served["trip_min"].mean())
    else:
        metrics["avg_wait_min"] = float("nan")
        metrics["p95_wait_min"] = float("nan")
        metrics["avg_trip_min"] = float("nan")

    total = len(requests)
    metrics["unserved_rate"] = float(len(unserved) / total) if total > 0 else float("nan")
    metrics["utilization"] = float(utilization)

    if total > 0:
        metrics["throughput_per_hour"] = float(len(served) / (sim_minutes / 60 + 1e-6))
    else:
        metrics["throughput_per_hour"] = float("nan")

    return metrics
