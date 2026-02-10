from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from bsm.io import ensure_dirs, load_config
from bsm.simulation.engine import run_simulation
from bsm.simulation.metrics import compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    processed_path = Path(cfg.paths["processed_dir"]) / f"{cfg.dataset['name']}_{cfg.dataset['year']}-{cfg.dataset['month']:02d}_clean.parquet"
    df = pd.read_parquet(processed_path)

    if cfg.simulation.get("sample_fraction", 1.0) < 1.0:
        df = df.sample(frac=cfg.simulation["sample_fraction"], random_state=cfg.simulation["random_seed"])\
            .sort_values("pickup_ts")

    result = run_simulation(
        requests_df=df,
        n_vehicles=cfg.simulation["n_vehicles"],
        step_minutes=cfg.simulation["step_minutes"],
        max_wait_minutes=cfg.simulation["max_wait_minutes"],
        pickup_eta_minutes=cfg.simulation["pickup_eta_minutes"],
        policy=cfg.simulation["policy"],
        random_seed=cfg.simulation["random_seed"],
    )

    metrics = compute_metrics(result.requests, result.vehicle_utilization, result.sim_minutes)

    metrics_path = Path(cfg.paths["reports_dir"]) / f"metrics_{cfg.dataset['name']}_{cfg.dataset['year']}-{cfg.dataset['month']:02d}.csv"
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    print("Simulation complete")
    print(metrics)
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
