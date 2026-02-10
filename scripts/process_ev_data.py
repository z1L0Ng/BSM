from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from bsm.io import ensure_dirs, load_config
from bsm.evsim.data import DataProcessor, build_energy_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    raw_dir = Path(cfg.paths["raw_dir"])
    filename = cfg.dataset.get("filename")
    if not filename:
        years = [cfg.dataset["year"]] + cfg.dataset.get("fallback_years", [])
        filename = None
        for year in years:
            candidate = cfg.dataset["filename_template"].format(year=year, month=cfg.dataset["month"])
            if (raw_dir / candidate).exists():
                filename = candidate
                break
        if filename is None:
            filename = cfg.dataset["filename_template"].format(year=cfg.dataset["year"], month=cfg.dataset["month"])

    trips_path = raw_dir / filename
    zones_path = raw_dir / cfg.data["zone_lookup"]["filename"]

    processor = DataProcessor(cfg.data)
    bundle = processor.process(trips_path=trips_path, zones_path=zones_path)

    processed_dir = Path(cfg.paths["processed_dir"])
    demand_path = processed_dir / "ev_demand.parquet"
    od_path = processed_dir / "ev_od.parquet"
    travel_path = processed_dir / "ev_travel_stats.parquet"

    bundle.demand.to_parquet(demand_path, index=False)
    bundle.od.to_parquet(od_path, index=False)
    bundle.travel_stats.to_parquet(travel_path, index=False)

    energy_table = build_energy_table(bundle.travel_stats, cfg.data["ev"]["kwh_per_mile"])
    energy_path = processed_dir / "ev_energy.parquet"
    energy_table.to_parquet(energy_path, index=False)

    zones_path_out = processed_dir / "ev_zones.csv"
    pd.DataFrame({"zone_id": bundle.zones}).to_csv(zones_path_out, index=False)

    print(f"Demand saved to {demand_path}")
    print(f"OD saved to {od_path}")
    print(f"Travel stats saved to {travel_path}")
    print(f"Energy table saved to {energy_path}")
    print(f"Zones saved to {zones_path_out}")


if __name__ == "__main__":
    main()
