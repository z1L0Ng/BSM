from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from bsm.io import ensure_dirs, load_config
from bsm.processing.clean import clean_yellow_tripdata
from bsm.processing.od_stats import build_od_stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    raw_path = Path(cfg.paths["raw_dir"]) / cfg.dataset["filename"]
    processed_path = Path(cfg.paths["processed_dir"]) / f"{cfg.dataset['name']}_{cfg.dataset['year']}-{cfg.dataset['month']:02d}_clean.parquet"
    interim_od_path = Path(cfg.paths["interim_dir"]) / f"{cfg.dataset['name']}_{cfg.dataset['year']}-{cfg.dataset['month']:02d}_od_stats.parquet"

    clean_yellow_tripdata(
        raw_path=raw_path,
        output_path=processed_path,
        cols=cfg.dataset,
        filters=cfg.processing,
    )

    build_od_stats(processed_path=processed_path, output_path=interim_od_path)

    print(f"Processed data saved to {processed_path}")
    print(f"OD stats saved to {interim_od_path}")


if __name__ == "__main__":
    main()
