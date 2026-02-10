from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_od_stats(processed_path: str | Path, output_path: str | Path) -> Path:
    processed_path = Path(processed_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(processed_path, columns=["pu_zone", "do_zone", "duration_min", "distance_mi"])
    stats = (
        df.groupby(["pu_zone", "do_zone"], as_index=False)
        .agg(
            mean_duration_min=("duration_min", "mean"),
            median_duration_min=("duration_min", "median"),
            mean_distance_mi=("distance_mi", "mean"),
            trips=("duration_min", "size"),
        )
    )
    stats.to_parquet(output_path, index=False)
    return output_path
