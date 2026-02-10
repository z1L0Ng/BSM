from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


def clean_yellow_tripdata(
    raw_path: str | Path,
    output_path: str | Path,
    cols: Dict[str, str],
    filters: Dict[str, float],
) -> Path:
    raw_path = Path(raw_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    use_cols = [
        cols["pickup_col"],
        cols["dropoff_col"],
        cols["pu_col"],
        cols["do_col"],
        cols["passenger_col"],
        cols["distance_col"],
        cols["fare_col"],
        cols["total_col"],
    ]

    df = pd.read_parquet(raw_path, columns=use_cols)

    df = df.rename(
        columns={
            cols["pickup_col"]: "pickup_ts",
            cols["dropoff_col"]: "dropoff_ts",
            cols["pu_col"]: "pu_zone",
            cols["do_col"]: "do_zone",
            cols["passenger_col"]: "passengers",
            cols["distance_col"]: "distance_mi",
            cols["fare_col"]: "fare_amount",
            cols["total_col"]: "total_amount",
        }
    )

    df["pickup_ts"] = pd.to_datetime(df["pickup_ts"], errors="coerce")
    df["dropoff_ts"] = pd.to_datetime(df["dropoff_ts"], errors="coerce")

    df = df.dropna(subset=["pickup_ts", "dropoff_ts", "pu_zone", "do_zone"])

    df["duration_min"] = (df["dropoff_ts"] - df["pickup_ts"]).dt.total_seconds() / 60
    df["speed_mph"] = df["distance_mi"] / (df["duration_min"] / 60)

    df = df[(df["duration_min"] >= filters["min_duration_min"]) & (df["duration_min"] <= filters["max_duration_min"])]
    df = df[(df["speed_mph"] >= filters["min_speed_mph"]) & (df["speed_mph"] <= filters["max_speed_mph"])]

    df["pickup_date"] = df["pickup_ts"].dt.date
    df["pickup_hour"] = df["pickup_ts"].dt.hour

    df.to_parquet(output_path, index=False)
    return output_path
