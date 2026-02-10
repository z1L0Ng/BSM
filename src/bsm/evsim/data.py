from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class DataBundle:
    zones: List[int]
    time_bins: List[pd.Timestamp]
    demand: pd.DataFrame
    od: pd.DataFrame
    travel_stats: pd.DataFrame


class DataProcessor:
    def __init__(self, config: Dict):
        self.cfg = config

    def load_trip_data(self, path: str | Path) -> pd.DataFrame:
        cols = self.cfg["dataset"]["columns"]
        df = pd.read_parquet(path, columns=cols)
        df = df.rename(
            columns={
                "tpep_pickup_datetime": "pickup_ts",
                "tpep_dropoff_datetime": "dropoff_ts",
                "PULocationID": "pu_zone",
                "DOLocationID": "do_zone",
                "trip_distance": "distance_mi",
            }
        )
        df["pickup_ts"] = pd.to_datetime(df["pickup_ts"], errors="coerce")
        df["dropoff_ts"] = pd.to_datetime(df["dropoff_ts"], errors="coerce")
        df = df.dropna(subset=["pickup_ts", "dropoff_ts", "pu_zone", "do_zone", "distance_mi"])
        df["duration_min"] = (df["dropoff_ts"] - df["pickup_ts"]).dt.total_seconds() / 60
        return df

    def load_zone_lookup(self, path: str | Path) -> pd.DataFrame:
        zones = pd.read_csv(path)
        return zones

    def filter_manhattan(self, trips: pd.DataFrame, zones: pd.DataFrame) -> pd.DataFrame:
        manhattan = zones[zones["Borough"] == "Manhattan"]["LocationID"].tolist()
        return trips[trips["pu_zone"].isin(manhattan) & trips["do_zone"].isin(manhattan)]

    def select_top_zones(self, trips: pd.DataFrame) -> List[int]:
        top_k = int(self.cfg["processing"]["top_k_zones"])
        return trips["pu_zone"].value_counts().head(top_k).index.astype(int).tolist()

    def time_bin(self, trips: pd.DataFrame) -> pd.DataFrame:
        freq = int(self.cfg["processing"]["time_bin_minutes"])
        trips = trips.copy()
        trips["time_bin"] = trips["pickup_ts"].dt.floor(f"{freq}min")
        return trips

    def build_demand_od(self, trips: pd.DataFrame, zones: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        trips = trips[trips["pu_zone"].isin(zones) & trips["do_zone"].isin(zones)]

        demand = (
            trips.groupby(["time_bin", "pu_zone"], as_index=False)
            .size()
            .rename(columns={"size": "demand"})
        )

        od_counts = (
            trips.groupby(["time_bin", "pu_zone", "do_zone"], as_index=False)
            .size()
            .rename(columns={"size": "trips"})
        )

        totals = od_counts.groupby(["time_bin", "pu_zone"], as_index=False)["trips"].sum()
        od = od_counts.merge(totals, on=["time_bin", "pu_zone"], suffixes=("", "_total"))
        od["prob"] = od["trips"] / od["trips_total"]

        travel_stats = (
            trips.groupby(["pu_zone", "do_zone"], as_index=False)
            .agg(mean_distance_mi=("distance_mi", "mean"), mean_duration_min=("duration_min", "mean"))
        )

        return demand, od, travel_stats

    def process(self, trips_path: str | Path, zones_path: str | Path) -> DataBundle:
        trips = self.load_trip_data(trips_path)
        zones = self.load_zone_lookup(zones_path)
        trips = self.filter_manhattan(trips, zones)
        trips = self.time_bin(trips)
        top_zones = self.select_top_zones(trips)

        demand, od, travel_stats = self.build_demand_od(trips, top_zones)
        time_bins = sorted(demand["time_bin"].unique())
        max_bins = self.cfg["processing"].get("max_time_bins")
        if max_bins:
            time_bins = time_bins[: int(max_bins)]
            demand = demand[demand["time_bin"].isin(time_bins)]
            od = od[od["time_bin"].isin(time_bins)]

        return DataBundle(
            zones=top_zones,
            time_bins=time_bins,
            demand=demand,
            od=od,
            travel_stats=travel_stats,
        )


def build_energy_table(travel_stats: pd.DataFrame, kwh_per_mile: float) -> pd.DataFrame:
    table = travel_stats.copy()
    table["energy_kwh"] = table["mean_distance_mi"] * kwh_per_mile
    return table


def make_od_tensor(od: pd.DataFrame, zones: List[int], time_bins: List[pd.Timestamp]) -> np.ndarray:
    idx = {z: i for i, z in enumerate(zones)}
    tidx = {t: i for i, t in enumerate(time_bins)}
    n = len(zones)
    t = len(time_bins)
    tensor = np.zeros((t, n, n), dtype=float)

    for row in od.itertuples(index=False):
        ti = tidx[row.time_bin]
        i = idx[int(row.pu_zone)]
        j = idx[int(row.do_zone)]
        tensor[ti, i, j] = float(row.prob)

    return tensor


def make_demand_matrix(demand: pd.DataFrame, zones: List[int], time_bins: List[pd.Timestamp]) -> np.ndarray:
    idx = {z: i for i, z in enumerate(zones)}
    tidx = {t: i for i, t in enumerate(time_bins)}
    t = len(time_bins)
    n = len(zones)
    mat = np.zeros((t, n), dtype=float)

    for row in demand.itertuples(index=False):
        mat[tidx[row.time_bin], idx[int(row.pu_zone)]] = float(row.demand)

    return mat
