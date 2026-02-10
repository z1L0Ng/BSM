from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class Config:
    data: Dict[str, Any]

    @property
    def paths(self) -> Dict[str, Any]:
        return self.data.get("paths", {})

    @property
    def dataset(self) -> Dict[str, Any]:
        return self.data.get("dataset", {})

    @property
    def processing(self) -> Dict[str, Any]:
        return self.data.get("processing", {})

    @property
    def simulation(self) -> Dict[str, Any]:
        return self.data.get("simulation", {})

    @property
    def metrics(self) -> Dict[str, Any]:
        return self.data.get("metrics", {})


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(data=data)


def ensure_dirs(cfg: Config) -> None:
    for key in ["raw_dir", "processed_dir", "interim_dir", "reports_dir"]:
        value = cfg.paths.get(key)
        if value:
            Path(value).mkdir(parents=True, exist_ok=True)
