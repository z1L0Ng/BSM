from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import HTTPError, URLError

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from bsm.io import ensure_dirs, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    if "url_template" in cfg.dataset:
        url_template = cfg.dataset["url_template"]
        filename_template = cfg.dataset["filename_template"]
        years = [cfg.dataset["year"]] + cfg.dataset.get("fallback_years", [])
        url = None
        filename = None
        for year in years:
            candidate_url = url_template.format(year=year, month=cfg.dataset["month"])
            candidate_filename = filename_template.format(year=year, month=cfg.dataset["month"])
            try:
                print(f"Checking {candidate_url}")
                urlretrieve(candidate_url, Path(cfg.paths["raw_dir"]) / candidate_filename)
                url = candidate_url
                filename = candidate_filename
                cfg.dataset["year"] = year
                break
            except (HTTPError, URLError):
                continue
        if url is None or filename is None:
            raise RuntimeError("Failed to download any fallback dataset.")
    else:
        url = cfg.dataset["url"]
        filename = cfg.dataset["filename"]
    raw_dir = Path(cfg.paths["raw_dir"])
    dest = raw_dir / filename

    if dest.exists():
        print(f"File already exists: {dest}")
    else:
        print(f"Downloading {url} -> {dest}")
        urlretrieve(url, dest)
        print("Done")

    zone_cfg = cfg.data.get("zone_lookup")
    if zone_cfg:
        zone_path = raw_dir / zone_cfg["filename"]
        if not zone_path.exists():
            print(f"Downloading {zone_cfg['url']} -> {zone_path}")
            urlretrieve(zone_cfg["url"], zone_path)
            print("Done")


if __name__ == "__main__":
    main()
