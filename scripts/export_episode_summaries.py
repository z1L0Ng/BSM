from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_summary_payload(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.get("summary", data)
    payload = {
        "run_id": data.get("run_id", path.parent.name),
        "generated_at": data.get("generated_at"),
        "config": data.get("config"),
        "schema_version": data.get("schema_version", "v1"),
    }
    if isinstance(summary, dict):
        payload.update(summary)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-dir", default="results/episodes")
    parser.add_argument("--output-csv", default="results/episodes/summary_index.csv")
    args = parser.parse_args()

    episodes_dir = Path(args.episodes_dir)
    rows: list[dict] = []

    for summary_path in sorted(episodes_dir.glob("*/summary.json")):
        rows.append(load_summary_payload(summary_path))

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        output_csv.write_text("", encoding="utf-8")
        print(f"No episode summaries found under {episodes_dir}")
        return

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    main()
