#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG_PATH="${1:-configs/ev_yellow_2025_11_fcfs.yaml}"
OUTPUT_CSV="${2:-results/baseline_combinations/stress_full_matrix_6combo.csv}"
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/envs/bsm/bin/python}"

export PROJ_DATA="/opt/anaconda3/envs/bsm/share/proj"
export PROJ_LIB="/opt/anaconda3/envs/bsm/share/proj"

INVENTORIES="5,10,20"
CHARGERS="3,5,8"
SWAP_CAPACITIES="3,6,9"

OUT_DIR="$(dirname "${OUTPUT_CSV}")"
TMP_DIR="${OUT_DIR}/_tmp_stress_6combo"
mkdir -p "${TMP_DIR}"

declare -a COMBOS=(
  "algorithm_plus_fcfs gurobi fcfs"
  "heuristic_plus_fcfs heuristic fcfs"
  "ideal_plus_fcfs ideal fcfs"
  "algorithm_plus_gurobi gurobi gurobi"
  "heuristic_plus_gurobi heuristic gurobi"
  "ideal_plus_gurobi ideal gurobi"
)

for combo in "${COMBOS[@]}"; do
  read -r name reposition_solver charging_solver <<< "${combo}"
  out_file="${TMP_DIR}/${name}.csv"
  echo "Running ${name}: reposition=${reposition_solver}, charging=${charging_solver}"
  "${PYTHON_BIN}" scripts/run_param_sweep.py \
    --config "${CONFIG_PATH}" \
    --inventories "${INVENTORIES}" \
    --chargers "${CHARGERS}" \
    --swap-capacities "${SWAP_CAPACITIES}" \
    --reposition-solver "${reposition_solver}" \
    --charging-solver "${charging_solver}" \
    --output-csv "${out_file}"
done

TMP_DIR="${TMP_DIR}" OUTPUT_CSV="${OUTPUT_CSV}" "${PYTHON_BIN}" - << 'PY'
import csv
import os
from pathlib import Path

tmp_dir = Path(os.environ["TMP_DIR"])
output_csv = Path(os.environ["OUTPUT_CSV"])

rows = []
for path in sorted(tmp_dir.glob("*.csv")):
    combo = path.stem
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["combo"] = combo
            rows.append(row)

if not rows:
    raise RuntimeError("No rows were generated from stress 6-combo sweep.")

fieldnames = sorted({k for row in rows for k in row.keys()})
output_csv.parent.mkdir(parents=True, exist_ok=True)
with output_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved 6-combo stress matrix to: {output_csv}")
print(f"Rows: {len(rows)}")
PY
