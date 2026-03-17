#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG_PATH="${1:-configs/ev_yellow_2025_11_eval_6_22_fcfs.yaml}"
OUTPUT_CSV="${2:-results/baseline_combinations/stress_gate48_nogurobi_6_22.csv}"
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/envs/bsm/bin/python}"
CONDA_ENV="${CONDA_ENV:-}"

export PROJ_DATA="/opt/anaconda3/envs/bsm/share/proj"
export PROJ_LIB="/opt/anaconda3/envs/bsm/share/proj"

INVENTORIES="5,10"
CHARGERS="3,8"
SWAP_CAPACITIES="3,6"

OUT_DIR="$(dirname "${OUTPUT_CSV}")"
TMP_DIR="${OUT_DIR}/_tmp_stress_gate48_nogurobi"
mkdir -p "${TMP_DIR}"
rm -f "${TMP_DIR}"/*.csv

INCLUDE_GUROBI_REPOSITION="${INCLUDE_GUROBI_REPOSITION:-0}"
INCLUDE_IDEAL="${INCLUDE_IDEAL:-0}"

declare -a COMBOS=(
  "heuristic_plus_fcfs heuristic fcfs"
)

if [[ "${INCLUDE_IDEAL}" == "1" ]]; then
  COMBOS+=("ideal_plus_fcfs ideal fcfs")
fi

if [[ "${INCLUDE_GUROBI_REPOSITION}" == "1" ]]; then
  COMBOS+=("algorithm_plus_fcfs gurobi fcfs")
fi

if [[ -n "${CONDA_ENV}" ]]; then
  PY_CMD=(conda run -n "${CONDA_ENV}" python)
else
  PY_CMD=("${PYTHON_BIN}")
fi

for combo in "${COMBOS[@]}"; do
  read -r name reposition_solver charging_solver <<< "${combo}"
  out_file="${TMP_DIR}/${name}.csv"
  echo "Running ${name}: reposition=${reposition_solver}, charging=${charging_solver}"
  "${PY_CMD[@]}" scripts/run_param_sweep.py \
    --config "${CONFIG_PATH}" \
    --inventories "${INVENTORIES}" \
    --chargers "${CHARGERS}" \
    --swap-capacities "${SWAP_CAPACITIES}" \
    --reposition-solver "${reposition_solver}" \
    --charging-solver "${charging_solver}" \
    --output-csv "${out_file}" \
    --resume
done

TMP_DIR="${TMP_DIR}" OUTPUT_CSV="${OUTPUT_CSV}" "${PY_CMD[@]}" - << 'PY'
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
    raise RuntimeError("No rows were generated from stress 48-run no-gurobi gate sweep.")

fieldnames = sorted({k for row in rows for k in row.keys()})
output_csv.parent.mkdir(parents=True, exist_ok=True)
with output_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved 48-run no-gurobi gate matrix to: {output_csv}")
print(f"Rows: {len(rows)}")
PY
