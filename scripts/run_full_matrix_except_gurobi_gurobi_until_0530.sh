#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG_PATH="${1:-configs/ev_yellow_2025_11_eval_6_22_fcfs.yaml}"
OUTPUT_CSV="${2:-results/baseline_combinations/full_matrix_except_gurobi_gurobi_6_22.csv}"
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/envs/bsm/bin/python}"
CONDA_ENV="${CONDA_ENV:-}"

INVENTORIES="${INVENTORIES:-5,10,20}"
CHARGERS="${CHARGERS:-3,5,8}"
SWAP_CAPACITIES="${SWAP_CAPACITIES:-3,6,9}"
DEADLINE_LOCAL="${DEADLINE_LOCAL:-2026-03-17 05:30:00}"
DEADLINE_TZ="${DEADLINE_TZ:-America/Chicago}"

export PROJ_DATA="/opt/anaconda3/envs/bsm/share/proj"
export PROJ_LIB="/opt/anaconda3/envs/bsm/share/proj"

OUT_DIR="$(dirname "${OUTPUT_CSV}")"
TMP_DIR="${OUT_DIR}/_tmp_full_matrix_except_gurobi_gurobi_6_22"
mkdir -p "${TMP_DIR}"

declare -a COMBOS=(
  "heuristic_plus_fcfs heuristic fcfs"
  "heuristic_plus_gurobi heuristic gurobi"
  "algorithm_plus_fcfs gurobi fcfs"
  "ideal_plus_fcfs ideal fcfs"
  "ideal_plus_gurobi ideal gurobi"
)

if [[ -n "${CONDA_ENV}" ]]; then
  PY_CMD=(conda run -n "${CONDA_ENV}" python)
else
  PY_CMD=("${PYTHON_BIN}")
fi

deadline_epoch="$(TZ="${DEADLINE_TZ}" date -j -f "%Y-%m-%d %H:%M:%S" "${DEADLINE_LOCAL}" "+%s")"
echo "Run deadline (${DEADLINE_TZ}): ${DEADLINE_LOCAL} (epoch=${deadline_epoch})"

for combo in "${COMBOS[@]}"; do
  now_epoch="$(TZ="${DEADLINE_TZ}" date "+%s")"
  if [[ "${now_epoch}" -ge "${deadline_epoch}" ]]; then
    echo "Reached deadline, stop launching new combos."
    break
  fi

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
    --resume &
  sweep_pid=$!
  terminated_by_deadline=0
  while kill -0 "${sweep_pid}" 2>/dev/null; do
    now_epoch="$(TZ="${DEADLINE_TZ}" date "+%s")"
    if [[ "${now_epoch}" -ge "${deadline_epoch}" ]]; then
      echo "Reached deadline during ${name}, stopping pid=${sweep_pid}"
      kill -TERM "${sweep_pid}" 2>/dev/null || true
      sleep 2
      kill -KILL "${sweep_pid}" 2>/dev/null || true
      terminated_by_deadline=1
      break
    fi
    sleep 20
  done
  wait "${sweep_pid}" || true

  if [[ "${terminated_by_deadline}" == "1" ]]; then
    echo "Stopped by deadline. Keep partial progress and resume later."
    break
  fi

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
    raise RuntimeError("No rows were generated from except-gurobi-gurobi matrix sweep.")

fieldnames = sorted({k for row in rows for k in row.keys()})
output_csv.parent.mkdir(parents=True, exist_ok=True)
with output_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved matrix snapshot to: {output_csv}")
print(f"Rows: {len(rows)}")
PY
done

echo "Done: until deadline loop finished."
