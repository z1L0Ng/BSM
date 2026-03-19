#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

MODE="${1:-full}" # preflight | full
GROUPS_CSV="${GROUPS_CSV:-configs/eval_critical3_groups.csv}"
PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/envs/bsm/bin/python}"

if [[ "${MODE}" == "preflight" ]]; then
  CONFIG_PATH="${CONFIG_PATH:-configs/ev_yellow_2025_11_eval_6_22_strict30_smoke.yaml}"
  GROUP_IDS="${GROUP_IDS:-G2}"
  REQUIRE_ALL="${REQUIRE_ALL:-false}"
  TMP_DIR="${TMP_DIR:-results/baseline_combinations/_tmp_eval_critical3_6combo_preflight}"
  MATRIX_CSV="${MATRIX_CSV:-results/baseline_combinations/full_matrix_6combo_critical3_6_22_preflight.csv}"
  SUMMARY_MD="${SUMMARY_MD:-results/baseline_combinations/eval_summary_critical3_6combo_preflight.md}"
  SUMMARY_CSV="${SUMMARY_CSV:-results/baseline_combinations/eval_summary_critical3_6combo_preflight.csv}"
  FIG_DIR="${FIG_DIR:-results/baseline_combinations/figures_eval_critical3_6combo_preflight}"
else
  CONFIG_PATH="${CONFIG_PATH:-configs/ev_yellow_2025_11_eval_6_22_strict30.yaml}"
  GROUP_IDS="${GROUP_IDS:-}"
  REQUIRE_ALL="${REQUIRE_ALL:-true}"
  TMP_DIR="${TMP_DIR:-results/baseline_combinations/_tmp_eval_critical3_6combo}"
  MATRIX_CSV="${MATRIX_CSV:-results/baseline_combinations/full_matrix_6combo_critical3_6_22.csv}"
  SUMMARY_MD="${SUMMARY_MD:-results/baseline_combinations/eval_summary_critical3_6combo.md}"
  SUMMARY_CSV="${SUMMARY_CSV:-results/baseline_combinations/eval_summary_critical3_6combo.csv}"
  FIG_DIR="${FIG_DIR:-results/baseline_combinations/figures_eval_critical3_6combo}"
fi
FIG_REPORT_MD="${FIG_REPORT_MD:-${FIG_DIR}/README.md}"

PY_ENV_ROOT="$(cd "$(dirname "${PYTHON_BIN}")/.." && pwd)"
PROJ_DIR_DEFAULT="${PY_ENV_ROOT}/share/proj"
if [[ -d "${PROJ_DIR_DEFAULT}" ]]; then
  export PROJ_DATA="${PROJ_DIR_DEFAULT}"
  export PROJ_LIB="${PROJ_DIR_DEFAULT}"
else
  export PROJ_DATA="/opt/anaconda3/envs/bsm/share/proj"
  export PROJ_LIB="/opt/anaconda3/envs/bsm/share/proj"
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
mkdir -p "${MPLCONFIGDIR}"

echo "MODE=${MODE}"
echo "CONFIG=${CONFIG_PATH}"
echo "GROUPS_CSV=${GROUPS_CSV}"
echo "GROUP_IDS=${GROUP_IDS:-<all>}"

RUN_ARGS=(
  --groups-csv "${GROUPS_CSV}"
  --config "${CONFIG_PATH}"
  --python-bin "${PYTHON_BIN}"
  --tmp-dir "${TMP_DIR}"
  --output-csv "${MATRIX_CSV}"
  --group-ids "${GROUP_IDS}"
  --resume
)
if [[ "${REQUIRE_ALL}" == "true" ]]; then
  RUN_ARGS+=(--require-all)
else
  RUN_ARGS+=(--no-require-all)
fi

"${PYTHON_BIN}" scripts/run_eval_critical3_6combo.py \
  "${RUN_ARGS[@]}"

SUMMARY_ARGS=(
  --input-csv "${MATRIX_CSV}"
  --groups-csv "${GROUPS_CSV}"
  --output-md "${SUMMARY_MD}"
  --output-csv "${SUMMARY_CSV}"
)
if [[ "${REQUIRE_ALL}" == "true" ]]; then
  SUMMARY_ARGS+=(--strict-validate)
else
  SUMMARY_ARGS+=(--no-strict-validate)
fi

"${PYTHON_BIN}" scripts/summarize_eval_critical3.py \
  "${SUMMARY_ARGS[@]}"

"${PYTHON_BIN}" scripts/plot_eval_critical3_results.py \
  --input-csv "${MATRIX_CSV}" \
  --output-dir "${FIG_DIR}" \
  --report-md "${FIG_REPORT_MD}"

echo "Done."
