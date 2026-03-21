#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <RUN_ROOT> [CONFIG_PATH] [SCENARIO_CSV] [OLD_MATRIX_CSV]" >&2
  exit 2
fi

ROOT="$1"
CFG="${2:-configs/ev_yellow_2025_11_eval_6_22_day1_recal_v1.yaml}"
SCENARIO_SRC="${3:-results/evaluation_runs/20260319_093539_day1_12scenarios/agg/scenario_list_day1_12.csv}"
OLD_MATRIX="${4:-results/evaluation_runs/20260319_093539_day1_12scenarios/agg/full_matrix_day1_12scenarios.csv}"

SCENARIO_CSV="$ROOT/agg/scenario_list_day1_12.csv"
STATUS="$ROOT/agg/run_status_recal_v1.csv"
FULL_MATRIX="$ROOT/agg/full_matrix_gurobi_gurobi_recal_v1.csv"
COMPARE_CSV="$ROOT/agg/compare_vs_prev_alg_gurobi.csv"
VALIDATION_JSON="$ROOT/agg/validation_recal_v1.json"
SUMMARY_MD="$ROOT/agg/compare_summary_recal_v1.md"
FAILURES_CSV="$ROOT/agg/failed_cases_recal_v1.csv"
MANIFEST_JSON="$ROOT/agg/run_manifest_recal_v1.json"

export PROJ_DATA=/opt/anaconda3/envs/bsm/share/proj
export PROJ_LIB=/opt/anaconda3/envs/bsm/share/proj
export MPLCONFIGDIR=/tmp/matplotlib
mkdir -p "$MPLCONFIGDIR"
mkdir -p "$ROOT/full" "$ROOT/agg" "$ROOT/logs"

if [[ ! -f "$SCENARIO_SRC" ]]; then
  echo "Missing scenario list: $SCENARIO_SRC" >&2
  exit 2
fi
if [[ ! -f "$OLD_MATRIX" ]]; then
  echo "Missing old matrix: $OLD_MATRIX" >&2
  exit 2
fi

cp "$SCENARIO_SRC" "$SCENARIO_CSV"
echo "scenario_id,inventory_per_station,chargers_per_station,swap_capacity_per_station,combo,reposition_solver,charging_solver,output_csv,log_path,exit_code,row_count" > "$STATUS"

run_case() {
  local sid="$1" inv="$2" chg="$3" sw="$4"
  local name="algorithm_plus_gurobi"
  local rep="gurobi"
  local chs="gurobi"
  local out="$ROOT/full/${sid}__${name}.csv"
  local log="$ROOT/logs/${sid}__${name}.log"
  echo "[$(date +%H:%M:%S)] ${sid} ${name} (inv=${inv},chg=${chg},sw=${sw})"
  set +e
  /opt/anaconda3/envs/bsm/bin/python scripts/run_param_sweep.py \
    --config "$CFG" \
    --inventories "$inv" \
    --chargers "$chg" \
    --swap-capacities "$sw" \
    --reposition-solver "$rep" \
    --charging-solver "$chs" \
    --output-csv "$out" \
    --no-resume > "$log" 2>&1
  code=$?
  set -e
  rows=0
  if [[ -f "$out" ]]; then
    rows=$(($(wc -l < "$out") - 1))
    if [[ "$rows" -lt 0 ]]; then rows=0; fi
  fi
  echo "${sid},${inv},${chg},${sw},${name},${rep},${chs},${out},${log},${code},${rows}" >> "$STATUS"
}

while IFS=, read -r sid inv chg sw; do
  [[ "$sid" == "scenario_id" ]] && continue
  run_case "$sid" "$inv" "$chg" "$sw"
done < "$SCENARIO_CSV"

/opt/anaconda3/envs/bsm/bin/python - <<'PY' "$STATUS" "$FULL_MATRIX" "$COMPARE_CSV" "$VALIDATION_JSON" "$SUMMARY_MD" "$FAILURES_CSV" "$OLD_MATRIX" "$CFG" "$MANIFEST_JSON"
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

(
    status_path,
    full_path,
    compare_path,
    validation_path,
    summary_md,
    failures_path,
    old_matrix_path,
    cfg_path,
    manifest_path,
) = [Path(x) for x in sys.argv[1:10]]

status_df = pd.read_csv(status_path)
failed = status_df[(status_df["exit_code"] != 0) | (status_df["row_count"] <= 0)].copy()
failed.to_csv(failures_path, index=False)

rows = []
for r in status_df.itertuples(index=False):
    p = Path(r.output_csv)
    if not p.exists():
        continue
    try:
        df = pd.read_csv(p)
    except Exception:
        continue
    if df.empty:
        continue
    row = df.iloc[0].to_dict()
    row["scenario_id"] = r.scenario_id
    row["combo"] = r.combo
    row["reposition_solver"] = r.reposition_solver
    row["charging_solver"] = r.charging_solver
    row["inventory_per_station"] = int(r.inventory_per_station)
    row["chargers_per_station"] = int(r.chargers_per_station)
    row["swap_capacity_per_station"] = int(r.swap_capacity_per_station)
    row["sim_date"] = "2025-11-01"
    rows.append(row)

full_df = pd.DataFrame(rows)
if not full_df.empty:
    full_df = full_df.sort_values(["scenario_id"]).reset_index(drop=True)
full_df.to_csv(full_path, index=False)

kpi_cols = [
    "total_served",
    "battery_swap_success_ratio",
    "deadline_miss_ratio",
    "charging_deadline_miss_ratio",
    "total_waiting_vehicles",
    "total_idle_moves",
    "total_idle_driving_distance",
    "max_station_total_power_kw",
]
key_cols = [
    "scenario_id",
    "inventory_per_station",
    "chargers_per_station",
    "swap_capacity_per_station",
]

old_all = pd.read_csv(old_matrix_path)
old_alg = old_all[old_all["combo"].eq("algorithm_plus_gurobi")].copy()
old_alg = old_alg[key_cols + [c for c in kpi_cols if c in old_alg.columns]].copy()
new_alg = full_df[key_cols + [c for c in kpi_cols if c in full_df.columns]].copy()
old_alg = old_alg.rename(columns={c: f"old_{c}" for c in kpi_cols if c in old_alg.columns})
new_alg = new_alg.rename(columns={c: f"new_{c}" for c in kpi_cols if c in new_alg.columns})
compare_df = old_alg.merge(new_alg, on=key_cols, how="outer")
for c in kpi_cols:
    oc = f"old_{c}"
    nc = f"new_{c}"
    dc = f"delta_recal_minus_old__{c}"
    if oc in compare_df.columns and nc in compare_df.columns:
        compare_df[dc] = compare_df[nc] - compare_df[oc]
compare_df = compare_df.sort_values(["scenario_id"]).reset_index(drop=True)
compare_df.to_csv(compare_path, index=False)

expected_rows = 12
unique_ok = False
if not full_df.empty:
    unique_ok = full_df[["scenario_id", "combo"]].drop_duplicates().shape[0] == len(full_df)
hour_ok = False
if not full_df.empty and {"sim_start_hour", "sim_end_hour", "effective_horizon"}.issubset(full_df.columns):
    hour_ok = (
        full_df["sim_start_hour"].astype(float).eq(6).all()
        and full_df["sim_end_hour"].astype(float).eq(22).all()
        and full_df["effective_horizon"].astype(float).eq(64).all()
    )
audit_cols = [c for c in full_df.columns if c.startswith("reposition_step_") or c.startswith("reposition_event_")]
audit_ok = bool(audit_cols) and (not full_df[audit_cols].isna().any().any())

bonus_zero_ok = bool(
    "reposition_low_energy_swap_bonus" in full_df.columns
    and full_df["reposition_low_energy_swap_bonus"].astype(float).eq(0.0).all()
)
idle_weight_ok = bool(
    "reposition_idle_cost_weight" in full_df.columns
    and full_df["reposition_idle_cost_weight"].astype(float).eq(0.15).all()
)
gamma_ok = bool(
    "reposition_service_discount_gamma" in full_df.columns
    and full_df["reposition_service_discount_gamma"].astype(float).eq(1.0).all()
)

validation = {
    "expected_rows": expected_rows,
    "actual_rows": int(len(full_df)),
    "row_count_ok": int(len(full_df)) == expected_rows,
    "scenario_combo_unique_ok": bool(unique_ok),
    "time_window_horizon_ok": bool(hour_ok),
    "sim_date_unique": sorted(full_df["sim_date"].dropna().astype(str).unique().tolist())
    if not full_df.empty and "sim_date" in full_df.columns
    else [],
    "audit_ok_for_algorithm_plus_gurobi": bool(audit_ok),
    "audit_columns_checked": audit_cols,
    "failed_case_count": int(len(failed)),
    "failed_cases_csv": str(failures_path),
    "old_compare_rows": int(len(compare_df)),
    "old_compare_coverage_ok": int(len(compare_df)) == expected_rows,
    "bonus_zero_ok": bonus_zero_ok,
    "idle_weight_015_ok": idle_weight_ok,
    "service_discount_gamma_1_ok": gamma_ok,
}
validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")

direction = {
    "total_served": "higher",
    "battery_swap_success_ratio": "higher",
    "deadline_miss_ratio": "lower",
    "charging_deadline_miss_ratio": "lower",
    "total_waiting_vehicles": "lower",
    "total_idle_moves": "lower",
    "total_idle_driving_distance": "lower",
    "max_station_total_power_kw": "lower",
}

summary_lines = [
    "# Recal_v1 Comparison Summary",
    "",
    f"- New rows: {len(full_df)}/{expected_rows}",
    f"- Failed cases: {len(failed)}",
    f"- Validation JSON: `{validation_path}`",
    f"- Compare CSV: `{compare_path}`",
    "",
    "## KPI Delta (recal_v1 - old algorithm*gurobi)",
]
for metric in kpi_cols:
    dcol = f"delta_recal_minus_old__{metric}"
    if dcol not in compare_df.columns:
        continue
    s = compare_df[dcol].dropna()
    if s.empty:
        continue
    if direction.get(metric) == "higher":
        improve = int((s > 0).sum())
        worse = int((s < 0).sum())
    else:
        improve = int((s < 0).sum())
        worse = int((s > 0).sum())
    summary_lines.append(
        f"- `{metric}`: mean_delta={s.mean():.6f}, median_delta={s.median():.6f}, improve/worse/tie={improve}/{worse}/{int((s==0).sum())}"
    )
Path(summary_md).write_text("\n".join(summary_lines), encoding="utf-8")

cfg_sha = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
try:
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
except Exception:
    git_head = ""
manifest = {
    "config_path": str(cfg_path),
    "config_sha256": cfg_sha,
    "git_commit": git_head,
    "old_matrix_path": str(old_matrix_path),
    "status_csv": str(status_path),
    "full_matrix_csv": str(full_path),
    "compare_csv": str(compare_path),
}
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

print(full_path)
print(compare_path)
print(validation_path)
print(summary_md)
print(manifest_path)
PY

echo "=== RECAL_V1 DONE ==="
