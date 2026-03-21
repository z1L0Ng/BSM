#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/envs/bsm/bin/python}"
PROJ_DIR="${PROJ_DIR:-/opt/anaconda3/envs/bsm/share/proj}"
GDAL_DIR="${GDAL_DIR:-/opt/anaconda3/envs/bsm/share/gdal}"
export PROJ_DATA="${PROJ_DIR}"
export PROJ_LIB="${PROJ_DIR}"
export GDAL_DATA="${GDAL_DIR}"
BASE_CFG="configs/ev_yellow_2025_11_eval_6_22_strict30.yaml"
DATES=("2025-11-01" "2025-11-05" "2025-11-09")
RESOURCE_GROUPS=(
  "G1 4 3 4"
  "G2 6 3 4"
  "G3 6 5 4"
)
COMBOS=(
  "algorithm_plus_gurobi gurobi gurobi"
  "heuristic_plus_fcfs heuristic fcfs"
  "ideal_plus_fcfs ideal fcfs"
)

if [[ ! -f "${BASE_CFG}" ]]; then
  echo "Missing base config: ${BASE_CFG}" >&2
  exit 1
fi

mkdir -p configs/generated_main_only
"${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import yaml

base = Path("configs/ev_yellow_2025_11_eval_6_22_strict30.yaml")
raw = yaml.safe_load(base.read_text(encoding="utf-8"))
dates = ["2025-11-01", "2025-11-05", "2025-11-09"]
out_dir = Path("configs/generated_main_only")
out_dir.mkdir(parents=True, exist_ok=True)

for d in dates:
    obj = dict(raw)
    obj["paths"] = dict(raw.get("paths", {}))
    obj["data"] = dict(raw.get("data", {}))
    obj["sim"] = dict(raw["sim"])
    obj["sim"]["sim_date"] = d
    # Keep data references semantically identical after moving config one level deeper.
    for section in ("paths", "data"):
        for k, v in list(obj.get(section, {}).items()):
            if isinstance(v, str) and v.startswith("../data"):
                obj[section][k] = v.replace("../data", "../../data", 1)
    out = out_dir / f"ev_eval_6_22_strict30_{d}.yaml"
    out.write_text(yaml.safe_dump(obj, sort_keys=False), encoding="utf-8")
    print(out)
PY

TS="$(date +"%Y%m%d_%H%M%S")"
ROOT="results/evaluation_runs/${TS}_main_only_3day"
mkdir -p "${ROOT}"/{raw,agg,logs}
echo "${ROOT}" > results/evaluation_runs/.latest_main_only_root
echo "ROOT=${ROOT}"

"${PYTHON_BIN}" - <<'PY'
import hashlib
import json
import subprocess
from pathlib import Path

root = Path(open("results/evaluation_runs/.latest_main_only_root", encoding="utf-8").read().strip())
cfgs = sorted(Path("configs/generated_main_only").glob("ev_eval_6_22_strict30_*.yaml"))
commit_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()

manifest = {
    "commit_hash": commit_hash,
    "configs": [],
    "scope": {
        "sim_start_hour": 6,
        "sim_end_hour": 22,
        "timestep_minutes": 15,
        "horizon": 64,
        "dates": ["2025-11-01", "2025-11-05", "2025-11-09"],
        "groups": [
            {"group_id": "G1", "inventory": 4, "chargers": 3, "swap_capacity": 4},
            {"group_id": "G2", "inventory": 6, "chargers": 3, "swap_capacity": 4},
            {"group_id": "G3", "inventory": 6, "chargers": 5, "swap_capacity": 4},
        ],
        "combos": ["algorithm_plus_gurobi", "heuristic_plus_fcfs", "ideal_plus_fcfs"],
    },
}
for p in cfgs:
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    manifest["configs"].append({"path": str(p), "sha256": digest})

(root / "agg" / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
print(root / "agg" / "run_manifest.json")
PY

run_case () {
  local cfg=$1
  local inv=$2
  local ch=$3
  local sw=$4
  local name=$5
  local rep=$6
  local chg=$7
  local out=$8
  local log=$9
  "${PYTHON_BIN}" scripts/run_param_sweep.py \
    --config "${cfg}" \
    --inventories "${inv}" \
    --chargers "${ch}" \
    --swap-capacities "${sw}" \
    --reposition-solver "${rep}" \
    --charging-solver "${chg}" \
    --output-csv "${out}" \
    --no-resume \
    > "${log}" 2>&1
}

SMOKE_CFG="configs/generated_main_only/ev_eval_6_22_strict30_2025-11-01.yaml"
for c in "${COMBOS[@]}"; do
  read -r name rep chg <<< "${c}"
  out="${ROOT}/raw/smoke_${name}_2025-11-01_G2.csv"
  log="${ROOT}/logs/smoke_${name}.log"
  run_case "${SMOKE_CFG}" 6 3 4 "${name}" "${rep}" "${chg}" "${out}" "${log}"
done

ROOT="${ROOT}" "${PYTHON_BIN}" - <<'PY'
import pandas as pd
import glob
import os
from pathlib import Path

root = Path(os.environ["ROOT"])
smoke_files = sorted(glob.glob(str(root / "raw" / "smoke_*.csv")))
if len(smoke_files) != 3:
    raise RuntimeError(f"Smoke file count mismatch: {len(smoke_files)}")

rows = []
for p in smoke_files:
    df = pd.read_csv(p)
    if len(df) != 1:
        raise RuntimeError(f"Smoke rows must be 1: {p} -> {len(df)}")
    row = df.iloc[0]
    for k in ["total_served", "battery_swap_success_ratio", "deadline_miss_ratio", "charging_deadline_miss_ratio"]:
        if pd.isna(row.get(k)):
            raise RuntimeError(f"Smoke KPI missing {k} in {p}")
    rows.append({"file": Path(p).name, "total_served": float(row["total_served"])})

algo_path = root / "raw" / "smoke_algorithm_plus_gurobi_2025-11-01_G2.csv"
algo = pd.read_csv(algo_path)
audit_cols = [c for c in algo.columns if c.startswith("reposition_step_") or c.startswith("reposition_event_")]
if not audit_cols:
    raise RuntimeError("Smoke audit fields missing for algorithm_plus_gurobi")

summary = [
    "# Stage A Summary",
    "",
    f"- Completed scenarios: {len(smoke_files)} / 3",
    "- Failed scenarios: 0",
    f"- Audit columns (algorithm): {len(audit_cols)}",
    "- Next: run full matrix (27 runs).",
]
(root / "agg" / "stage_A_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
print("smoke_ok")
PY

for d in "${DATES[@]}"; do
  cfg="configs/generated_main_only/ev_eval_6_22_strict30_${d}.yaml"
  for g in "${RESOURCE_GROUPS[@]}"; do
    read -r gid inv ch sw <<< "${g}"
    for c in "${COMBOS[@]}"; do
      read -r name rep chg <<< "${c}"
      out="${ROOT}/raw/${name}_${d}_${gid}.csv"
      log="${ROOT}/logs/${name}_${d}_${gid}.log"
      run_case "${cfg}" "${inv}" "${ch}" "${sw}" "${name}" "${rep}" "${chg}" "${out}" "${log}"
    done
  done
done

ROOT="${ROOT}" "${PYTHON_BIN}" - <<'PY'
import csv
import glob
import os
from pathlib import Path

root = os.environ["ROOT"]
files = sorted(glob.glob(f"{root}/raw/*.csv"))
rows = []
for p in files:
    name = Path(p).name
    if name.startswith("smoke_"):
        continue
    combo, date, gid = name[:-4].rsplit("_", 2)
    with open(p, newline="", encoding="utf-8") as f:
        recs = list(csv.DictReader(f))
    if len(recs) != 1:
        raise RuntimeError(f"Expect 1 row in {p}, got {len(recs)}")
    r = dict(recs[0])
    r["combo"] = combo
    r["sim_date"] = date
    r["group_id"] = gid
    rows.append(r)

if len(rows) != 27:
    raise RuntimeError(f"Main rows mismatch: {len(rows)} vs expected 27")

keys = sorted({k for r in rows for k in r})
out = f"{root}/agg/main_matrix_raw.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
print(out)
PY

ROOT="${ROOT}" "${PYTHON_BIN}" - <<'PY'
import os
import pandas as pd

root = os.environ["ROOT"]
df = pd.read_csv(f"{root}/agg/main_matrix_raw.csv")
if len(df) != 27:
    raise RuntimeError(f"Row count mismatch: got {len(df)}, expected 27")

if df[["sim_date", "group_id", "combo"]].duplicated().any():
    raise RuntimeError("Duplicate key rows detected")

for col, exp in [("sim_start_hour", 6), ("sim_end_hour", 22), ("effective_horizon", 64)]:
    if col not in df.columns:
        raise RuntimeError(f"Missing scope column: {col}")
    vals = sorted(pd.to_numeric(df[col], errors="coerce").dropna().unique().tolist())
    if vals != [exp]:
        raise RuntimeError(f"{col} scope mismatch: {vals} != {[exp]}")

required = [
    "total_served",
    "battery_swap_success_ratio",
    "deadline_miss_ratio",
    "charging_deadline_miss_ratio",
]
for c in required:
    if c not in df.columns:
        raise RuntimeError(f"Missing required KPI: {c}")
    if pd.to_numeric(df[c], errors="coerce").isna().any():
        raise RuntimeError(f"KPI contains NaN: {c}")

algo = df[df["combo"] == "algorithm_plus_gurobi"].copy()
heu = df[df["combo"] == "heuristic_plus_fcfs"].copy()
ide = df[df["combo"] == "ideal_plus_fcfs"].copy()

audit_cols = [c for c in algo.columns if c.startswith("reposition_step_") or c.startswith("reposition_event_")]
if not audit_cols:
    raise RuntimeError("Missing algorithm audit columns in full matrix")
if algo[audit_cols].isna().all(axis=1).any():
    raise RuntimeError("Algorithm audit fields are empty for some rows")

summary = (
    df.groupby("combo", as_index=False)[
        ["total_served", "battery_swap_success_ratio", "deadline_miss_ratio", "charging_deadline_miss_ratio"]
    ]
    .agg(["mean", "std"])
)
summary.columns = ["_".join([x for x in c if x]).strip("_") for c in summary.columns.to_flat_index()]
summary.to_csv(f"{root}/agg/main_summary.csv", index=False)

key = ["sim_date", "group_id"]
def paired_delta(a, b, bname):
    m = a.merge(b, on=key, suffixes=("_a", "_b"), how="inner")
    m["delta_total_served"] = m["total_served_a"] - m["total_served_b"]
    m["baseline"] = bname
    return m[["baseline", *key, "delta_total_served"]]

pd.concat(
    [paired_delta(algo, heu, "heuristic_plus_fcfs"), paired_delta(algo, ide, "ideal_plus_fcfs")],
    ignore_index=True,
).to_csv(f"{root}/agg/paired_deltas.csv", index=False)

wtl_rows = []
for bname, bdf in [("heuristic_plus_fcfs", heu), ("ideal_plus_fcfs", ide)]:
    m = algo.merge(bdf, on=key, suffixes=("_a", "_b"), how="inner")
    d = m["total_served_a"] - m["total_served_b"]
    wtl_rows.append({
        "baseline": bname,
        "wins": int((d > 0).sum()),
        "ties": int((d == 0).sum()),
        "losses": int((d < 0).sum()),
        "mean_delta_total_served": float(d.mean()),
        "std_delta_total_served": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
    })
pd.DataFrame(wtl_rows).to_csv(f"{root}/agg/win_tie_loss.csv", index=False)

from math import comb
def sign_test_two_sided(deltas):
    vals = [x for x in deltas if x != 0]
    n = len(vals)
    if n == 0:
        return 1.0, 0
    k = sum(1 for x in vals if x > 0)
    tail = min(k, n - k)
    p = 2.0 * sum(comb(n, i) for i in range(0, tail + 1)) / (2 ** n)
    return min(1.0, p), n

rows = []
for bname, bdf in [("heuristic_plus_fcfs", heu), ("ideal_plus_fcfs", ide)]:
    m = algo.merge(bdf, on=key, suffixes=("_a", "_b"), how="inner")
    d = (m["total_served_a"] - m["total_served_b"]).tolist()
    p_sign, n_eff = sign_test_two_sided(d)
    s = pd.Series(d, dtype=float)
    rows.append({
        "baseline": bname,
        "n_pairs": int(len(d)),
        "n_nonzero_pairs": int(n_eff),
        "mean_delta_total_served": float(s.mean()),
        "median_delta_total_served": float(s.median()),
        "pvalue_sign_test_two_sided": float(p_sign),
    })
pd.DataFrame(rows).to_csv(f"{root}/agg/paired_stats.csv", index=False)

stage_b = [
    "# Stage B Summary",
    "",
    f"- Completed scenarios: {len(df)} / 27",
    "- Failed scenarios: 0",
    f"- Mean total_served (algorithm_plus_gurobi): {algo['total_served'].mean():.2f}",
    f"- Mean total_served (heuristic_plus_fcfs): {heu['total_served'].mean():.2f}",
    f"- Mean total_served (ideal_plus_fcfs): {ide['total_served'].mean():.2f}",
    "- Next: review aggregated tables and logs.",
]
with open(f"{root}/agg/stage_B_summary.md", "w", encoding="utf-8") as f:
    f.write("\n".join(stage_b) + "\n")
print("aggregate_ok")
PY

ROOT="${ROOT}" "${PYTHON_BIN}" - <<'PY'
import os
import pandas as pd

root = os.environ["ROOT"]
df = pd.read_csv(f"{root}/agg/main_matrix_raw.csv")
s = pd.read_csv(f"{root}/agg/main_summary.csv")
p = pd.read_csv(f"{root}/agg/paired_stats.csv")

lines = [
    "# Main Findings (Main-Only 3-Day Evaluation)",
    "",
    f"- Total scenarios completed: {len(df)} / 27",
    f"- Date coverage: {', '.join(sorted(df['sim_date'].astype(str).unique().tolist()))}",
    f"- Group coverage: {', '.join(sorted(df['group_id'].astype(str).unique().tolist()))}",
]
for _, row in s.iterrows():
    lines.append(
        f"- {row['combo']}: total_served_mean={row['total_served_mean']:.2f}, "
        f"swap_success_mean={row['battery_swap_success_ratio_mean']:.4f}, "
        f"deadline_miss_mean={row['deadline_miss_ratio_mean']:.4f}"
    )
for _, row in p.iterrows():
    lines.append(
        f"- algorithm_plus_gurobi vs {row['baseline']}: "
        f"mean_delta_total_served={row['mean_delta_total_served']:.2f}, "
        f"p_sign={row['pvalue_sign_test_two_sided']:.4f} (n={int(row['n_pairs'])})"
    )

with open(f"{root}/README_main_findings.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"{root}/README_main_findings.md")
PY

echo "DONE ROOT=${ROOT}"
