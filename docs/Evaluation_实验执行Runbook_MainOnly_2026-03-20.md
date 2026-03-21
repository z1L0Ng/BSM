# Evaluation 实验执行 Runbook（Main-Only, 2026-03-20）

适用范围：正文主实验（不含附录扩展目标）  
目标：在统一口径下完成 baseline 对比，并输出可直接写入 `Baseline Evaluation Setup` 与 `Primary Results` 的结果。

---

## 0. 本次版本约束（重要）

- 不执行附录扩展目标（`paper_plus_swap_penalty`）相关实验。
- 仅执行主叙事三方法：
  - `algorithm_plus_gurobi`（`reposition=gurobi`, `charging=gurobi`）
  - `heuristic_plus_fcfs`（`reposition=heuristic`, `charging=fcfs`）
  - `ideal_plus_fcfs`（`reposition=ideal`, `charging=fcfs`）
- 固定评估口径：`06:00-22:00`, `15-min`, `horizon=64`。
- 重复日期固定为 3 天：`2025-11-01`, `2025-11-05`, `2025-11-09`。

---

## 1. Git 与环境同步检查

在仓库根目录执行：

```bash
git fetch --prune origin
git status --short --branch
git rev-parse --short HEAD
git rev-parse --short origin/main
```

通过条件：
- 工作区干净（无未提交改动）。
- 记录 `HEAD` commit 到实验日志。

推荐 Python：
- `PYTHON_BIN=/opt/anaconda3/envs/bsm/bin/python`

---

## 2. 实验矩阵定义（Main-Only）

### 2.1 资源组（Critical3）

使用 `configs/eval_critical3_groups.csv`：

- `G1`: inventory=4, chargers=3, swap=4
- `G2`: inventory=6, chargers=3, swap=4
- `G3`: inventory=6, chargers=5, swap=4

### 2.2 运行规模

- 日期：3
- 资源组：3
- 方法：3
- 总运行数：`3 x 3 x 3 = 27`（每个 run 对应一个 date+group+method）

---

## 3. 生成 3 天配置文件

基配置：`configs/ev_yellow_2025_11_eval_6_22_strict30.yaml`

```bash
mkdir -p configs/generated_main_only
python - <<'PY'
from pathlib import Path
import yaml

base = Path("configs/ev_yellow_2025_11_eval_6_22_strict30.yaml")
raw = yaml.safe_load(base.read_text(encoding="utf-8"))
dates = ["2025-11-01", "2025-11-05", "2025-11-09"]
out_dir = Path("configs/generated_main_only")
out_dir.mkdir(parents=True, exist_ok=True)

for d in dates:
    obj = dict(raw)
    obj["sim"] = dict(raw["sim"])
    obj["sim"]["sim_date"] = d
    out = out_dir / f"ev_eval_6_22_strict30_{d}.yaml"
    out.write_text(yaml.safe_dump(obj, sort_keys=False), encoding="utf-8")
    print(out)
PY
```

---

## 4. 输出目录与运行日志

```bash
TS=$(date +"%Y%m%d_%H%M%S")
ROOT="results/evaluation_runs/${TS}_main_only_3day"
mkdir -p "${ROOT}"/{raw,agg,logs}
echo "ROOT=${ROOT}"
```

---

## 5. Smoke Gate（先过门）

先只跑 `2025-11-01` + `G2` + 三方法，确认流程、字段、审计列齐全。

```bash
set -euo pipefail
PYTHON_BIN=${PYTHON_BIN:-/opt/anaconda3/envs/bsm/bin/python}
CFG="configs/generated_main_only/ev_eval_6_22_strict30_2025-11-01.yaml"

run_case () {
  local name=$1
  local rep=$2
  local chg=$3
  local out="${ROOT}/raw/smoke_${name}_2025-11-01_G2.csv"
  "${PYTHON_BIN}" scripts/run_param_sweep.py \
    --config "${CFG}" \
    --inventories 6 \
    --chargers 3 \
    --swap-capacities 4 \
    --reposition-solver "${rep}" \
    --charging-solver "${chg}" \
    --output-csv "${out}" \
    --no-resume \
    > "${ROOT}/logs/smoke_${name}.log" 2>&1
}

run_case algorithm_plus_gurobi gurobi gurobi
run_case heuristic_plus_fcfs heuristic fcfs
run_case ideal_plus_fcfs ideal fcfs
```

Smoke 通过条件：
- 3 个 smoke 文件均非空；
- 每个文件恰好 1 行结果；
- `algorithm_plus_gurobi` 结果含 `reposition_step_*` 与 `reposition_event_*` 字段；
- 关键 KPI 非空：`total_served`, `battery_swap_success_ratio`, `deadline_miss_ratio`, `charging_deadline_miss_ratio`。

---

## 6. 主实验（27 runs）

```bash
set -euo pipefail
PYTHON_BIN=${PYTHON_BIN:-/opt/anaconda3/envs/bsm/bin/python}

DATES=(2025-11-01 2025-11-05 2025-11-09)
GROUPS=(
  "G1 4 3 4"
  "G2 6 3 4"
  "G3 6 5 4"
)
COMBOS=(
  "algorithm_plus_gurobi gurobi gurobi"
  "heuristic_plus_fcfs heuristic fcfs"
  "ideal_plus_fcfs ideal fcfs"
)

for d in "${DATES[@]}"; do
  cfg="configs/generated_main_only/ev_eval_6_22_strict30_${d}.yaml"
  for g in "${GROUPS[@]}"; do
    read -r gid inv ch sw <<< "${g}"
    for c in "${COMBOS[@]}"; do
      read -r name rep chg <<< "${c}"
      out="${ROOT}/raw/${name}_${d}_${gid}.csv"
      "${PYTHON_BIN}" scripts/run_param_sweep.py \
        --config "${cfg}" \
        --inventories "${inv}" \
        --chargers "${ch}" \
        --swap-capacities "${sw}" \
        --reposition-solver "${rep}" \
        --charging-solver "${chg}" \
        --output-csv "${out}" \
        --no-resume \
        > "${ROOT}/logs/${name}_${d}_${gid}.log" 2>&1
    done
  done
done
```

---

## 7. 聚合与统计（用于 Primary Results）

### 7.1 合并 raw 结果

```bash
python - <<'PY'
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
    stem = name[:-4]  # remove .csv
    # format: combo_date_gid
    # combo may contain underscores, so split from right
    combo, date, gid = stem.rsplit("_", 2)
    with open(p, newline="", encoding="utf-8") as f:
        recs = list(csv.DictReader(f))
    if len(recs) != 1:
        raise RuntimeError(f"Expect exactly 1 row in {p}, got {len(recs)}")
    r = dict(recs[0])
    r["combo"] = combo
    r["sim_date"] = date
    r["group_id"] = gid
    rows.append(r)

out = f"{root}/agg/main_matrix_raw.csv"
if not rows:
    raise RuntimeError("No rows merged")
keys = sorted({k for r in rows for k in r})
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
print("wrote", out, "rows", len(rows))
PY
```

### 7.2 统计摘要与配对对比

```bash
python - <<'PY'
import os
import pandas as pd
import numpy as np

root = os.environ["ROOT"]
df = pd.read_csv(f"{root}/agg/main_matrix_raw.csv")

expected_rows = 27
if len(df) != expected_rows:
    raise RuntimeError(f"Row count mismatch: got {len(df)}, expected {expected_rows}")

required = [
    "combo", "sim_date", "group_id",
    "total_served",
    "battery_swap_success_ratio",
    "deadline_miss_ratio",
    "charging_deadline_miss_ratio",
]
missing = [c for c in required if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing required columns: {missing}")

for c in ["total_served", "battery_swap_success_ratio", "deadline_miss_ratio", "charging_deadline_miss_ratio"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

summary = (
    df.groupby("combo", as_index=False)[
        ["total_served", "battery_swap_success_ratio", "deadline_miss_ratio", "charging_deadline_miss_ratio"]
    ]
    .agg(["mean", "std"])
)
summary.columns = ["_".join([x for x in c if x]).strip("_") for c in summary.columns.to_flat_index()]
summary.to_csv(f"{root}/agg/main_summary.csv", index=False)

algo = df[df["combo"] == "algorithm_plus_gurobi"].copy()
heu = df[df["combo"] == "heuristic_plus_fcfs"].copy()
ide = df[df["combo"] == "ideal_plus_fcfs"].copy()
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

print("wrote", f"{root}/agg/main_summary.csv")
print("wrote", f"{root}/agg/paired_deltas.csv")
print("wrote", f"{root}/agg/win_tie_loss.csv")
PY
```

### 7.3 主指标显著性（配对）

```bash
python - <<'PY'
import os
import pandas as pd
from math import comb

root = os.environ["ROOT"]
df = pd.read_csv(f"{root}/agg/main_matrix_raw.csv")
key = ["sim_date", "group_id"]

algo = df[df["combo"] == "algorithm_plus_gurobi"].copy()
baselines = {
    "heuristic_plus_fcfs": df[df["combo"] == "heuristic_plus_fcfs"].copy(),
    "ideal_plus_fcfs": df[df["combo"] == "ideal_plus_fcfs"].copy(),
}

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
for bname, bdf in baselines.items():
    m = algo.merge(bdf, on=key, suffixes=("_a", "_b"), how="inner")
    d = (m["total_served_a"] - m["total_served_b"]).tolist()
    p_sign, n_eff = sign_test_two_sided(d)
    rows.append({
        "baseline": bname,
        "n_pairs": int(len(d)),
        "n_nonzero_pairs": int(n_eff),
        "mean_delta_total_served": float(pd.Series(d).mean()),
        "median_delta_total_served": float(pd.Series(d).median()),
        "pvalue_sign_test_two_sided": float(p_sign),
    })

out = f"{root}/agg/paired_stats.csv"
pd.DataFrame(rows).to_csv(out, index=False)
print("wrote", out)
PY
```

---

## 8. 质量门槛

必须同时满足：
- 完成率：27/27；
- 每个 `(sim_date, group_id, combo)` 唯一且仅 1 条；
- 时间口径一致：`sim_start_hour=6`, `sim_end_hour=22`, `effective_horizon=64`；
- 主方法审计字段存在（`reposition_step_*`, `reposition_event_*`）；
- 无静默失败（有失败必须在日志与汇总中显式标记）。

---

## 9. 交付文件（给论文写作）

在 `${ROOT}/agg/` 产出：
- `main_matrix_raw.csv`
- `main_summary.csv`
- `paired_deltas.csv`
- `paired_stats.csv`
- `win_tie_loss.csv`

在 `${ROOT}/` 产出：
- `logs/`（完整运行日志）
- `README_main_findings.md`（5-8 条结果要点，主语义仅围绕 Primary Results）

---

## 10. 给试验 Agent 的启动提示词（Main-Only）

你是 BSM 仓库的实验执行 agent。请严格执行 `docs/Evaluation_实验执行Runbook_MainOnly_2026-03-20.md`，只做正文主实验，不跑附录扩展目标。

要求：
1. 先做 Git/环境检查，再跑 Smoke Gate；
2. Smoke 通过后跑 27 个主实验 run（3 dates x 3 groups x 3 methods）；
3. 只比较三方法：`algorithm_plus_gurobi`, `heuristic_plus_fcfs`, `ideal_plus_fcfs`；
4. 固定口径：`06:00-22:00`, `15-min`, `horizon=64`；
5. 输出 `main_matrix_raw.csv`, `main_summary.csv`, `paired_deltas.csv`, `win_tie_loss.csv`；
6. 输出 `paired_stats.csv`（主指标配对显著性）；
7. 最后给出简洁实验报告：完成率、失败数、关键数值结论、风险点。
