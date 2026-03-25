# Evaluation 老师反馈对齐执行计划（2026-03-25）

目标：把老师新增反馈落地为可执行实验任务，并保持当前 CDC 主叙事稳定。  
范围：仅更新 Evaluation 相关实验与写作输入；不改算法实现逻辑。

---

## 0. 冻结约束（先锁定）

- 主实验口径保持不变：`06:00-22:00`、`15-min`、`horizon=64`、日期 `2025-11-01/05/09`。
- 现有主结果（27 runs）继续作为主文核心证据，不推翻已有结论。
- 新增“容量约束”基线作为**单独子节**（stress-style），不与主表混写。
- 论文命名统一使用功能名，内部实现名仅用于脚本和日志。

---

## 1. 论文中需要补充的 Setup 口径

### 1.1 数据与仿真构造（放在 Baseline Evaluation Setup 首段）

- 需求数据来自 NYC yellow taxi trip records（2025-11）。
- 当前无真实换电站运营日志，因此换电站网络在仿真中按统一参数构造（同构站点）。
- 每个站点的库存、电池充电桩数量、换电服务能力由资源配置给定。
- 资源设置采用三组预定义配置（Critical3），并在三天上重复评估。

### 1.2 方法命名（论文用）

- `COP`: Coordinated Optimization Policy（本文方法）。
- `FDB`: Fleet Dispatch Baseline（车队级调度 + FCFS 充电，对应 `algorithm_plus_fcfs`）。
- `HDB`: Heuristic Dispatch Baseline（车辆独立启发式调度 + FCFS 充电，对应 `heuristic_plus_fcfs`）。
- `ODB`: Oracle Dispatch Baseline（带未来需求信息的 dispatch + FCFS 充电，对应 `ideal_plus_fcfs`，作为参考基线）。
- `CCB`: Capacity-Constrained Charging Baseline（与 FDB 相同 dispatch/charging 规则，但施加充电并发上限）。

说明：`ODB` 继续保留为参考基线；`CCB`单列为容量受限场景，不并入主结果显著性表。

---

## 2. 试验任务分解（给试验 agent）

## 2.1 任务 A：复用已有结果生成论文主表输入（不重跑）

输入：
- `results/evaluation_runs/20260320_214439_main_only_3day/agg/main_matrix_raw.csv`（COP/HDB/ODB）
- `results/evaluation_runs/20260321_223203_incremental_fix_maintext/agg/ablation_6combo_3day_all.csv`（含 `algorithm_plus_fcfs`）

目标：
- 产出统一方法对比表输入（COP/FDB/HDB/ODB）。
- 保持主文“27-run paired comparison”用于 COP vs HDB/ODB，不改变统计口径。

交付：
- `results/evaluation_runs/20260325_eval_refresh/agg/method_summary_primary_plus_fdb.csv`
- `results/evaluation_runs/20260325_eval_refresh/agg/date_summary_primary_plus_fdb.csv`

---

## 2.2 任务 B：新增容量约束基线 CCB（最小补跑）

定义：
- 基于 FDB 逻辑（`reposition=algorithm`, `charging=fcfs`）。
- 对每站可同时工作的充电桩施加 90% 上限。

实现口径（当前代码无“比例阈值”开关，采用整数化近似）：
- G1: chargers `3 -> 2`
- G2: chargers `3 -> 2`
- G3: chargers `5 -> 4`

规模：
- 仅跑 CCB：`3 dates x 3 resource settings x 1 method = 9 runs`

建议输出目录：
- `results/evaluation_runs/20260325_capacity_cap90_3day/`

通过条件：
- 行数=9，键唯一（`sim_date,resource_id,method`）。
- 关键列非空：`total_served`、`charging_deadline_miss_ratio`、`total_waiting_vehicles`、`peak_station_power_kw`。
- 同步生成聚合表：
  - `agg/cap90_ccb_raw.csv`
  - `agg/cap90_ccb_summary.csv`

---

## 2.3 任务 C：汇总写作输入（主实验 + 容量约束）

将 A/B 合并成两层结果：
- `Layer-1 Primary`（COP/FDB/HDB/ODB，标准口径）。
- `Layer-2 Capacity-Constrained Check`（CCB vs FDB 或 CCB vs COP 的补充对比）。

交付文件：
- `results/evaluation_runs/20260325_eval_refresh/agg/evaluation_tables_for_paper.csv`
- `results/evaluation_runs/20260325_eval_refresh/agg/evaluation_notes_for_paper.md`

`evaluation_notes_for_paper.md` 至少包含：
- 主实验结论（不变部分）。
- CCB 相对 FDB/COP 的变化方向与幅度。
- 明确声明：CCB 为容量受限情景，不参与主文 paired significance 结论。

---

## 3. 推荐执行命令（试验 agent）

## 3.1 环境与目录

```bash
git fetch --prune origin
git status --short --branch
PYTHON_BIN=${PYTHON_BIN:-/opt/anaconda3/envs/bsm/bin/python}
```

## 3.2 跑 CCB（9 runs，整数化 90% cap）

```bash
set -euo pipefail
TS=$(date +"%Y%m%d_%H%M%S")
ROOT="results/evaluation_runs/${TS}_capacity_cap90_3day"
mkdir -p "${ROOT}"/{raw,agg,logs}

DATES=(2025-11-01 2025-11-05 2025-11-09)
GROUPS=(
  "R1 4 2 4"  # from G1: chargers 3->2
  "R2 6 2 4"  # from G2: chargers 3->2
  "R3 6 4 4"  # from G3: chargers 5->4
)

for d in "${DATES[@]}"; do
  cfg="configs/generated_main_only/ev_eval_6_22_strict30_${d}.yaml"
  for g in "${GROUPS[@]}"; do
    read -r rid inv ch sw <<< "${g}"
    out="${ROOT}/raw/ccb_${d}_${rid}.csv"
    "${PYTHON_BIN}" scripts/run_param_sweep.py \
      --config "${cfg}" \
      --inventories "${inv}" \
      --chargers "${ch}" \
      --swap-capacities "${sw}" \
      --reposition-solver algorithm \
      --charging-solver fcfs \
      --output-csv "${out}" \
      --no-resume \
      > "${ROOT}/logs/ccb_${d}_${rid}.log" 2>&1
  done
done
```

## 3.3 合并 CCB 结果（示例）

```bash
python - <<'PY'
import csv, glob
from pathlib import Path

root = Path("results/evaluation_runs").glob("*_capacity_cap90_3day")
root = sorted(root)[-1]
rows = []
for p in sorted((root / "raw").glob("ccb_*.csv")):
    stem = p.stem  # ccb_2025-11-01_R1
    _, d, rid = stem.split("_", 2)
    recs = list(csv.DictReader(p.open()))
    if len(recs) != 1:
        raise RuntimeError(f"{p} row != 1")
    r = dict(recs[0])
    r["method"] = "CCB"
    r["sim_date"] = d
    r["resource_id"] = rid
    rows.append(r)

out = root / "agg" / "cap90_ccb_raw.csv"
out.parent.mkdir(parents=True, exist_ok=True)
keys = sorted({k for r in rows for k in r})
with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
print("wrote", out, "rows", len(rows))
PY
```

---

## 4. 风险与解释口径（写作时必须声明）

- CCB 使用“整数化 90% cap”（2/2/4）是当前代码能力下的工程近似，不是精确小数并发约束。
- 因 CCB 改变了可用资源上限，结论应作为“容量受限鲁棒性检查”，不与主实验显著性结论混同。
- 主实验显著性仍以 9 个 matched units（COP vs HDB/ODB）报告。

---

## 5. 可直接发给试验 agent 的提示词

```text
请执行 docs/Evaluation_老师反馈对齐执行计划_2026-03-25.md 的任务 A/B/C，仅做 Evaluation 增量，不改算法实现。

硬约束：
1) 不重跑主实验 27-run（复用已有结果）。
2) 新增仅 CCB 9-run：3 dates x 3 resource settings，方法固定为 reposition=algorithm + charging=fcfs。
3) CCB 的 chargers 采用整数化 cap90：G1 3->2，G2 3->2，G3 5->4。
4) 输出 CSV/MD 到 results/evaluation_runs/<timestamp>_* 目录，并给出最终路径。

必须交付：
- method_summary_primary_plus_fdb.csv
- date_summary_primary_plus_fdb.csv
- cap90_ccb_raw.csv
- cap90_ccb_summary.csv
- evaluation_tables_for_paper.csv
- evaluation_notes_for_paper.md

验收检查：
- CCB 原始行数=9；
- 主结果 primary=27、ablation=54（只检查不重跑）；
- 关键列非空：total_served、charging_deadline_miss_ratio、total_waiting_vehicles、peak_station_power_kw；
- 汇总文档中明确写出：CCB 属于容量受限补充，不参与主显著性检验结论。
```

