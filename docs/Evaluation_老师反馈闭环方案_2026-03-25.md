# Evaluation 老师反馈闭环方案（2026-03-25）

本文档按执行顺序组织：  
1) 先对比已完成/待补齐；  
2) 再更新 paper 文案；  
3) 再更新试验设计；  
4) 再下发试验 agent；  
5) 最后回填数据进入 Evaluation。

---

## 1) 当前完成度对照（针对老师反馈）

## 1.1 已完成

- 主实验口径已固定并可复现：06:00--22:00、15-min、horizon=64、3 dates。
- 主结果数据齐全：COP vs HDB vs ODB（27 runs）。
- 消融数据齐全：6 组合 x 3 dates x 3 settings（54 runs）。
- 时序校验已有 protocol-aligned 结果（64 slots）。
- 主要统计结论可复用：COP 在主任务 total served 上有显著优势（p=0.0039）。

## 1.2 还需补齐（老师新增）

- Setup 首段需明确：使用什么 data，以及“暂无真实 swapping station 日志”时如何构造站点仿真。
- Baseline 说明需从“实现名”改为“功能定义”，并把老师提到的三类 baseline 写清：
  - 车队级 dispatch + FCFS charging（FDB）
  - 车辆独立 heuristic + FCFS charging（HDB）
  - 充电并发受限（CCB, 例如 90% active chargers）
- CCB 需要补跑最小实验（建议 9 runs），并单独成节，不和主显著性混写。

结论：当前不是“推倒重来”，而是“主结果保留 + setup 叙述补齐 + CCB 增量补跑”。

---

## 2) Paper 文档先改什么（可直接贴 Overleaf）

下面是建议替换的 `Baseline Evaluation Setup` 前两段（英文）：

```latex
\subsection{Baseline Evaluation Setup}
We use NYC yellow taxi trip records (November 2025) to construct time-binned passenger demand.
The simulation horizon is fixed to 06:00--22:00 with 15-minute slots and receding horizon $H=64$.
Because real battery-swapping-station operation logs are not available in this study, station-side operations are simulated using parameterized station resources.
For each station, battery inventory, charging slots, and swap service capacity are configured by predefined resource settings, and all compared methods share exactly the same demand input, dates, and random-seed policy.

We evaluate the proposed method COP (Coordinated Optimization Policy) against three operational baselines.
FDB (Fleet Dispatch Baseline) performs fleet-level dispatch for passenger search and sends low-energy taxis to swapping stations; station charging follows FCFS.
HDB (Heuristic Dispatch Baseline) lets each taxi independently choose where to search and when to swap under low energy; station charging also follows FCFS.
ODB (Oracle Dispatch Baseline) uses future-demand-informed dispatch as a reference baseline while keeping the same FCFS charging rule.
In addition, we run a capacity-constrained charging baseline (CCB): dispatch follows the fleet-level policy, but only a fixed fraction of chargers can be active simultaneously (e.g., 90\%).
```

命名约束（正文统一）：
- `COP` / `FDB` / `HDB` / `ODB` / `CCB`
- 不出现 `gurobi+fcfs` 这类实现名（实现名只留在脚本映射和附录复现说明）。

---

## 3) 试验设计更新（最小增量）

## 3.1 主实验（不重跑）

- 继续使用已有 27 runs（COP/HDB/ODB）作为主文显著性结论来源。
- paired significance 仍基于 9 个 matched units（3 dates x 3 resource settings）。

## 3.2 新增 FDB（不重跑，从现有 54-run 中提取）

- 从 `ablation_6combo_3day_all.csv` 提取 `algorithm_plus_fcfs` 作为 FDB。
- 作用：补齐“老师定义的 baseline 1”。

## 3.3 新增 CCB（需要补跑，最小 9 runs）

目标：
- 验证“充电并发受限”对系统表现的影响。

当前实现约束：
- 代码暂无“active charger ratio”直接参数。
- 采用整数化 90% 近似（按每站 chargers 调整）：
  - 3 -> 2
  - 5 -> 4

矩阵：
- 3 dates x 3 resource settings x 1 method（CCB）= 9 runs。

定位：
- CCB 结果作为“capacity-constrained scenario”单独报告；
- 不并入主文 COP vs HDB/ODB 的 paired significance 主结论。

---

## 4) 给试验 agent 的执行文档与提示词

执行文档：
- [Evaluation_老师反馈对齐执行计划_2026-03-25.md](/Users/zilongzeng/.codex/worktrees/6f48/BSM/docs/Evaluation_老师反馈对齐执行计划_2026-03-25.md)

可直接发送给试验 agent 的提示词：

```text
请执行 docs/Evaluation_老师反馈对齐执行计划_2026-03-25.md 的任务 A/B/C，仅做 Evaluation 增量，不改算法实现。

目标：
1) 复用已有主实验（27 runs）与消融（54 runs）结果，不重跑它们。
2) 从消融结果提取 FDB（algorithm_plus_fcfs）并生成主文可用汇总表。
3) 新增 CCB 的 9 runs（3 dates x 3 settings）：dispatch 保持 fleet-level，charging 使用 FCFS，充电桩按 cap90 整数化（3->2, 5->4）。
4) 输出可直接写论文的聚合 CSV/MD，并给出最终目录。

硬约束：
- 不修改算法实现代码；
- 不改变主实验口径（06:00--22:00, 15-min, H=64, dates=2025-11-01/05/09）；
- CCB 作为单独补充场景，不写入主显著性结论。

必须交付：
- method_summary_primary_plus_fdb.csv
- date_summary_primary_plus_fdb.csv
- cap90_ccb_raw.csv
- cap90_ccb_summary.csv
- evaluation_tables_for_paper.csv
- evaluation_notes_for_paper.md

验收：
- CCB 行数=9；
- primary=27、ablation=54（仅核对，不重跑）；
- 关键列非空：total_served, charging_deadline_miss_ratio, total_waiting_vehicles, peak_station_power_kw。
```

---

## 5) 数据回填模板（试验完成后）

把下面条目回填到 Evaluation：

- 主实验结论（27 runs）：
  - COP vs HDB: total served 提升 `___%`，p=`___`
  - COP vs ODB: total served 提升 `___%`，p=`___`
- FDB 相对 COP/HDB 的位置：
  - total served：`___`
  - charging miss：`___`
- CCB（9 runs）相对 FDB：
  - total served 变化：`___`
  - charging miss 变化：`___`
  - waiting 变化：`___`
  - peak power 变化：`___`

写作约束：
- 主结论只用主实验显著性；
- CCB 只作容量约束下鲁棒性/代价分析；
- 明确 trade-off，不写“全面优于所有指标”。

