# TODO：仿真有效性改进清单（执行版）

更新时间：2026-03-16  
当前基线提交：`7f3f046`（已推送 `main`）  
目标：通过 P0 gate 后，产出可直接写入 paper evaluation 的完整证据链。

## 0) 已完成（本周关键）

- [x] 修复充电优化目标与评估口径不一致：
  - `gurobi charging` 改为分层目标：先最小化任务逾期，再最小化峰值，再最小化逾期 slot。
  - 文件：`etaxi_sim/policies/charging.py`
- [x] 新增诊断指标 `charging_deadline_missed_slots`：
  - 文件：`etaxi_sim/sim/core.py`, `etaxi_sim/sim/metrics.py`
- [x] 单点 stress 复核（`inventory=10, chargers=3, swap=3`）：
  - `gurobi+gurobi(after_fix)` 的 `charging_deadline_miss_ratio` 已与 `gurobi+fcfs` 对齐（约 `0.14`）。
- [x] 诊断脚本与结果归档已入库（见 `results/baseline_combinations/stress_single_*`）。

## 1) P0 Gate（必须先过）

### A. 3756 车辆场景求解稳定性

- [ ] 在重定位求解流程记录每时段 `Status/SolCount/Runtime`。
- [ ] 记录每次 fallback 的触发原因与次数（不得 silent fallback）。
- [ ] 验收：96 步完整跑通，日志可审计，且无“无解即中断”。

### B. `charging_deadline_miss_ratio` 异常闭环

- [ ] 用新指标拆解：`charging_deadline_misses` vs `charging_deadline_missed_slots` vs `successful_swaps`。
- [ ] 重新验证异常是否来自旧目标函数（历史 `stress_full_matrix.csv` 结果需谨慎使用）。
- [ ] 输出 root-cause 结论页（图+表+文字）。

### C. 小规模 gate 复跑（48-run）

- [ ] 矩阵：`6 combos × (inventory{5,10} × chargers{3,8} × swap{3,6}) = 48`
- [ ] 通过条件：
  - [ ] 无卡死/中断；
  - [ ] `algorithm+gurobi` 与 `algorithm+fcfs` 在关键指标上趋势合理；
  - [ ] 不再出现无法解释的系统性 `miss_ratio≈1` 伪异常。

## 2) Paper Evaluation 主实验（Gate 通过后执行）

### D. 全量 162-run 主矩阵

- [ ] 矩阵：`{gurobi, heuristic, ideal} × {gurobi, fcfs} × 3×3×3 = 162`
- [ ] 输出：`full_matrix_6combo.csv` + 按组合汇总表。
- [ ] 主表指标：
  - [ ] `total_served`
  - [ ] `battery_swap_success_ratio`
  - [ ] `deadline_miss_ratio`
  - [ ] `charging_deadline_miss_ratio`
  - [ ] `max_station_total_power_kw`
  - [ ] `total_waiting_vehicles`

### E. 24h 时序证据（削峰填谷）

- [ ] 至少 2 组 episode 对比：`ours(gurobi+gurobi)` vs `heuristic+fcfs`
- [ ] 图：`charging_power_kw`、`station_total_power_kw`、`waiting_vehicles`
- [ ] 在正文明确“服务质量 vs 峰值负荷”的 trade-off。

## 3) 文档一致性与写作收口

- [ ] 更新 `docs/论文公式一致性核对表.md`：补充 gate 结果与最终口径。
- [ ] 更新 `docs/代码审查报告_Paper对齐核查.md`：区分“历史异常”与“修复后结果”。
- [ ] 更新 `results/baseline_combinations/stress_summary.md`：替换旧结论，避免引用过时矩阵。

## 4) 当前执行顺序（从现在开始）

1. 执行 48-run gate（进行中）  
2. 输出 gate 汇总与是否放行结论  
3. 放行后启动 162-run 全量矩阵  
4. 完成时序图与 evaluation 文本草稿

## 5) 2026-03-16 继任者清理进展

- [x] 统一 `06:00-22:00` 口径能力已落地（配置/需求/转移/初始车辆估计同窗）
- [x] `run_param_sweep.py` 已支持逐 case 落盘与 `--resume` 断点续跑
- [x] 重定位 trace 已补充 `Runtime` 字段，并输出 fallback/retry/no-incumbent 事件聚合
- [ ] 仍需实跑 gate 确认：在 3756 车辆规模下是否满足“无卡死 + 可审计”
