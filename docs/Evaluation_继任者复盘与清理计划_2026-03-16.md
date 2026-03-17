# Evaluation 继任者复盘与清理计划（2026-03-16）

## 1) 已核查文档

- `docs/继任者交接_出差期实验执行方案.md`
- `docs/TODO_仿真有效性改进清单.md`
- `docs/论文公式一致性核对表.md`
- `docs/代码审查报告_Paper对齐核查.md`

## 2) 论文实现对齐复盘（面向 evaluation）

### A. 论文核心公式实现状态

- Eq.(1)(2)(3)(4)(5)(6)(9)：代码侧已有实现，且已有守恒/约束核查脚本支撑。
- Eq.(12)(14)：当前是软约束实现（`slack + penalty`），属于工程近似，文档已标注。
- 工程扩展：`waiting_queue`、MPC 目标扩展项、`P_tilde/Q_tilde` 估计近似，均已在 docs 里披露。

### B. 与当前 evaluation 目标的关键缺口（复盘结论）

1. 交接文档要求 `06:00-22:00` 口径统一，但原实现仅支持整天 + `horizon` 截断，窗口口径未真正打通。  
2. `run_param_sweep.py` 原先“整组结束才写盘”，不满足出差场景抗中断要求。  
3. P0 gate 要求的求解过程可审计字段（`Status/SolCount/Runtime` + fallback 事件）未在 sweep 主流程聚合输出。  
4. 历史 `results/baseline_combinations/*.csv` 中存在旧口径结果，不应直接用于当前 evaluation 结论。

## 3) 本轮仓库清理动作（已落地）

### 3.1 时间窗口口径统一（06:00-22:00）

- 新增 `sim.sim_start_hour` / `sim.sim_end_hour` 配置（默认 `0-24`，兼容旧配置）。
- `load_yellow_trip_demand` / `make_transition_from_tripdata` / `estimate_peak_concurrent_trips` 支持同一窗口参数。
- `run_param_sweep.py` / `run_simulation.py` / `run_reposition_diagnostic.py` / `validate_consistency.py` 全部改为使用 `effective_horizon`（窗口与 `horizon` 取最小）。

### 3.2 Sweep 抗中断改造

- `run_param_sweep.py` 改为每个 case 完成后立即写回 CSV。
- 新增 `--resume/--no-resume`，默认启用断点续跑，自动跳过已完成 case。
- 输出新增窗口字段：`sim_start_hour`、`sim_end_hour`、`effective_horizon`。

### 3.3 求解可审计字段补齐

- `etaxi_sim/policies/reposition.py` trace 新增 `runtime_sec`。
- 新增 trace 统计接口：`reset_gurobi_reposition_trace_stats` / `get_gurobi_reposition_trace_stats`。
- `run_param_sweep.py` 输出新增：
  - `reposition_step_*`（按时段聚合）
  - `reposition_event_*`（包含 retry/fallback/no_incumbent 事件计数）
- `run_reposition_diagnostic.py` CSV 新增 `reposition_runtime_sec` 字段。

### 3.4 非 Gurobi 评估入口

- 新增 `configs/ev_yellow_2025_11_eval_6_22_fcfs.yaml`（默认 `06:00-22:00`，`horizon=64`）。
- 新增 `scripts/run_stress_gate_48_nogurobi.sh`（仅跑 `algorithm/heuristic/ideal + fcfs`）。

## 4) 当前待执行（evaluation 运营顺序）

1. 先跑 `run_stress_gate_48_nogurobi.sh`，确认 48-run 无中断并产出新口径矩阵。  
2. 基于新矩阵刷新 `stress_summary.md`，标注与历史口径差异（`24h` vs `06:00-22:00`）。  
3. 通过 gate 后，继续跑非 Gurobi 全量组合；Gurobi 组合待用户确认供电稳定后再启动。  
4. 产出 2 组 episode 时序证据（`ours` vs `heuristic+fcfs`）补齐 evaluation 图表。

## 5) 风险与约束

- 若机器休眠/断电，长任务仍会中断，但现在可通过 `--resume` 恢复。
- `algorithm_plus_fcfs` 依赖重定位 Gurobi；若当前策略要求“完全不跑 Gurobi”，需从候选集合中排除该组合。
