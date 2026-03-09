# TODO：仿真有效性改进清单（讨论版）

更新时间：2026-03-09  
状态：阶段一至阶段三已落地，进入风险修复与结果重跑

## 第一阶段（最优先）：补充实验指标与日志

### 1. Simulation 指标

- [x] `served_passengers`（`total_served`）
- [x] `battery_swap_success_ratio`（`successful_swaps / swap_arrivals`）
- [x] `deadline_miss_ratio`（换电请求未满足比例，`(swap_requests - successful_swaps) / swap_requests`）
- [x] `charging_deadline_miss_ratio`（充电任务逾期比例，`total_charging_deadline_misses / total_number_of_swaps`）
- [x] `idle_driving_distance`（当前为能耗矩阵代理距离；接入真实路网后可替换为公里）
- [x] `waiting_time_for_battery`（已实现显式等待队列 `W` 与 vehicle-slot 累计）

### 2. Power System 指标

- [x] 每个 `time slot` 的 `charging_power_kw`
- [x] 所有站点 `total_charging_demand`（slot 级）
- [x] `maximum charging power demand`（`max_charging_power_demand_kw`）

### 3. Episode 完整日志

- [x] 每次 simulation 保存完整日志
- [x] 输出字段包含：
  - `time_slot`
  - `charging_demand`
  - `number_of_swaps`
  - `unmet_battery_demand`
- [x] 输出路径：
  - `results/episodes/<run_id>/timeseries.csv`
  - `results/episodes/<run_id>/summary.json`

---

## 第二阶段：参数设计（实验矩阵）

- [x] `battery_inventory_per_station` sweep：50 / 100 / 150 / 200（已在 `run_param_sweep.py`）
- [x] `charging_ports_per_station` sweep：5 / 10 / 15（已在 `run_param_sweep.py`）
- [x] `station_capacity` / `zone` 规模 sweep（当前先完成 `swap_capacity=6/12/18`）
- [x] 控制供给强度，保留可区分算法差异的瓶颈区间（`swap_capacity=6` 可稳定触发差异）

---

## 第三阶段：Baseline 实现与组合实验

- [x] Baseline 1（Station）：`FCFS`（已支持 `model.charging_solver=fcfs`）
- [x] Baseline 2（Vehicle）：`Ideal swap assumption`（`reposition_solver=ideal`）
- [x] Baseline 3（Vehicle）：`Heuristic dispatch`（`reposition_solver=heuristic`）
- [x] 组合实验：`车辆策略 × Station 策略`（已完成全矩阵，见 `results/baseline_combinations/full_matrix.csv`）
  - `Current Algorithm + FCFS`
  - `Heuristic + FCFS`
  - `Ideal schedule + FCFS`

---

## 下一阶段：风险修复（最高优先）

### A. 指标语义一致性（P0）

- [x] 修复 `deadline_miss_ratio`：改为“换电请求未满足比例”，不再复用充电任务逾期计数
- [x] 新增并单独输出 `charging_deadline_miss_ratio`（充电任务逾期比例）
- [x] 在 `timeseries.csv` 与 `summary.json` 同时输出两类 miss 指标，避免混淆
- [x] 更新文档中的指标定义，确保实现与 TODO 一致

### B. 参数区分度（P0）

- [x] 增加更强瓶颈 sweep（低库存/低换电能力/低充电口）
- [x] 验证 `algorithm` 与 `heuristic` 是否在新参数下拉开差异
- [x] 若仍无差异，继续调整 `initial_vehicles_scale`、`swap_low_energy_threshold`、`reposition` 权重（当前强瓶颈网格已出现差异，暂不继续下钻）
  - `results/baseline_combinations/stress_algorithm_plus_fcfs.csv`
  - `results/baseline_combinations/stress_heuristic_plus_fcfs.csv`

### C. 日志与结果稳定性（P1）

- [x] 固定 episode 输出 schema（字段稳定，不随版本漂移）
- [x] 给结果文件增加 `schema_version` 与关键配置快照字段
- [x] 统一后处理脚本读取逻辑（兼容旧结果，默认用新 schema）
  - `scripts/export_episode_summaries.py` -> `results/episodes/summary_index.csv`

### D. 结果交付（P1）

- [x] 基于修复后的结果重跑强瓶颈矩阵 `baseline_combinations/stress_full_matrix.csv`
- [x] 生成对比图（成功率、swap miss、charging miss、等待、峰值功率）
  - `results/baseline_combinations/figures/`
  - `scripts/plot_stress_results.py`
- [x] 形成主结论表（优先使用瓶颈参数子集）
  - `results/baseline_combinations/stress_summary.md`
