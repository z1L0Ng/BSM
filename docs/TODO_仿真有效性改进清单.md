# TODO：仿真有效性改进清单（讨论版）

更新时间：2026-03-08  
状态：阶段一至阶段三已落地，进入结果解释与策略增强

## 第一阶段（最优先）：补充实验指标与日志

### 1. Simulation 指标

- [x] `served_passengers`（`total_served`）
- [x] `battery_swap_success_ratio`（`successful_swaps / swap_arrivals`）
- [x] `deadline_miss_ratio`（`total_deadline_misses / total_swap_arrivals`）
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
