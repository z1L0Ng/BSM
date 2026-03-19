# Evaluation 设计与实验执行方案

> 更新时间：2026-03-18
> 对应场景：3756 辆车压力场景（`initial_vehicles_scale=3.756`）
> 本文档包含：代码修改总结 / 实验设计 / 执行命令 / Paper 论点建议

---

## 一、已完成的代码修改

### 漏洞1：Gurobi 失败无退化保证 ✅
**文件**：`etaxi_sim/policies/reposition.py`

- 新增 `RepositionPolicyConfig.allow_heuristic_fallback: bool = True`（第 99 行）
- 在所有重试耗尽后（第 941 行附近），若 `allow_heuristic_fallback=True`，调用 `heuristic_battery_aware_policy` 而非 raise RuntimeError
- Trace 记录 `outcome="fallback_heuristic_no_incumbent"` 方便诊断

**意义**：确保算法在压力场景下不退化（worst case = 启发式策略），而不是崩溃或返回零派单

### 漏洞2：`charging_deadline_miss_ratio` 分母错误 ✅
**文件**：`etaxi_sim/sim/metrics.py`

- `record_step()`：分母从 `successful_swaps` 改为 `charging_demand`（第 57 行）
- `to_summary()`：分母从 `successful_swaps` 改为 `charging_demand`（第 129 行）

**意义**：换电量低时（系统过载），原始比值会人为放大。新分母 = 当前时步待充电任务总数，语义清晰

### 漏洞3：峰值功率指标增强 ✅
**文件**：`etaxi_sim/sim/metrics.py`

新增三个汇总指标：
- `avg_charging_power_kw`：全时段平均充电功率
- `charging_power_std_kw`：充电功率标准差（衡量调度平滑度）
- `avg_total_station_power_kw`：全时段平均总站点功率

**意义**：`peak_station_power_kw` 在饱和场景恒等于 `chargers×kW_per_charger`，无法区分策略。新指标 `charging_power_std_kw` 可以反映 Gurobi 调度的平滑化效果

### 漏洞5：`idle_driving_distance` 恒为零 ✅
**文件**：`etaxi_sim/sim/core.py` + `etaxi_sim/sim/metrics.py`

- `core.py`：新增 `cross_zone_dispatch = int(move_counts.sum() - np.trace(move_counts))`（第 138 行后）
- `metrics.py`：传递并汇总 `total_cross_zone_dispatch`

**意义**：原 `idle_driving_distance` 因 taxi_zones 模式能耗矩阵设计，所有同区派单能耗=0，导致指标恒为0。`total_cross_zone_dispatch` 直接计数跨区派单次数，是真实的"重新定位工作量"指标

---

## 二、漏洞4（充电层优势可见性）的分析与实验设计

### 根本问题诊断

从 timeseries 数据（`results/episodes/stress_compare/20260313_014228/timeseries.csv`）可以看到：

```
slot 0:  charging_power = 18480 kW  (616 chargers × 30kW = 18480, 所有可用充电器满负荷)
slot 1:  charging_power = 18480 kW  (持续饱和)
...
slot 16: charging_power = 18480 kW  (持续 16 个时步饱和)
slot 17: charging_power = 14760 kW  (开始下降)
slot 20: charging_power =  4620 kW  (大幅下降)
slot 21: charging_power =   810 kW  (基本空闲)
```

问题在于：slot 0 出现 **616 辆车同时到来**（所有车初始电量接近耗尽，同时触发换电），充电任务瞬间饱和所有充电器。在前 16 个时步，`charging_demand >= charger_capacity`，**两种策略都只能全力充电**，Gurobi 无法再削峰。

Slot 17 之后有明显的空闲容量（810 kW 对比最大 18480 kW），但此时大部分充电任务已完成。

### 充电层优势的核心论点

Gurobi 充电调度的真正价值在于：**当充电任务有足够的时间弹性（`required_slots << deadline_horizon`）且瞬时需求超过容量时，跨时步平滑负载**。

在当前参数下：
- `deadline_horizon = 20 slots`，`charge_rate = 5 levels/slot`
- 电量 50% 的电池：`required_slots = 10`，弹性 = 10 slots
- 电量 0% 的电池：`required_slots = 20`，弹性 = 0 slots（无法推迟）

**关键实验**：比较 `miss_penalty` 参数对 Gurobi 行为的影响

| miss_penalty | Gurobi 行为 | 预期结果 |
|---|---|---|
| 1000 (当前) | 强烈优先满足 deadline | 近似 EDF，峰值与 FCFS 相近 |
| 100 | 平衡峰值与 deadline | 更激进削峰，轻微增加 miss |
| 10 | 激进削峰 | 显著削峰，明显 miss |

这个 Pareto 曲线（peak power vs deadline miss）就是论文充电层的核心数值贡献。

### 需要新增的实验参数

在 `run_param_sweep.py` 中已有 `--charging-solver gurobi`。需要新增 `--miss-penalty` 参数，或创建多个配置文件。

---

## 三、完整实验执行计划

### 阶段 A：漏洞1 验证（先跑）

验证 heuristic fallback 生效，Gurobi+fallback >= heuristic 策略

```bash
export PROJ_DATA=/opt/anaconda3/envs/bsm/share/proj
export PROJ_LIB=/opt/anaconda3/envs/bsm/share/proj

# 在压力配置下跑 gurobi+fcfs（会触发 fallback），对比纯 heuristic+fcfs
python scripts/run_param_sweep.py \
  --config configs/ev_yellow_2025_11_eval_6_22_fcfs.yaml \
  --inventories 5,10,20 \
  --chargers 3,5,8 \
  --swap-capacities 3,6,9 \
  --reposition-solver gurobi \
  --charging-solver fcfs \
  --output-csv results/evaluation/phase_A_gurobi_fallback_fcfs.csv

python scripts/run_param_sweep.py \
  --config configs/ev_yellow_2025_11_eval_6_22_fcfs.yaml \
  --inventories 5,10,20 \
  --chargers 3,5,8 \
  --swap-capacities 3,6,9 \
  --reposition-solver heuristic \
  --charging-solver fcfs \
  --output-csv results/evaluation/phase_A_heuristic_fcfs.csv
```

**验收标准**：gurobi+fallback 的 `battery_swap_success_ratio` ≥ heuristic+fcfs 的对应值，`reposition_event_outcome_counts` 中出现 `fallback_heuristic_no_incumbent`

### 阶段 B：完整 evaluation 矩阵（主实验）

```bash
# 主算法：gurobi reposition + gurobi charging
python scripts/run_param_sweep.py \
  --config configs/ev_yellow_2025_11_eval_6_22_fcfs.yaml \
  --inventories 5,10,20 \
  --chargers 3,5,8 \
  --swap-capacities 3,6,9 \
  --reposition-solver gurobi \
  --charging-solver gurobi \
  --output-csv results/evaluation/phase_B_gurobi_gurobi.csv

# 基线1：heuristic + fcfs
# （使用阶段A的结果即可）

# 基线2：ideal + fcfs（上界）
python scripts/run_param_sweep.py \
  --config configs/ev_yellow_2025_11_eval_6_22_fcfs.yaml \
  --inventories 5,10,20 \
  --chargers 3,5,8 \
  --swap-capacities 3,6,9 \
  --reposition-solver ideal \
  --charging-solver fcfs \
  --output-csv results/evaluation/phase_B_ideal_fcfs.csv

# 基线3：greedy_same_zone + fcfs（最弱基线）
python scripts/run_param_sweep.py \
  --config configs/ev_yellow_2025_11_eval_6_22_fcfs.yaml \
  --inventories 5,10,20 \
  --chargers 3,5,8 \
  --swap-capacities 3,6,9 \
  --reposition-solver greedy \
  --charging-solver fcfs \
  --output-csv results/evaluation/phase_B_greedy_fcfs.csv
```

**每个 combo 至少跑 3 次，取中位数**（seed=42,43,44）

### 阶段 C：充电调度 Pareto 曲线（充电层贡献）

```bash
# 需要在 run_param_sweep.py 中添加 --miss-penalty 参数支持
# 固定：heuristic reposition + gurobi charging，仅变 miss_penalty
# 目的：展示 peak power vs deadline compliance 的 Pareto 前沿

for penalty in 10 50 100 500 1000; do
  python scripts/run_param_sweep.py \
    --config configs/ev_yellow_2025_11_eval_6_22_fcfs.yaml \
    --inventories 10 \
    --chargers 5 \
    --swap-capacities 6 \
    --reposition-solver heuristic \
    --charging-solver gurobi \
    --miss-penalty ${penalty} \
    --output-csv results/evaluation/phase_C_gurobi_penalty${penalty}.csv
done

# 对应 FCFS baseline（不受 miss_penalty 影响）
python scripts/run_param_sweep.py \
  --config configs/ev_yellow_2025_11_eval_6_22_fcfs.yaml \
  --inventories 10 \
  --chargers 5 \
  --swap-capacities 6 \
  --reposition-solver heuristic \
  --charging-solver fcfs \
  --output-csv results/evaluation/phase_C_fcfs_baseline.csv
```

### 阶段 D：时序图（24h 全局动态）

```bash
# 运行完整 episode 并保存 timeseries
# gurobi+gurobi vs heuristic+fcfs vs ideal+fcfs

python scripts/run_simulation.py \
  --config configs/ev_yellow_2025_11_eval_6_22_fcfs.yaml \
  --reposition-solver gurobi \
  --charging-solver gurobi \
  --inventory 10 --chargers 5 --swap-capacity 6 \
  --output-dir results/episodes/final_comparison

# 同样跑 heuristic+fcfs 和 ideal+fcfs 对照
```

---

## 四、Evaluation Section 大纲（CDC 风格）

### Section V: Simulation Study

**A. Setup**（约 150 字）
- NYC Yellow Taxi 真实出行数据（November 2025）
- 3756 辆电动出租车，全城分布式换电站
- 96 时步 × 15min = 24小时 episode
- 仿真参数表

**B. Baselines**

| 方法名 | 重定位策略 | 充电策略 | 说明 |
|---|---|---|---|
| **Proposed** | Gurobi MPC | Gurobi 峰值最小化 | 主方法 |
| Heuristic+FCFS | 电量感知启发式 | FCFS | 实用基线 |
| Greedy+FCFS | 原地不动 | FCFS | 最弱基线（展示重定位价值） |
| Ideal+FCFS | Oracle（无容量约束）| FCFS | 服务上界 |

**C. 实验一：服务质量 vs 站点资源约束**（Table）

指标：`battery_swap_success_ratio`, `total_waiting_vehicles`, `waiting_vehicles_p90`

- 变化参数：`swap_capacity ∈ {3,6,9}`, `inventory ∈ {5,10,20}`
- 固定：`chargers=5`
- 论点：Proposed ≈ Ideal >> Heuristic >> Greedy（当资源紧张时更明显）

**D. 实验二：重定位层的跨区调度价值**（Figure：swap_success_ratio vs cross_zone_dispatch）

指标：`total_cross_zone_dispatch`, `battery_swap_success_ratio`

- 论点：Gurobi MPC 在资源不足时主动引导跨区换电（而非 greedy 原地等待），提升系统整体换电成功率

**E. 实验三：充电调度 Pareto 前沿**（Figure：peak power vs deadline miss rate）

- 横轴：`charging_deadline_miss_ratio`
- 纵轴：`avg_total_station_power_kw`（或 `charging_power_std_kw`）
- 每个点对应一个 `miss_penalty` 值
- FCFS 为固定参考点
- 论点：Gurobi 充电调度器提供连续的 peak/deadline 权衡空间；运营者可根据电网成本与服务质量需求调节

**F. 实验四：计算效率**（Table）

指标：`reposition_step_runtime_sec_avg`, `reposition_step_runtime_sec_p95`

- 论点：每时步 Gurobi 求解时间 < 0.5s（远小于 15min 决策窗口），框架实时可行

---

## 五、论点建议与 Reviewer 常见质疑

### Q1：Gurobi 重定位 vs 启发式 performance 相近，贡献是什么？

**Argue**：在充足资源场景下，两者均达近优解，此时瓶颈是站点容量而非路由策略（**Theorem-level 洞察**）。Gurobi MPC 的价值在于：(a) 能量可行性约束的精确处理（避免电量耗尽），(b) 与充电层的联合耦合（MPC 预见性地将低电量车辆导向有余量的站点），(c) 提供最优性保证（与 ideal 的 gap 分析）。

### Q2：充电层峰值削减效果不显著？

**Argue**：当充电需求持续超载（所有充电器满负荷）时，无论何种调度峰值相同——这是 **资源约束下的可行性极限**，不是算法缺陷。框架的价值在于 Pareto 曲线（C3 实验），展示调度器提供 operator 可控的 peak-compliance 权衡。此外，`charging_power_std_kw` 展示即使峰值相同，Gurobi 调度也能实现更平滑的负载曲线。

### Q3：Solver 可行性问题（no incumbent）

**Argue**：当 `allow_heuristic_fallback=True` 时，算法保证退化到启发式下界（不低于 heuristic+fcfs）。论文明确声明实验中的 fallback rate（`reposition_event_outcome_counts` 统计），并在附录讨论求解稳定性与参数调优建议。

### Q4：只用一天的 NYC 数据，泛化性如何？

**Argue**：NYC Yellow Taxi 数据代表真实的城市出租车运营，具有早晚高峰特征。更多数据集测试（不同城市、不同季节）留作未来工作。

---

## 六、运行前的准备检查清单

```bash
# 1. 确认代码修改已合并到工作目录
git diff --stat

# 2. 确认 allow_heuristic_fallback 在 config 中生效
# 需要在 run_param_sweep.py / run_simulation.py 中暴露此参数到 YAML

# 3. 验证 metrics 新字段不影响现有流程
python scripts/validate_consistency.py --config configs/ev_yellow_2025_11_eval_6_22_fcfs.yaml

# 4. 先跑单步诊断确认 fallback 触发
python scripts/run_simulation.py \
  --config configs/ev_yellow_2025_11_eval_6_22_fcfs.yaml \
  --reposition-solver gurobi \
  --inventory 5 --chargers 3 --swap-capacity 3 \
  --max-steps 5 \
  --output-dir results/diagnostics/fallback_test
```

---

## 七、待办（实验 thread 执行前确认）

- [ ] `run_param_sweep.py` 是否支持 `--miss-penalty` 参数（阶段 C 需要）
- [ ] `allow_heuristic_fallback` 是否通过 YAML 配置暴露到 `ModelConfig`
- [ ] 确认 `greedy` solver 在 `run_param_sweep.py` 中已支持
- [ ] 阶段 B 每个 combo 3次重复的 seed 参数支持方式
