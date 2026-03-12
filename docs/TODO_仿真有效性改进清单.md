# TODO：仿真有效性改进清单（讨论版）

更新时间：2026-03-12
状态：阶段一至三及风险修复全部完成，进入实验完善与论文对齐阶段

---

## ✅ 已完成（归档）

<details>
<summary>阶段一至三 + 风险修复（点击展开）</summary>

**阶段一：指标与日志**
- [x] `served_passengers`, `battery_swap_success_ratio`, `deadline_miss_ratio`, `charging_deadline_miss_ratio`, `idle_driving_distance`, `waiting_time_for_battery`
- [x] `charging_power_kw`, `total_charging_demand`, `max_charging_power_demand_kw`
- [x] Episode 完整日志：`results/episodes/<run_id>/timeseries.csv` + `summary.json`

**阶段二：参数 sweep 设计**
- [x] `inventory` sweep：50/100/150/200，`chargers` sweep：5/10/15，`swap_capacity` sweep：6/12/18
- [x] Stress grid（低资源）：inventory=5/10/20，chargers=3/5/9，swap_capacity=3/6/9

**阶段三：Baseline 组合实验**
- [x] FCFS 充电 baseline、Ideal 换电 baseline、Heuristic 调度 baseline
- [x] 完整矩阵：`full_matrix.csv`（108 runs）、`stress_full_matrix.csv`（81 runs）

**风险修复**
- [x] 两套 miss 指标语义分离（swap deadline miss vs charging deadline miss）
- [x] Stress grid 证明 algorithm vs heuristic 在低资源下有差异
- [x] Episode schema 固定，`export_episode_summaries.py` 统一后处理
- [x] 对比图和 `stress_summary.md` 生成完毕

</details>

---

## 第四阶段：实验完善与结论固化（当前优先）

### A. 关键发现诊断（P0）

当前 stress grid 出现一个重要异常需要先诊断清楚再推进实验：

**异常**：`algorithm_plus_fcfs` 的 `charging_deadline_miss_ratio ≈ 1.0`（几乎所有充电任务都逾期），而 `heuristic_plus_fcfs` 只有 0.07。这不应是调度策略差异导致的（两者都用 FCFS 充电），需要排查根因。

- [ ] **诊断 algorithm+FCFS 充电逾期率异常**
  - 检查 `gurobi_reposition_policy` 在低库存（inventory=5）下是否触发 exception fallback 到 `greedy_same_zone_policy`
  - 检查 `swap_arrivals` 分布：algorithm 是否将大量低电量车发到换电站，但 full_batteries=5 不够服务，导致大量换电任务无效生成却无法完成
  - 对比 algorithm 和 heuristic 在同一 run 下的 `total_number_of_swaps`（algorithm=768 vs 理论值）
  - 输出：写一段诊断说明到 `docs/`

### B. 补充 Gurobi vs EDF 充电策略对比（P0）

目前所有组合都使用 FCFS 充电，**缺少 Gurobi 峰值最优充电作为充电侧 baseline 上界**：

- [ ] 在 stress grid 上补跑 `algorithm + Gurobi charging` 组合
  - 预期：charging_deadline_miss_ratio 大幅下降，peak_station_power 也下降
  - 验证充电优化器的有效性（与 FCFS 对比）
- [ ] 补跑 `heuristic + Gurobi charging` 对比组
- [ ] 更新 `stress_full_matrix.csv` 或单独保存到 `stress_gurobi_charging_matrix.csv`
- [ ] 更新 `stress_summary.md` 补充充电策略维度的对比

### C. 完整实验矩阵设计（P1）

当前矩阵只覆盖"reposition × charging = 3×1（只有 FCFS）"，需要扩展为完整 3×2 矩阵：

- [ ] 定义完整 baseline 矩阵：
  ```
  Reposition:  [algorithm(Gurobi MPC), heuristic, ideal]
  Charging:    [Gurobi peak-min, FCFS]
  ```
- [ ] 在 stress grid 下跑全 6 组合（各 27 参数配置 × 6 = 162 runs）
- [ ] 更新 `run_baseline_combinations.py` 支持 charging_solver 维度
- [ ] 汇总为 `stress_full_matrix_v2.csv`

### D. 单次完整 Episode 分析（P1）

目前只有 sweep 聚合结果，缺少时序细节分析：

- [ ] 选取"典型中等压力参数"（inventory=20, chargers=5, swap_capacity=6）跑一次完整 24h episode
  - reposition=gurobi + charging=gurobi（主算法）
  - reposition=heuristic + charging=fcfs（对比）
- [ ] 对比两者的时序曲线：
  - `served` 随时间变化（高峰期差异）
  - `full_batteries` 库存随时间演化（是否出现库存耗尽）
  - `charging_power_kw` 时序（Gurobi 是否实现削峰填谷）
  - `waiting_vehicles` 时序（排队积压的时间分布）
- [ ] 生成时序对比图，保存到 `results/episode_analysis/`

---

## 第五阶段：论文对齐与实现文档补全（P2）

### E. 已知近似的文档化

评测发现代码中存在几处未在文档中明确说明的近似：

- [ ] 在技术规格说明书 §7.2 补充说明：**当前实现为解耦近似**（reposition 与 charging 分步优化，而非联合求解）
- [ ] 在技术规格说明书 §3.3.2 补充：Gurobi 充电模型中充电完成约束 Eq.(12)/(14) 为**软约束**（带 slack + 1000 惩罚），资源充足时 slack=0 等价于硬约束
- [ ] 在 §4 补充：`P_tilde`/`Q_tilde` 由于缺少空驶显式标签，采用需求强度校准近似，并说明误差校验方式
- [ ] 验证 `miss_penalty=1000` 量级：运行一次典型 episode 输出 slack 值，确认资源充足时 slack=0

### F. miss_penalty 量级验证（P1）

- [ ] 抽取一个宽松参数（inventory=100, chargers=10）的 Gurobi 充电 run
- [ ] 记录每个 slot 中 `slack` 变量的取值分布
- [ ] 若任何 slot 出现 slack > 0，评估是否需要调整惩罚系数
- [ ] 结论写入 `docs/论文公式一致性核对表.md` 的"仍需注意"一节

### G. `full_stock` 动态的近似说明（P2）

- [ ] 在 `reposition.py` 的 Gurobi 模型中添加注释：`full_stock` 动态不包含充电完成后的电池回补，属于保守近似
- [ ] 在技术规格说明书对应章节注明此近似对规划窗口的影响（低估未来可用满电电池）

---

## 第六阶段：论文写作支撑材料（P3，时间允许时推进）

- [ ] 整理 stress grid 主结论表（LaTeX 格式），用于论文 Table
  - 核心指标：served, swap_success_ratio, deadline_miss_ratio, charging_miss_ratio, peak_power, waiting_p90
  - 参数子集：只选典型瓶颈行（inventory=5 or 10, chargers=3, swap_capacity=3 or 6）
- [ ] 生成论文质量图（300dpi, 双栏宽度）
  - Fig 1：充电策略对比（Gurobi vs FCFS）—— 峰值功率时序
  - Fig 2：调度策略对比 —— swap success ratio 随 inventory 变化曲线
  - Fig 3：完整 episode 时序对比（algorithm vs heuristic）
- [ ] 确认 `idle_driving_distance` 单位：目前是能耗等级代理值，是否需要换算为公里

---

## 已知技术债务（不影响当前实验，记录备查）

| 项目 | 描述 | 优先级 |
|------|------|--------|
| `partial_batteries` Eq.(6) 对应 | `apply_charging` 中变量 y_i^{t,l} 是隐式的，无显式变量审计 | P3 |
| `swap_success_ratio` 分母 | 排队积压时 ratio_den=swap_requests（含排队），miss 语义略有偏差 | P3 |
| `P_tilde/Q_tilde` 误差量化 | 空驶接客近似的误差范围未定量分析 | P3 |
| 解耦优化误差上界 | reposition 与 charging 解耦的次优程度未分析 | P3 |
