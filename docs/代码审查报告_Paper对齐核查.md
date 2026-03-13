# 代码审查报告：Paper 与实现对齐核查

审查时间：2026-03-12
审查范围：`etaxi_sim/` 全部模块 vs `docs/E_taxi_Battery_Swapping_Stations.pdf`
审查方法：逐公式核对代码实现，检查逻辑漏洞和不一致

---

## 1. 审查总结

| 类别 | 数量 |
|------|------|
| 与论文一致 | 8 项 |
| 工程近似/扩展（需文档说明） | 5 项 |
| 代码缺陷（待在 main 验证） | 1 项 |

整体评价：代码对论文核心公式（Eq.1-6, 9, 12-14）的实现**基本正确**，验证脚本（`validate_consistency.py`）确认了车辆守恒、电池守恒、约束无违规。当前新增关注点是高压力场景下的求解稳定性（例如 3756 车辆场景的 `no incumbent`），该类风险不在本核查脚本覆盖范围内。

---

## 2. 逐公式核对

### 2.1 Eq.(1): 调度后车辆分布 — **一致**

**论文**: $B_{i,l}^t = \sum_{i'} Y_{i',i}^{t,l}$, $S_{i,l}^t = \sum_{i'} X_{i',i}^{t,l}$

**代码** (`etaxi_sim/sim/core.py:129-130`):
```python
B = Y.sum(axis=0)  # shape (m, L+1), axis=0 对出发地求和
S = X.sum(axis=0)
```

`Y` 的 shape 为 `(m_from, m_to, L+1)`，`sum(axis=0)` 对出发地维度 $i'$ 求和，得到目的地 $i$ 维度的分布。**正确**。

---

### 2.2 Eq.(2)(3): 状态转移 — **一致（含合理工程处理）**

**论文 Eq.(2)**: $O_{i,l}^{t+1} = \sum_{i'} \tilde{P}_{i',i}^t \cdot S_{i',l+E_{i',i}^t}^t + \sum_{i'} P_{i',i}^t \cdot U_{i',l+E_{i',i}^t}^t$

**论文 Eq.(3)**: $V_{i,l}^{t+1} = \sum_{i'} \tilde{Q}_{i',i}^t \cdot S_{i',l+E_{i',i}^t}^t + \sum_{i'} Q_{i',i}^t \cdot U_{i',l+E_{i',i}^t}^t + H_{i,l}^t$

**代码** (`etaxi_sim/sim/core.py:270-378`, `_state_transition` + `_assign_transitions`):

实现要点：
1. 遍历每个出发区域 `i_prev` 和电量 `l_prev`
2. 将概率中不可达的目的地置零后**重归一化**（保证车辆守恒）
3. 使用 `rng.multinomial` 做随机采样分配（将连续概率转为离散整数分配）
4. 目标电量 `target_level = l_prev - E[i_prev, dst]`

**工程处理说明**:
- **概率重归一化**: 因能量不足导致部分目的地不可行时，将概率质量重分配到可行目的地。这保证了车辆守恒但改变了转移分布的形状。
- **`residual_vacant` 直接保留**: 未被调度的空车直接加入 `V_next`，不参与概率转移。与论文一致（论文假设未调度的车辆在原地不动）。
- **`multinomial` 采样**: 将连续期望值的状态转移转为随机整数分配，是离散仿真的标准做法。

---

### 2.3 Eq.(4): 换电约束 — **一致**

**论文**: $\mu_{i,l}^t \leq B_{i,l}^t$, $\sum_l \mu_{i,l}^t \leq M_i^{t,L}$, $\sum_l \mu_{i,l}^t \leq p_i$

**代码** (`etaxi_sim/models/station.py:31-32`):
```python
total_demand = int(swap_requests[: self.battery_levels].sum())
actual_swaps = min(total_demand, self.full_batteries, self.swapping_capacity)
```

三重约束 `min(需求, 满电库存, 换电容量)` **正确**。优先级策略为低电量优先，合理。

---

### 2.4 Eq.(5)(6): 充电与库存更新 — **基本一致（含候选防御性修复）**

**论文 Eq.(5)**: $y_i^{t,l} \leq \hat{M}_i^{t,l}$, $\sum_l y_i^{t,l} \leq q_i$

**论文 Eq.(6)**: $M_i^{t+1,l} = \hat{M}_i^{t,l} - y_i^{t,l} + y_i^{t,l-\hat{l}}$

**代码** (`etaxi_sim/models/station.py:92-112`):

实现通过逐 task 更新 `partial_batteries` 和 `pending_charge` 来追踪 Eq.(6)。每个被选中充电的 task：
1. 从当前 level 移除（`partial_batteries[prev] -= 1`）
2. 推进到新 level（`current_level += charge_rate`）
3. 若未充满，加入新 level 计数（`partial_batteries[new] += 1`）
4. 若已充满，`full_batteries += 1`

Eq.(5) 的充电器容量约束由充电策略（EDF/Gurobi/FCFS）保证，每个站点选中的 task 数 <= `chargers`。

**修复说明**: `partial_batteries` 和 `pending_charge` 的减操作使用 `max(0, x-1)` 防止竞态条件下的负值。

---

### 2.5 Eq.(9): 可达性约束 — **一致**

**论文**: $X_{i,i'}^{t,l} \cdot \nu_{i,i'}^t = 0$, $Y_{i,i'}^{t,l} \cdot \nu_{i,i'}^t = 0$

**代码** (`etaxi_sim/sim/core.py:111-119`):
```python
reachable = (self.reachability == 0)[:, :, None]
X = (X * reachable).astype(int, copy=False)
Y = (Y * reachable).astype(int, copy=False)
enough_energy = levels >= self.energy_consumption[:, :, None]
X = (X * enough_energy).astype(int, copy=False)
Y = (Y * enough_energy).astype(int, copy=False)
```

**正确**。额外的能量可行性约束（电量 >= 消耗）是论文隐含的物理约束。

---

### 2.6 Eq.(7)(8)(10): 目标函数 — **工程扩展，需文档说明**

**论文 Eq.(7)**: $J_{service} = \sum_t \sum_i \min(\sum_l S_{i,l}^t, D_i^t)$
**论文 Eq.(8)**: $J_{idle} = \sum_{i,i'} \omega_{i,i'} \sum_{t,l} (X + Y)$
**论文 Eq.(10)**: $\max J = J_{service} + \beta \cdot J_{idle}$

**代码** (`etaxi_sim/policies/reposition.py:464-469`):

与论文的差异：

| 项目 | 论文 | 代码 | 说明 |
|------|------|------|------|
| 时间折扣 | 无 | `0.95**k` | MPC 标准做法，避免远期不确定性 |
| 低电量奖励 | 无 | `low_energy_swap_bonus * Σ(low_y)` | 额外激励低电量车辆换电 |
| 空驶距离 | $\omega_{i,i'}$ | `energy_consumption[i,j]` | taxi_zones 模式下为 0/1，非真实距离 |
| 规划范围 | 全 horizon | MPC 滚动窗口 (4 slots) | 计算可行性权衡 |

这些差异属于**合理的工程优化**，不影响核心逻辑。

---

### 2.7 Eq.(12)(13)(14): 充电调度 — **一致（含软约束近似）**

**论文 Eq.(13)**: $\min \max_t (\sum_{\tau_i} x_i^t P + D_j)$
**论文 Eq.(14)**: $\sum_{t=A_i}^{D_i-1} x_i^t = C_i$

**代码** (`etaxi_sim/policies/charging.py:107-191`):

Gurobi 模型构建：
- 二进制变量 $x_{idx,t}$，约束 `avail_start <= t < avail_end`
- Eq.(14) 实现为**软约束** `Σx + slack = remaining_slots`，目标含 `miss_penalty * Σslack`
- 充电器容量约束 `Σx <= cap` per station per slot
- 峰值最小化 `z >= base_kw + power * Σx` for each (station, slot)

**已知设计决策**: Eq.(14) 为软约束（公式核对表已记录），避免资源不足时模型不可行。

---

### 2.8 转移概率估计 — **已知近似**

**代码** (`etaxi_sim/data/preprocess.py:29-144`):

| 参数 | 论文定义 | 代码近似 |
|------|---------|---------|
| $P_{i',i}^t$ | 载客车从 $i'$ 到 $i$ 仍载客 | 行程时长 > 1 slot 的 OD 分布 |
| $Q_{i',i}^t$ | 载客车从 $i'$ 到 $i$ 变空车 | 行程时长 ≤ 1 slot 的 OD 分布 |
| $\tilde{P}_{i',i}^t$ | 空车从 $i'$ 接客到 $i$，变载客 | `pick × (1-comp) × od` 校准 |
| $\tilde{Q}_{i',i}^t$ | 空车从 $i'$ 到 $i$，仍空车 | `pick × comp × od + (1-pick) × δ(i=i')` |

归一化条件 $\sum_i (P+Q) = 1$, $\sum_i (\tilde{P}+\tilde{Q}) = 1$ 在代码中得到保证。

---

## 3. 工程扩展（与论文的非bug差异）

### ISSUE-1: 等待队列机制

**论文**: 未换电的车辆 $H_{i,l}^t = B_{i,l}^t - \mu_{i,l}^t$ 直接回到空车 $V_{i,l}^{t+1}$。

**代码** (`core.py:136-178`): 未换电的车辆进入 `waiting_queue`，下一时步继续尝试换电。等待中的车辆不在 $V$ 或 $O$ 中，但被单独计入总车辆数。

**影响**: 比论文更现实（车辆不会因一次失败就放弃换电），但引入了论文模型之外的状态。车辆总数守恒通过 `vacant + occupied + waiting` 维护。

### ISSUE-2: MPC 中满电电池只减不增

**代码** (`reposition.py:410-414`):
```python
full_stock[(k+1, j)] == full_stock[(k, j)] - Σmu
```

规划窗口内不考虑充电回补满电电池。在短规划窗口（4 slots = 1 hour）内，由于充电需要多个时段，这个近似是合理的，但对高充电速率或长规划窗口可能导致悲观估计。

### ISSUE-3: 目标函数扩展项

时间折扣（`0.95^k`）和低电量换电奖励（`low_energy_swap_bonus`）是论文中没有的工程优化项，但不改变核心优化目标的方向。

### ISSUE-4: 空驶距离近似

当 `distance_mode=taxi_zones` 时，能量消耗矩阵为 0/1（相邻=1，同区=0），与论文的 $\omega_{i,i'}$（真实距离）有差距。这在对应的 idle cost 权重调参时需要考虑。

### ISSUE-5: P_tilde 语义近似

$\tilde{P}_{i',i}^t$ 使用载客行程的完成率 (`comp`) 作为空车接客后的状态转移参数。严格来说，空车刚接到客，新行程不太可能在一个时段内完成，但这取决于行程时长分布。此近似已在公式核对表中标注。

---

## 4. 候选修复（需在 main 分支确认）

### FIX-1: `station.py` 中 `apply_charging` 防御性计数（候选）

**文件**: `etaxi_sim/models/station.py:92-112`

**问题**: `partial_batteries` 和 `pending_charge` 的减操作以前使用 `if > 0` 条件保护，但加操作无条件执行。理论上两者应始终同步（每个 task 在某一 level 有且仅有一个对应的电池计数），但防御性编程建议使用 `max(0, x-1)` 确保不会因潜在的竞态或时序问题导致负值。

**候选修复**: 改用无条件 `max(0, x-1)` 确保减操作总是执行，同时保证不会产生负值。  
**状态**: 该项需以 `main` 实际代码为准重新核验，不应仅依据分支提交文字判断“已修复”。

---

## 5. 验证结果确认（适用范围）

根据 `scripts/validate_consistency.py` 的实测结果（给定配置）：

```
车辆总量守恒: vehicle_total_min = vehicle_total_max = 3756
电池总量守恒: battery_total_min = battery_total_max = 5300
Eq.(4) 约束违规: 0
Eq.(5) 约束违规: 0
Eq.(9) 约束违规: 0
转移矩阵归一化误差: < 4.44e-16（浮点精度级别）
```

---

## 6. 建议后续工作

1. **文档更新**: 将上述 ISSUE-1 至 ISSUE-5 的工程扩展说明添加到技术规格文档中
2. **P0 稳定性诊断**: 单独追踪 `gurobi_reposition_policy` 的 `Status/SolCount`，先解决 3756 场景卡死边界条件
3. **指标诊断**: 拆解 `charging_deadline_miss_ratio` 的分子分母，确认是否存在分母放大效应
4. **通过 gate 再全量**: 先跑诊断 episode，确认稳定后再启动 162-run 全量矩阵
5. **P_tilde 改进**: 考虑使用独立的空车接客数据或更精细的模型估计 $\tilde{P}$
6. **MPC 充电回补**: 在规划窗口较长时，考虑在 `full_stock` 动态中加入充电完成的回补项
7. **真实距离**: 接入真实路网距离矩阵替代 taxi_zones 的 0/1 近似
