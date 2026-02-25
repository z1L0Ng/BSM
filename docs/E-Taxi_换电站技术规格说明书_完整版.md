# E-Taxi 换电站协同系统技术规格说明书（完整版）

> **版本说明**：本文档已与 `docs/E_taxi_Battery_Swapping_Stations.pdf`（创建时间：2026-02-25）对齐，补充并校正了状态转移、换电/充电约束、调度模型以及相关工作与参考文献。

---

## 1. 项目概述

本项目旨在构建一个针对电动出租车（E-taxi）车队与电池换电站（BSS）的协同仿真与优化系统。

系统需要解决两个核心耦合问题：

1. **车队调度优化**：决定何时何地将出租车调度去载客或换电，以最大化服务质量并最小化空驶里程
2. **充电排程优化**：在换电站内部安排电池充电计划，以满足换电需求并最小化电网峰值负荷

---

## 2. 系统环境与实体定义 (System Entities)

### 2.1 时空模型

- **空间** ($m$)：城市被划分为 $m$ 个区域，每个区域至少配备一个换电站
- **时间** ($t$)：一天被离散化为多个等长的时间片 $t$
  - $t'$ 表示当前时刻
  - 优化周期为 $[t', t' + T - 1]$，共 $T$ 个时段

### 2.2 电动出租车 (E-Taxi)

#### 2.2.1 电量状态

- **离散化等级** ($L$)：电池电量被离散化为 $L$ 个等级（从 0 到 $L$）
- **电量表示**：$E^t \in \{0, 1, ..., L\}$ 表示时刻 $t$ 的电量等级
- **能量消耗模型**：
  - 行驶消耗：$E^{t+1} = E^t - E_{i,i'}^t$（从区域 $i$ 移动到 $i'$）
  - 其中 $E_{i,i'}^t$ 为从区域 $i$ 到 $i'$ 在时段 $t$ 的能量消耗

#### 2.2.2 车辆状态分类

在时刻 $t$ 开始时的车辆状态：

- **$V_{i,l}^t$**：空车（Vacant），在区域 $i$，电量等级 $l$
- **$O_{i,l}^t$**：载客车（Occupied），在区域 $i$，电量等级 $l$

#### 2.2.3 中间状态变量

调度后在时段 $t$ 内的车辆分类：

- **$B_{i,l}^t$**：正在换电站等待/进行换电的车辆
- **$S_{i,l}^t$**：被派遣去接客的空车
- **$U_{i,l}^t$**：正在运送乘客的载客车

#### 2.2.4 状态流转关系

```
时刻 t 开始：V_{i,l}^t, O_{i,l}^t
      ↓ (调度决策 X, Y)
时段 t 内：  B_{i,l}^t (换电), S_{i,l}^t (接客), U_{i,l}^t (载客)
      ↓ (概率转移)
时刻 t+1 开始：V_{i,l}^{t+1}, O_{i,l}^{t+1}
```

### 2.3 电池换电站 (BSS)

#### 2.3.1 硬件组件

- **换电机器（Swapping Machines）**：
  - 容量 $p_i$：站点 $i$ 每时段最大换电服务数量
  - 功能：物理更换电池包（约 3-5 分钟/次）

- **充电器（Chargers）**：
  - 数量 $q_i$：站点 $i$ 的充电桩数量
  - 功能：为亏电电池充电

#### 2.3.2 电池库存状态

在时刻 $t$ 开始时：

- **$M_i^{t,L}$**：满电电池数量（电量等级为 $L$）
  - **关键**：仅此状态可用于换电服务
  
- **$M_i^{t,l}$** ($l < L$)：其他电量等级的电池库存

- **$\hat{M}_i^{t,l}$**：待充电的亏电电池数量（电量等级 $l$）
  - 包括：初始库存 + 换电后新收回的亏电电池

#### 2.3.3 换电站服务流程

```
车辆到达 (B_{i,l}^t) 
    ↓
检查条件：满电电池 (M_i^{t,L})、换电机器容量 (p_i)
    ↓
换电成功 (μ_{i,l}^t) → 车辆获得满电电池 (H_{i,L}^t)
    ↓
亏电电池入库 (Mˆ_i^{t,l}) → 加入充电队列
```

---

## 3. 核心决策变量与逻辑 (Core Logic & Variables)

### 3.1 车队重定位决策 (Fleet Repositioning)

#### 输入信息
- 当前车队分布：$V_{i,l}^t$, $O_{i,l}^t$
- 乘客需求预测：$D_i^t$（区域 $i$ 在时段 $t$ 的预期乘客数）
- 电池库存状态：$M_i^{t,L}$

#### 决策变量

- **$X_{i,i'}^{t,l}$**：在时刻 $t$ 开始时，将电量 $l$ 的空车从 $i$ 调度到 $i'$ 进行载客服务的数量
- **$Y_{i,i'}^{t,l}$**：在时刻 $t$ 开始时，将电量 $l$ 的空车从 $i$ 调度到 $i'$ 进行电池更换的数量

#### 物理约束

##### 约束 1：可达性约束

定义 $\nu_{i,i'}^t \in \{0, 1\}$：
- $\nu_{i,i'}^t = 0$：区域 $i$ 到 $i'$ 可在时段 $t$ 内到达
- $\nu_{i,i'}^t = 1$：区域 $i$ 到 $i'$ 无法在时段 $t$ 内到达

约束：
$$
X_{i,i'}^{t,l} \cdot \nu_{i,i'}^t = 0, \quad Y_{i,i'}^{t,l} \cdot \nu_{i,i'}^t = 0
$$

**物理意义**：车辆最高速度和时段长度限制了可调度的区域范围。

##### 约束 2：车辆守恒

调度后的车辆分布必须等于调度前的总和：

$$
B_{i,l}^t = \sum_{i'=1}^{m} Y_{i',i}^{t,l}
$$

$$
S_{i,l}^t = \sum_{i'=1}^{m} X_{i',i}^{t,l}
$$

**说明**：
- $B_{i,l}^t$：到达区域 $i$ 换电站、电量为 $l$ 的车辆总数
- $S_{i,l}^t$：到达区域 $i$ 准备接客、电量为 $l$ 的车辆总数

---

### 3.2 换电站服务逻辑 (Station Operations)

#### 3.2.1 换电服务约束

换电成功量 $\mu_{i,l}^t$ 受三重约束：

1. **到达车辆数**：$\mu_{i,l}^t \leq B_{i,l}^t$
2. **满电电池库存**：$\sum_l \mu_{i,l}^t \leq M_i^{t,L}$
3. **换电机器容量**：$\sum_l \mu_{i,l}^t \leq p_i$

数学表示：
$$
\mu_{i,l}^t \leq B_{i,l}^t
$$
$$
\sum_{l} \mu_{i,l}^t \leq M_i^{t,L}
$$
$$
\sum_{l} \mu_{i,l}^t \leq p_i
$$

#### 3.2.2 换电后的车辆状态

- **未换电车辆**（电量保持 $l$）：
  $$
  H_{i,l}^t = B_{i,l}^t - \mu_{i,l}^t, \quad \forall l < L
  $$

- **换电成功车辆**（电量变为 $L$）：
  $$
  H_{i,L}^t = \sum_{l} \mu_{i,l}^t
  $$

#### 3.2.3 电池库存更新

换电后，换下来的亏电电池进入待充电库存：

$$
\hat{M}_i^{t,l} = M_i^{t,l} + \mu_{i,l}^t
$$

**解释**：
- $M_i^{t,l}$：原有库存中电量 $l$ 的电池
- $\mu_{i,l}^t$：本时段换下的电量 $l$ 的电池

---

### 3.3 充电排程决策 (Charging Scheduling)

#### 3.3.1 充电任务定义

每个电池充电任务 $\tau_i = (A_i, C_i, D_i)$：

- **$A_i$**：电池可开始充电的时刻（到达时间）
- **$C_i$**：充满所需的时间片数
- **$D_i$**：截止时间（Deadline），必须在此之前充满

**计算示例**：
- 电池当前电量：30%（等级 3）
- 满电需求：100%（等级 10）
- 每时段充电增量：$\hat{l} = 2$ 个等级
- 则 $C_i = \lceil (10-3)/2 \rceil = 4$ 个时段

#### 3.3.2 充电任务集合

- **$\Psi_j$**：换电站 $j$ 的充电任务集合
- **$\Psi = \bigcup_{j=1}^{m} \Psi_j$**：整个系统的充电任务集合

#### 3.3.3 决策变量

论文中将每个任务的充电计划写为向量：
$$
x_i \in \{0, 1\}^{1 \times T}
$$

其中标量 $x_i^t$ 表示任务 $\tau_i$ 在时刻 $t'+t$ 是否充电：

- $x_i^t = 1$：在时刻 $t' + t$ 充电
- $x_i^t = 0$：在时刻 $t' + t$ 不充电

时间索引满足：
$$
t' \le A_i, C_i, D_i \le t' + T - 1
$$

#### 3.3.4 充电特性（❗新增内容）

**关键特性**：充电过程是**可抢占**（preemptive）和**可迁移**（migratable）的

这意味着：

1. **可抢占性**：
   - 正在充电的电池可以暂停充电
   - 让出充电器给更紧急的任务（如deadline更近的电池）
   - 之后可以恢复充电

2. **可迁移性**：
   - 电池可以在同一换电站内的不同充电器之间切换
   - 无需固定在某个特定充电器上

**实现影响**：
- 调度算法需要支持动态重分配充电器
- 需要追踪每个电池的累计充电时间（而非连续充电）
- 优先级队列可用于处理不同deadline的任务

#### 3.3.5 充电约束

##### 约束 1：充电完成约束

在截止时间前必须充够 $C_i$ 个时段：

$$
\sum_{t=A_i}^{D_i-1} x_i^t = C_i
$$

**说明**：累计充电时段数必须达到要求，但不需要连续。

##### 约束 2：充电器容量约束

同一时刻充电的电池总数不能超过充电器数量：

$$
\sum_{\tau_i \in \Psi_j} x_i^t \leq q_j, \quad \forall t \in [t', t'+T-1]
$$

**说明**：这是站点 $j$ 的硬件限制。

#### 3.3.6 电池库存动态更新（❗新增内容）

在每个时刻 $t+1$ 开始时，电池库存更新为：

$$
M_i^{t+1,l} = \hat{M}_i^{t,l} - y_i^{t,l} + y_i^{t,l-\hat{l}}
$$

其中：
- $y_i^{t,l}$：在时段 $t$ 选择充电的、电量为 $l$ 的电池数量
- $\hat{l}$：单个时段的充电增量（电量等级）
- $y_i^{t,l-\hat{l}}$：在时段 $t$ 充电后从等级 $l-\hat{l}$ 升级到 $l$ 的电池数

**约束条件**：
$$
y_i^{t,l} \leq \hat{M}_i^{t,l}
$$
$$
\sum_l y_i^{t,l} \leq q_i
$$

**控制边界说明（与论文一致）**：
- 车队运营方不直接决定站内细粒度充电序列，$y_i^{t,l}$ 用于刻画站内充电执行结果
- 充电调度器在满足 Eq. (5)(6)(12)(14) 前提下，优化峰值功率目标

**注意**：
- 当 $l = L$ 时，$y_i^{t,L} = 0$（满电电池无需充电）
- 当 $l - \hat{l} < 0$ 时，$y_i^{t,l-\hat{l}} = 0$

---

## 4. 状态转移概率模型（❗新增内容）

### 4.1 概率参数定义

这是论文中最关键但技术文档缺失的部分：

#### 4.1.1 载客车辆的状态转移

- **$P_{i',i}^t$**：在时段 $t$ 内，一辆载客车从区域 $i'$ 到达区域 $i$，且在时段结束时仍然载客（即尚未送达乘客）的概率

- **$Q_{i',i}^t$**：在时段 $t$ 内，一辆载客车从区域 $i'$ 到达区域 $i$，且在时段结束时变为空车（即已送达乘客）的概率

**归一化条件**：
$$
\sum_{i=1}^{m} (P_{i',i}^t + Q_{i',i}^t) = 1
$$

#### 4.1.2 空车的状态转移

- **$\tilde{P}_{i',i}^t$**：在时段 $t$ 内，一辆被派遣接客的空车从区域 $i'$ 到达区域 $i$，且在时段结束时已接到乘客（变为载客车）的概率

- **$\tilde{Q}_{i',i}^t$**：在时段 $t$ 内，一辆被派遣接客的空车从区域 $i'$ 到达区域 $i$，但在时段结束时仍为空车（未接到乘客）的概率

**归一化条件**：
$$
\sum_{i=1}^{m} (\tilde{P}_{i',i}^t + \tilde{Q}_{i',i}^t) = 1
$$

### 4.2 状态转移方程

#### 4.2.1 载客车辆的演化

$$
O_{i,l}^{t+1} = \sum_{i'=1}^{m} \tilde{P}_{i',i}^t \cdot S_{i',l+E_{i',i}^t}^t + \sum_{i'=1}^{m} P_{i',i}^t \cdot U_{i',l+E_{i',i}^t}^t
$$

**解释**：
- 第一项：在时段 $t$ 被派遣接客且成功接到乘客的车辆
- 第二项：在时段 $t$ 持续载客且未送达的车辆
- 能量消耗：车辆从 $i'$ 移动到 $i$ 消耗 $E_{i',i}^t$，因此原始电量为 $l + E_{i',i}^t$

#### 4.2.2 空车的演化

$$
V_{i,l}^{t+1} = \sum_{i'=1}^{m} \tilde{Q}_{i',i}^t \cdot S_{i',l+E_{i',i}^t}^t + \sum_{i'=1}^{m} Q_{i',i}^t \cdot U_{i',l+E_{i',i}^t}^t + H_{i,l}^t
$$

**解释**：
- 第一项：被派遣接客但未接到乘客的车辆
- 第二项：完成乘客运送后变为空车的车辆
- 第三项：在换电站处理后的车辆（包括换电成功和未换电的）

### 4.3 概率参数估计方法

**数据驱动方法**：

从历史轨迹数据中统计：

1. **$P_{i',i}^t$ 估计**：
   ```
   P_{i',i}^t = (时段t内从i'出发的载客车，到达i后仍载客的数量) / 
                (时段t内从i'出发的所有载客车数量)
   ```

2. **$\tilde{P}_{i',i}^t$ 估计**：
   ```
   P̃_{i',i}^t = (时段t内从i'出发接客的空车，到达i后已载客的数量) / 
                (时段t内从i'出发接客的所有空车数量)
   ```

**模型驱动方法**：

基于出行时长分布和需求分布建模：
- 使用泊松过程建模乘客到达
- 使用对数正态分布建模出行时长
- 结合重力模型估计OD流量

---

## 5. 优化目标函数 (Optimization Objectives)

Agent 需针对以下目标函数进行求解或强化学习训练：

### 5.1 目标一：车队运营优化

最大化服务质量与最小化空驶成本的加权和：

$$
\max_{X,Y} J = J_{\text{service}} + \beta \cdot J_{\text{idle}}
$$

其中 $\beta < 0$ 为负权重参数。

#### 5.1.1 服务质量 ($J_{\text{service}}$)

实际服务的乘客数量：

$$
J_{\text{service}} = \sum_{t=t'}^{t'+T-1} \sum_{i=1}^{m} \min \left( \underbrace{\sum_{l} S_{i,l}^t}_{\text{供给}}, \underbrace{D_i^t}_{\text{需求}} \right)
$$

**解释**：
- 供给：区域 $i$ 在时段 $t$ 可用于接客的空车总数
- 需求：区域 $i$ 在时段 $t$ 的预期乘客数
- 实际服务量 = min(供给, 需求)

#### 5.1.2 空驶惩罚 ($J_{\text{idle}}$)

车辆为了换电或接客产生的空跑距离：

$$
J_{\text{idle}} = - \sum_{i,i'} \omega_{i,i'} \sum_{t,l} (X_{i,i'}^{t,l} + Y_{i,i'}^{t,l})
$$

其中：
- $\omega_{i,i'}$：区域 $i$ 到 $i'$ 的距离
- 负号：在总目标中 $\beta < 0$，因此这是一个惩罚项

**优化意义**：
- 减少车辆无效移动
- 降低能量消耗
- 减少机会成本（空跑时无法接客）

---

### 5.2 目标二：电网友好型充电（❗补充基础负荷）

在满足电池需求截止时间的前提下，最小化峰值充电功率：

$$
\min \max_{t \in [t', t'+T-1]} \left( P \cdot \sum_{\tau_i \in \Psi} x_i^t + D_j \right)
$$

并满足充电完成约束（论文 Eq. (14)）：
$$
\sum_{t=A_i}^{D_i-1} x_i^t = C_i
$$

其中：
- $P$：单个电池的充电功率（kW）
- $x_i^t$：任务 $\tau_i$ 在时刻 $t$ 是否充电（0或1）
- **$D_j$**：换电站 $j$ 的基础负荷（background power demand）

**$D_j$ 说明（技术文档中缺失）**：
- 换电站除了充电外的其他电力消耗
- 包括：照明、空调、控制系统、监控设备等
- 通常是相对稳定的基线功率

**优化意义**：
- **削峰填谷**：避免多个电池同时充电造成的峰值
- **减少电费**：许多地区采用峰值电价（demand charge）
- **电网稳定**：降低对配电网的冲击

**符号统一说明**：
- 论文 III.B 中充电器数量使用 $q_i$
- 论文 IV 中个别位置写作 $p_j$（上下文含义仍是“站点充电器数量”）
- 本说明书统一记作 $q_j$，避免与换电服务容量 $p_i$ 混淆

---

## 6. 完整约束条件总结

### 6.1 车队调度约束

```
Eq. (1): B_{i,l}^t = Σ_{i'} Y_{i',i}^{t,l}
         S_{i,l}^t = Σ_{i'} X_{i',i}^{t,l}

Eq. (2): O_{i,l}^{t+1} = Σ_{i'} P̃_{i',i}^t S_{i',l+E_{i',i}^t}^t + Σ_{i'} P_{i',i}^t U_{i',l+E_{i',i}^t}^t

Eq. (3): V_{i,l}^{t+1} = Σ_{i'} Q̃_{i',i}^t S_{i',l+E_{i',i}^t}^t + Σ_{i'} Q_{i',i}^t U_{i',l+E_{i',i}^t}^t + H_{i,l}^t

Eq. (9): X_{i,i'}^{t,l} · ν_{i,i'}^t = 0
         Y_{i,i'}^{t,l} · ν_{i,i'}^t = 0
```

### 6.2 换电站约束

```
Eq. (4): μ_{i,l}^t ≤ B_{i,l}^t
         Σ_l μ_{i,l}^t ≤ M_i^{t,L}
         Σ_l μ_{i,l}^t ≤ p_i

电池库存更新:
         Mˆ_i^{t,l} = M_i^{t,l} + μ_{i,l}^t
```

### 6.3 充电调度约束

```
Eq. (12): Σ_{t=A_i}^{D_i-1} x_i^t = C_i
Eq. (14): Σ_{t=A_i}^{D_i-1} x_i^t = C_i

Eq. (5):  y_i^{t,l} ≤ Mˆ_i^{t,l}
          Σ_l y_i^{t,l} ≤ q_i

Eq. (6):  M_i^{t+1,l} = Mˆ_i^{t,l} - y_i^{t,l} + y_i^{t,l-l̂}
```

---

## 7. 算法实现流程 (Implementation Workflow)

### 7.1 初始化阶段

```python
# 伪代码
def initialize_system(config):
    # 1. 空间初始化
    m = config.num_regions
    stations = [BatteryStation(i) for i in range(m)]
    
    # 2. 车队初始化
    fleet_state = {
        'vacant': np.zeros((m, L+1)),  # V_{i,l}^0
        'occupied': np.zeros((m, L+1))  # O_{i,l}^0
    }
    
    # 3. 电池库存初始化
    for station in stations:
        station.battery_inventory = {
            'full': config.initial_full_batteries,      # M_i^{0,L}
            'partial': {l: 0 for l in range(L)}        # M_i^{0,l}
        }
        station.pending_charge = {l: [] for l in range(L)}  # Mˆ_i^{0,l}
    
    # 4. 加载转移概率矩阵
    transition_probs = load_transition_probabilities(historical_data)
    # P_{i',i}^t, Q_{i',i}^t, P̃_{i',i}^t, Q̃_{i',i}^t
    
    # 5. 加载需求预测
    demand_forecast = load_demand_forecast()  # D_i^t
    
    return fleet_state, stations, transition_probs, demand_forecast
```

### 7.2 主循环：时间步仿真

```python
def simulation_loop(T, fleet_state, stations, transition_probs, demand_forecast):
    
    charging_task_pool = []  # Ψ：全局充电任务集合
    
    for t in range(T):
        print(f"=== Time Step {t} ===")
        
        # ========================================
        # 步骤 1: 车队控制决策
        # ========================================
        X, Y = fleet_repositioning_optimization(
            fleet_state=fleet_state,
            demand=demand_forecast[t],
            battery_availability=[s.battery_inventory['full'] for s in stations],
            objective='max_service_min_idle'
        )
        # 输出: X_{i,i'}^{t,l}, Y_{i,i'}^{t,l}
        
        # ========================================
        # 步骤 2: 状态更新 - 车辆移动
        # ========================================
        # 2.1 计算到达各站点/区域的车辆
        B, S, U = compute_arriving_vehicles(X, Y, fleet_state)
        # B_{i,l}^t = Σ_{i'} Y_{i',i}^{t,l}
        # S_{i,l}^t = Σ_{i'} X_{i',i}^{t,l}
        # U_{i,l}^t = O_{i,l}^t (继续载客的车辆)
        
        # 2.2 更新能量消耗
        energy_consumption = compute_energy_cost(X, Y, distance_matrix)
        
        # ========================================
        # 步骤 3: 换电站交互
        # ========================================
        for i, station in enumerate(stations):
            # 3.1 处理到达换电站的车辆
            arriving_vehicles = B[i]  # B_{i,l}^t
            
            # 3.2 换电服务（受约束）
            swapped, not_swapped = station.perform_swapping(
                arriving_vehicles=arriving_vehicles,
                constraints={
                    'full_batteries': station.battery_inventory['full'],  # M_i^{t,L}
                    'capacity': station.swapping_capacity                  # p_i
                }
            )
            # 输出: μ_{i,l}^t, H_{i,l}^t
            
            # 3.3 生成充电任务
            new_tasks = station.generate_charging_tasks(
                depleted_batteries=swapped,
                current_time=t,
                deadline_policy='T_hours_ahead'
            )
            charging_task_pool.extend(new_tasks)
            # 每个任务: τ_i = (A_i=t, C_i, D_i)
        
        # ========================================
        # 步骤 4: 充电调度优化
        # ========================================
        charging_schedule = charging_scheduler_optimization(
            task_pool=charging_task_pool,
            stations=stations,
            objective='minimize_peak_load',
            current_time=t
        )
        # 输出: x_i^t ∈ {0,1} for all τ_i ∈ Ψ
        
        # 4.1 执行充电计划
        for station in stations:
            station.execute_charging(charging_schedule, time_slot=t)
            # 更新: M_i^{t+1,l} 根据 Eq. (6)
        
        # 4.2 移除已完成的充电任务
        charging_task_pool = [
            task for task in charging_task_pool 
            if not task.is_completed()
        ]
        
        # ========================================
        # 步骤 5: 概率转移 - 更新下一时刻状态
        # ========================================
        fleet_state = state_transition(
            S=S, U=U, H=[s.vehicles_after_swapping for s in stations],
            transition_probs=transition_probs[t],
            energy_consumption=energy_consumption
        )
        # 更新: V_{i,l}^{t+1}, O_{i,l}^{t+1} 根据 Eq. (2), (3)
        
        # ========================================
        # 步骤 6: 性能指标记录
        # ========================================
        record_metrics(t, fleet_state, stations, X, Y)
    
    return simulation_results
```

### 7.3 关键子函数详解

#### 7.3.1 车队重定位优化

```python
def fleet_repositioning_optimization(fleet_state, demand, battery_availability, objective):
    """
    求解优化问题 Eq. (10):
        max J = J_service + β·J_idle
    subject to Eq. (1), (2), (3), (9)
    """
    # 方法选项:
    # 1. 线性规划 (LP/MILP) - 适用于小规模问题
    # 2. 启发式算法 (Greedy, Genetic Algorithm)
    # 3. 强化学习 (DQN, A3C, PPO)
    
    if objective == 'max_service_min_idle':
        # 构建优化模型
        model = build_repositioning_model(
            vacant_vehicles=fleet_state['vacant'],  # V_{i,l}^t
            demand=demand,                           # D_i^t
            battery_availability=battery_availability,
            beta=config.idle_weight
        )
        
        # 求解
        solution = solver.solve(model)
        
        # 提取决策变量
        X = solution['pickup_dispatch']   # X_{i,i'}^{t,l}
        Y = solution['swapping_dispatch'] # Y_{i,i'}^{t,l}
        
        return X, Y
```

#### 7.3.2 换电服务逻辑

```python
class BatteryStation:
    def perform_swapping(self, arriving_vehicles, constraints):
        """
        执行换电服务，受 Eq. (4) 约束
        """
        full_batteries = constraints['full_batteries']  # M_i^{t,L}
        capacity = constraints['capacity']               # p_i
        
        swapped = {}
        not_swapped = {}
        
        total_demand = sum(arriving_vehicles.values())
        
        # 实际换电量受三重约束
        actual_swaps = min(
            total_demand,
            full_batteries,
            capacity
        )
        
        # 优先级策略（可配置）
        priority_order = sorted(
            arriving_vehicles.items(),
            key=lambda x: x[0]  # 按电量等级排序，低电量优先
        )
        
        remaining_capacity = actual_swaps
        for energy_level, count in priority_order:
            if remaining_capacity == 0:
                break
            
            swappable = min(count, remaining_capacity)
            swapped[energy_level] = swappable
            not_swapped[energy_level] = count - swappable
            remaining_capacity -= swappable
        
        # 更新库存
        self.battery_inventory['full'] -= actual_swaps
        
        # 换下的电池进入待充电队列
        for energy_level, count in swapped.items():
            self.pending_charge[energy_level] += count
            # Mˆ_i^{t,l} = M_i^{t,l} + μ_{i,l}^t
        
        # 换电成功的车辆电量变为 L
        vehicles_after = {
            **not_swapped,         # H_{i,l}^t (未换电)
            L: actual_swaps        # H_{i,L}^t (换电成功)
        }
        
        self.vehicles_after_swapping = vehicles_after
        
        return swapped, not_swapped
```

#### 7.3.3 充电任务生成

```python
def generate_charging_tasks(self, depleted_batteries, current_time, deadline_policy):
    """
    为换下的亏电电池生成充电任务 τ_i = (A_i, C_i, D_i)
    """
    new_tasks = []
    
    for energy_level, count in depleted_batteries.items():
        for _ in range(count):
            # 计算充电时长
            energy_needed = L - energy_level
            charge_time = math.ceil(energy_needed / charging_rate_per_slot)  # C_i
            
            # 设置截止时间
            if deadline_policy == 'T_hours_ahead':
                deadline = current_time + T_deadline  # D_i
            elif deadline_policy == 'adaptive':
                # 基于需求预测动态调整
                deadline = self.estimate_next_demand_peak(current_time)
            
            task = ChargingTask(
                battery_id=uuid.uuid4(),
                arrival_time=current_time,           # A_i
                required_charge_units=charge_time,   # C_i
                deadline=deadline,                   # D_i
                station_id=self.id
            )
            
            new_tasks.append(task)
    
    return new_tasks
```

#### 7.3.4 充电调度优化

```python
def charging_scheduler_optimization(task_pool, stations, objective, current_time):
    """
    求解优化问题 Eq. (13):
        min max_t (P · Σ_{τ_i ∈ Ψ} x_i^t + D_j)
    subject to Eq. (12), (5)
    """
    if objective == 'minimize_peak_load':
        # 方法选项:
        # 1. 整数线性规划 (ILP)
        # 2. 启发式: Earliest Deadline First (EDF)
        # 3. 启发式: Load Balancing Algorithm
        
        # 使用EDF启发式
        schedule = earliest_deadline_first_scheduler(
            tasks=task_pool,
            chargers={s.id: s.num_chargers for s in stations},
            horizon=T,
            current_time=current_time
        )
        
        return schedule

def earliest_deadline_first_scheduler(tasks, chargers, horizon, current_time):
    """
    EDF调度算法实现
    利用充电的可抢占性和可迁移性
    """
    schedule = {task.id: np.zeros(horizon, dtype=int) for task in tasks}
    
    for t in range(horizon):
        # 1. 获取当前可充电的任务
        eligible_tasks = [
            task for task in tasks
            if task.arrival_time <= current_time + t < task.deadline
            and sum(schedule[task.id]) < task.required_charge_units
        ]
        
        # 2. 按deadline排序（EDF策略）
        eligible_tasks.sort(key=lambda task: task.deadline)
        
        # 3. 为每个站点分配充电器
        for station_id, capacity in chargers.items():
            station_tasks = [
                task for task in eligible_tasks
                if task.station_id == station_id
            ]
            
            # 分配充电器（可抢占式）
            assigned = 0
            for task in station_tasks:
                if assigned >= capacity:
                    break
                schedule[task.id][t] = 1
                assigned += 1
    
    return schedule
```

#### 7.3.5 状态转移函数

```python
def state_transition(S, U, H, transition_probs, energy_consumption):
    """
    根据 Eq. (2), (3) 更新车辆状态
    """
    V_next = np.zeros((m, L+1))
    O_next = np.zeros((m, L+1))
    
    P = transition_probs['P']      # P_{i',i}^t
    Q = transition_probs['Q']      # Q_{i',i}^t
    P_tilde = transition_probs['P_tilde']  # P̃_{i',i}^t
    Q_tilde = transition_probs['Q_tilde']  # Q̃_{i',i}^t
    
    for i in range(m):
        for l in range(L+1):
            # 载客车状态转移 (Eq. 2)
            for i_prime in range(m):
                energy_consumed = energy_consumption[i_prime, i]
                original_energy = l + energy_consumed
                
                if 0 <= original_energy <= L:
                    O_next[i, l] += (
                        P_tilde[i_prime, i] * S[i_prime, original_energy] +
                        P[i_prime, i] * U[i_prime, original_energy]
                    )
            
            # 空车状态转移 (Eq. 3)
            for i_prime in range(m):
                energy_consumed = energy_consumption[i_prime, i]
                original_energy = l + energy_consumed
                
                if 0 <= original_energy <= L:
                    V_next[i, l] += (
                        Q_tilde[i_prime, i] * S[i_prime, original_energy] +
                        Q[i_prime, i] * U[i_prime, original_energy]
                    )
            
            # 加上换电站返回的车辆
            V_next[i, l] += H[i][l]
    
    return {'vacant': V_next, 'occupied': O_next}
```

---

## 8. 数据结构参考 (Data Schema)

### 8.1 完整JSON Schema

```json
{
  "system_config": {
    "num_regions": "int (m)",
    "num_energy_levels": "int (L)",
    "time_horizon": "int (T)",
    "time_slot_length": "int (minutes)",
    "charging_rate_per_slot": "float (energy levels per slot, l̂)"
  },
  
  "taxi_fleet_state": {
    "timestamp": "int (t)",
    "vacant_vehicles": {
      "location_id": "int (1 to m)",
      "energy_level": "int (0 to L)",
      "count": "int (V_{i,l}^t)"
    },
    "occupied_vehicles": {
      "location_id": "int (1 to m)",
      "energy_level": "int (0 to L)",
      "count": "int (O_{i,l}^t)"
    }
  },
  
  "station_state": {
    "station_id": "int (1 to m)",
    "swapping_capacity": "int (p_i)",
    "num_chargers": "int (q_i)",
    "battery_inventory": {
      "full_batteries": "int (M_i^{t,L})",
      "partial_batteries": {
        "energy_level": "int (l)",
        "count": "int (M_i^{t,l})"
      }
    },
    "pending_charge_queue": {
      "energy_level": "int (l)",
      "count": "int (Mˆ_i^{t,l})"
    },
    "base_power_demand": "float (D_j, kW)"
  },
  
  "charging_task": {
    "task_id": "uuid",
    "battery_id": "uuid",
    "station_id": "int",
    "arrival_time": "int (A_i)",
    "required_charge_units": "int (C_i)",
    "deadline": "int (D_i)",
    "is_preemptible": "bool (always true)",
    "is_migratable": "bool (always true)",
    "current_charge_progress": "int (Σ x_i^t)"
  },
  
  "transition_probabilities": {
    "timestamp": "int (t)",
    "P_matrix": "array[m][m] (P_{i',i}^t)",
    "Q_matrix": "array[m][m] (Q_{i',i}^t)",
    "P_tilde_matrix": "array[m][m] (P̃_{i',i}^t)",
    "Q_tilde_matrix": "array[m][m] (Q̃_{i',i}^t)"
  },
  
  "decision_variables": {
    "timestamp": "int (t)",
    "pickup_dispatch": {
      "from_location": "int (i)",
      "to_location": "int (i')",
      "energy_level": "int (l)",
      "count": "int (X_{i,i'}^{t,l})"
    },
    "swapping_dispatch": {
      "from_location": "int (i)",
      "to_location": "int (i')",
      "energy_level": "int (l)",
      "count": "int (Y_{i,i'}^{t,l})"
    }
  },
  
  "demand_forecast": {
    "timestamp": "int (t)",
    "location_id": "int (i)",
    "expected_passengers": "int (D_i^t)"
  },
  
  "distance_matrix": {
    "from_location": "int (i)",
    "to_location": "int (i')",
    "distance": "float (ω_{i,i'}, km)"
  },
  
  "reachability_matrix": {
    "timestamp": "int (t)",
    "from_location": "int (i)",
    "to_location": "int (i')",
    "is_reachable": "bool (ν_{i,i'}^t == 0)"
  },
  
  "energy_consumption_matrix": {
    "from_location": "int (i)",
    "to_location": "int (i')",
    "energy_cost": "int (E_{i,i'}^t, energy levels)"
  },
  
  "optimization_parameters": {
    "idle_weight": "float (β < 0)",
    "single_battery_power": "float (P, kW)",
    "deadline_buffer": "int (time slots)"
  }
}
```

### 8.2 关键数据流图

```
时刻 t:
  输入状态: V_{i,l}^t, O_{i,l}^t, M_i^{t,l}
       ↓
  决策变量: X_{i,i'}^{t,l}, Y_{i,i'}^{t,l}
       ↓
  中间状态: B_{i,l}^t, S_{i,l}^t, U_{i,l}^t
       ↓
  换电服务: μ_{i,l}^t → H_{i,l}^t
       ↓
  充电调度: x_i^t → 更新 M_i^{t+1,l}
       ↓
  状态转移: (概率模型) → V_{i,l}^{t+1}, O_{i,l}^{t+1}
```

---

## 9. 实现建议与注意事项

### 9.1 数值稳定性

1. **能量等级边界检查**：
   ```python
   # 确保 l + E_{i',i}^t 在有效范围内
   original_energy = l + energy_consumed
   if not (0 <= original_energy <= L):
       continue  # 跳过无效转移
   ```

2. **概率归一化**：
   ```python
   # 确保转移概率之和为1
   assert abs(sum(P[i_prime, :]) + sum(Q[i_prime, :]) - 1.0) < 1e-6
   ```

### 9.2 计算复杂度优化

1. **稀疏矩阵表示**：
   - $X_{i,i'}^{t,l}$ 和 $Y_{i,i'}^{t,l}$ 通常是稀疏的（大部分为0）
   - 使用稀疏矩阵数据结构（如scipy.sparse）

2. **状态聚合**：
   - 对于大规模问题，考虑聚合相邻区域
   - 或减少电量等级的粒度

### 9.3 求解器选择

#### 车队重定位优化
- **小规模** (m < 20, L < 10)：Gurobi, CPLEX (MILP)
- **中规模** (m < 50)：启发式算法 + 局部搜索
- **大规模**：深度强化学习 (PPO, SAC)

#### 充电调度优化
- **精确解**：ILP求解器 (Gurobi)
- **近似解**：EDF + Load Balancing
- **在线算法**：动态规划 + 优先队列

### 9.4 参数估计

#### 转移概率矩阵
```python
def estimate_transition_probabilities(trajectory_data, time_slot):
    """
    从历史轨迹数据估计 P, Q, P̃, Q̃
    """
    # 统计方法
    P = np.zeros((m, m))
    Q = np.zeros((m, m))
    
    for trajectory in trajectory_data:
        if trajectory.time_slot == time_slot:
            i_prime = trajectory.origin
            i = trajectory.destination
            
            if trajectory.status_end == 'occupied':
                P[i_prime, i] += 1
            else:
                Q[i_prime, i] += 1
    
    # 归一化
    for i_prime in range(m):
        total = P[i_prime, :].sum() + Q[i_prime, :].sum()
        if total > 0:
            P[i_prime, :] /= total
            Q[i_prime, :] /= total
    
    return P, Q
```

### 9.5 调试与验证

#### 关键不变量检查
```python
def validate_state_invariants(fleet_state, stations):
    """
    检查系统不变量
    """
    # 1. 车辆总数守恒
    total_vehicles = (
        fleet_state['vacant'].sum() +
        fleet_state['occupied'].sum()
    )
    assert total_vehicles == INITIAL_FLEET_SIZE
    
    # 2. 电池总数守恒
    total_batteries = sum(
        station.battery_inventory['full'] +
        sum(station.battery_inventory['partial'].values()) +
        sum(station.pending_charge.values())
        for station in stations
    )
    assert total_batteries == INITIAL_BATTERY_COUNT
    
    # 3. 非负性约束
    assert (fleet_state['vacant'] >= 0).all()
    assert (fleet_state['occupied'] >= 0).all()
```

---

## 10. 补充说明

### 10.1 与 2026-02-25 版论文的对齐更新

| 对齐项 | 调整内容 |
|------|--------|
| 换电约束 Eq. (4) | 改为三条独立不等式，显式对应到达车辆、满电库存、换电服务容量 |
| 充电计划变量 | 补充向量形式 $x_i \in \{0,1\}^{1\times T}$ 与索引范围 $t' \le A_i,C_i,D_i \le t'+T-1$ |
| 充电控制边界 | 明确 $y_i^{t,l}$ 用于描述站内执行结果，车队侧不直接下发站内细粒度充电序列 |
| 充电目标约束 | 在峰值目标下补充 Eq. (14) 完成约束 |
| 符号一致性 | 统一“充电器数量”符号为 $q_j$，避免与换电容量 $p_i$ 混淆 |

### 10.2 论文相关工作摘要（新增）

论文将相关研究分为三类：

1. **电动出租车换电调度**：关注车队调度、换电排队与运营成本平衡，强调“调度-换电”联合优化优于解耦策略。  
2. **换电站运行模型**：关注库存动态、服务能力与排队特性，常将车辆到达视为外生输入。  
3. **充电站调度与电网协同**：关注削峰填谷、截止时间约束与分布式负荷控制，但通常不显式建模换电库存与车队重定位。  

### 10.3 参考文献（与论文一致）

[1] P. You, S. H. Low, W. Tushar, G. Geng, C. Yuen, Y. Sun, and Z. Yang, "Scheduling of EV Battery Swapping, I: Centralized Solution," arXiv, 2016.  
[2] F. Ahmad, M. S. Alam, I. S. Alsaidan, and S. M. Shariff, "Battery Swapping Station for Electric Vehicles: Opportunities and Challenges," IET Smart Grid, 2020.  
[3] P. You, S. H. Low, L. Zhang, R. Deng, G. B. Giannakis, Y. Sun, and Z. Yang, "Scheduling of EV Battery Swapping, II: Distributed Solutions," arXiv, 2016.  
[4] H. Qiang, Y. Hu, W. Tang, and X. Zhang, "Research on Optimization Strategy of Battery Swapping for Electric Taxis," Energies, 2023.  
[5] Z. Lai and S. Li, "Towards a Multimodal Charging Network: Joint Planning of Charging Stations and Battery Swapping Stations for Electrified Ride-Hailing Fleets," arXiv, 2022.  
[6] Z. Liu, X. Ma, X. Liu, G. H. d. A. Correia, R. Shi, and W. Shang, "Optimizing Electric Taxi Battery Swapping Stations Featuring Modular Battery Swapping: A Data-Driven Approach," Applied Sciences, 2023.  
[7] Z. Zhao, D. Tian, X. Duan, and R. Xiao, "Joint Optimization of Battery Swapping Scheduling for Electric Taxis," Sustainability, 2023.  
[8] Y. Feng, X. Lu, X. Huang, and J. Ma, "Real-Time Electric Taxi Guidance for Battery Swapping Stations under Dynamic Demand," Energies, 2025.  
[9] S. Zhang, X. Li, Y. Li, and J. Xue, "A Bi-Objective Battery Dispatching Model of Taxi Battery Swapping Station Network Considering Green Power Consumption," Renewable Energy, 2025.  
[10] S. Yang, J. Yao, T. Kang, and X. Zhu, "Dynamic Operation Model of the Battery Swapping Station for EV in Electricity Market," Energy, 2014.  
[11] G. Hua, Y. Liu, and Y. Xu, "Inventory Policy for Electric Vehicle Battery Swapping Stations in Beijing," Journal of System and Management Sciences, 2023.  
[12] N. Krivulin and A. Garg, "Tropical Modeling of Battery Swapping and Charging Station," Mathematics, 2024.  
[13] D. Renga and M. Meo, "Modeling Battery Swapping Stations for Sustainable Urban Mobility," Sustainable Energy, Grids and Networks, 2025.  
[14] Z. Ma, D. S. Callaway, and I. A. Hiskens, "Decentralized Charging Control of Large Populations of Plug-In Electric Vehicles," IEEE TCST, 2013.  
[15] L. Gan, U. Topcu, and S. H. Low, "Optimal Decentralized Protocol for Electric Vehicle Charging," IEEE TPS, 2013.

### 10.4 与论文对应关系

| 本文档章节 | 论文章节 |
|-----------|---------|
| 2. 系统实体定义 | II. System Design, III.A |
| 3. 核心决策变量与约束 | III.A, III.B, III.C |
| 4. 状态转移模型 | III.A (Eq. 2, 3) |
| 5. 优化目标 | III.C, IV (Eq. 13, 14) |
| 10.2/10.3 相关工作与文献 | VI. Related Work, References |
| 7. 算法流程（工程化补充） | 论文未展开（实现扩展） |

---

## 附录：符号表 (Notation Table)

### 索引与集合
| 符号 | 含义 |
|------|------|
| $m$ | 区域总数 |
| $i, i'$ | 区域索引 |
| $t$ | 时间索引 |
| $l$ | 电量等级索引 |
| $L$ | 最大电量等级 |
| $\Psi$ | 充电任务集合 |

### 状态变量
| 符号 | 含义 |
|------|------|
| $V_{i,l}^t$ | 区域$i$、电量$l$的空车数量 |
| $O_{i,l}^t$ | 区域$i$、电量$l$的载客车数量 |
| $M_i^{t,l}$ | 站点$i$、电量$l$的电池库存 |
| $\hat{M}_i^{t,l}$ | 站点$i$、电量$l$的待充电电池 |

### 决策变量
| 符号 | 含义 |
|------|------|
| $X_{i,i'}^{t,l}$ | 从$i$到$i'$接客的车辆数 |
| $Y_{i,i'}^{t,l}$ | 从$i$到$i'$换电的车辆数 |
| $x_i^t$ | 充电任务$\tau_i$在时刻$t$是否充电 |

### 中间变量
| 符号 | 含义 |
|------|------|
| $B_{i,l}^t$ | 在换电站的车辆数 |
| $S_{i,l}^t$ | 被派遣接客的车辆数 |
| $U_{i,l}^t$ | 正在载客的车辆数 |
| $H_{i,l}^t$ | 换电后的车辆分布 |
| $\mu_{i,l}^t$ | 换电成功数量 |

### 参数
| 符号 | 含义 |
|------|------|
| $p_i$ | 换电机器容量 |
| $q_i$ | 充电器数量 |
| $D_i^t$ | 乘客需求预测 |
| $\omega_{i,i'}$ | 区域间距离 |
| $E_{i,i'}^t$ | 能量消耗 |
| $P$ | 单电池充电功率 |
| $D_j$ | 基础负荷 |

### 概率参数
| 符号 | 含义 |
|------|------|
| $P_{i',i}^t$ | 载客车保持载客的转移概率 |
| $Q_{i',i}^t$ | 载客车变为空车的转移概率 |
| $\tilde{P}_{i',i}^t$ | 空车成功接客的转移概率 |
| $\tilde{Q}_{i',i}^t$ | 空车保持空车的转移概率 |

---

**文档结束**

*本技术规格说明书现在包含了论文中所有关键的技术细节，可以直接用于系统开发和Agent训练。*
