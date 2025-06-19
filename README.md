# 电动出租车电池交换系统模拟框架 (BSM)

基于论文《E-taxi Fleet Formulation》实现的电动出租车电池交换系统模拟框架。该框架实现了论文中提出的时空优化模型，用于研究电池交换站的最优布局、出租车调度策略和充电管理算法。

## 系统架构

系统由以下主要组件构成：

1. **电动出租车(E-taxi)**：实现论文中的状态转移模型，包含能量消耗和移动模式
2. **电池交换站(BSS)**：实现论文中的电池库存管理和充电调度模型
3. **联合优化器**：实现论文中的联合优化问题求解
4. **城市区域网络**：表示城市空间划分和交通网络
5. **数据处理模块**：处理NYC出租车真实数据

## 核心特性

### 基于论文的数学模型
- **时空状态建模**：将城市划分为m个区域，时间离散化为T个时段
- **能量状态离散化**：电池电量分为L个等级
- **状态转移方程**：实现论文公式(1)-(7)的车辆状态演化
- **联合优化**：最大化服务质量同时最小化空驶距离 (公式11-12)

### 真实数据驱动
- 基于NYC出租车数据进行模拟
- 真实的需求分布和行程模式
- 考虑交通拥堵的动态行驶时间

### 优化算法
- Gurobi优化器求解MILP问题
- 启发式算法作为备选方案
- 动态充电任务生成

## 安装指南

### 前提条件

- Python 3.8+
- Conda或虚拟环境管理工具
- Gurobi Optimizer (可选，用于精确优化)

### 环境安装

```bash
# 克隆仓库
git clone https://github.com/username/bsm.git
cd bsm

# 使用Conda创建环境
conda env create -f environment.yml

# 激活环境
conda activate bsm

# 如果需要使用Gurobi
# 1. 从 https://www.gurobi.com 获取学术许可
# 2. 设置许可文件路径
export GRB_LICENSE_FILE=/path/to/gurobi.lic
```

### 数据准备

```bash
# 下载NYC出租车数据 (示例)
# 或使用项目提供的样本数据生成器
python create_sample_data.py
```

## 项目结构

```
bsm/
├── agent/                 # 代理模型
│   └── taxi.py           # 出租车代理 (实现状态转移模型)
├── bss/                  # 电池交换站
│   └── bss.py            # 换电站模型 (实现库存管理)
├── dataprocess/          # 数据处理 (基于论文需求)
│   ├── loaddata.py       # NYC数据加载和清洗
│   └── distance.py       # 距离矩阵计算
├── environment/          # 环境模型
│   ├── blocks.py         # 区块网络 (城市区域划分)
│   └── timestep.py       # 时间步管理
├── optimization/         # 优化模块
│   ├── interface.py      # 优化器接口
│   ├── joint_optimizer.py # 联合优化器 (论文公式11-12)
│   └── charge_scheduler.py # 充电任务调度 (论文第III-D节)
├── scheduler/            # 调度器
│   ├── bss_scheduler.py  # 站点调度
│   └── taxi_scheduler.py # 出租车调度
├── simulation/           # 模拟核心
│   └── simulation.py     # 主模拟流程
├── utils/                # 工具函数
│   └── visualization.py # 可视化工具
├── models/               # 数学模型 (新增)
│   ├── state_transition.py # 状态转移模型 (公式1-7)
│   └── optimization_model.py # 优化模型 (公式11-12)
├── main.py               # 主程序
├── environment.yml       # 环境配置
└── README.md            # 项目说明
```

## 快速开始

### 1. 基本模拟

```bash
# 使用默认配置运行模拟
python main.py

# 使用自定义参数
python main.py --duration 480 --stations 15 --data data/your_data.parquet
```

### 2. 配置文件模式

```python
# 创建配置文件 config.json
{
    "simulation": {
        "duration": 480,
        "time_periods": 96,
        "energy_levels": 10
    },
    "network": {
        "areas": 20,
        "consider_traffic": true
    },
    "taxis": {
        "count": 200,
        "battery_capacity": 100,
        "consumption_rate": 1.0
    },
    "optimization": {
        "solver": "gurobi",
        "objective_weights": {
            "service_quality": 1.0,
            "idle_distance": -0.1
        }
    }
}

# 运行
python main.py --config config.json
```

### 3. 分析结果

```python
from utils.visualization import plot_optimization_results
from utils.analysis import analyze_performance

# 加载结果
results = load_simulation_results('results/simulation_results.json')

# 性能分析
metrics = analyze_performance(results)
print(f"服务质量: {metrics['service_quality']}")
print(f"平均等待时间: {metrics['avg_wait_time']}")

# 可视化
plot_optimization_results(results)
```

## 核心算法

### 1. 联合优化模型 (论文公式11-12)

```python
# 目标函数：最大化服务质量 - β * 空驶距离
max J = J_service + β * J_idle

# 约束条件包括：
# - 状态转移方程 (1)-(7)
# - 距离约束 (10)
# - 电池交换站容量约束
```

### 2. 状态转移模型 (论文公式1-7)

实现论文中的关键状态转移方程：
- 车辆重新定位决策变量 X^{t,l}_{i,i'} 和 Y^{t,l}_{i,i'}
- 车队状态转移方程 (2)-(3)
- 电池交换站动态 (4)-(7)

### 3. 充电任务生成 (论文第III-D节)

根据论文第III-D节实现充电任务生成算法，考虑：
- 电力价格波动
- 需求预测
- 库存管理策略

## 基于NYC数据的功能

### 数据处理流程
1. **数据加载**: 支持parquet和CSV格式的NYC出租车数据
2. **数据清洗**: 过滤异常记录，标准化格式
3. **空间划分**: 将NYC区域划分为网格区块
4. **时间特征**: 提取高峰时段、工作日等特征
5. **需求分布**: 分析真实的出行需求模式

### 真实数据驱动的模拟
- 基于真实行程数据的需求预测
- 考虑NYC交通模式的动态行驶时间
- 真实出租车ID和运营模式

## 性能指标

### 服务质量指标
- 平均乘客等待时间
- 服务请求满足率
- 车辆利用率

### 效率指标
- 平均空驶距离
- 电池交换频率
- 充电器利用率

### 经济指标
- 运营成本
- 收入分析
- 能耗成本

## 配置说明

### 模拟参数
```json
{
    "simulation": {
        "duration": 480,          # 模拟时长(分钟)
        "time_periods": 96,       # 时间段数量 T
        "energy_levels": 10,      # 能量等级数量 L
        "start_hour": 6          # 模拟开始时间
    }
}
```

### 网络配置
```json
{
    "network": {
        "areas": 20,             # 城市区域数量 m
        "consider_traffic": true, # 是否考虑交通拥堵
        "max_travel_time": 60    # 最大行驶时间(分钟)
    }
}
```

### 优化配置
```json
{
    "optimization": {
        "solver": "gurobi",      # 求解器选择
        "time_limit": 300,       # 求解时间限制(秒)
        "mip_gap": 0.05,        # MIP gap tolerance
        "objective_weights": {
            "service_quality": 1.0,
            "idle_distance": -0.1
        }
    }
}
```

## 论文实现对应关系

| 论文章节 | 实现文件 | 说明 |
|---------|---------|------|
| III-A E-taxi Systems | models/state_transition.py | 车辆状态模型 |
| III-B BSS Model | models/state_transition.py | 换电站模型 |
| III-C Fleet Optimization | models/optimization_model.py | 联合优化 |
| III-D Charging Tasks | optimization/charge_scheduler.py | 充电调度 |
| 公式(1)-(7) | models/state_transition.py | 状态转移方程 |
| 公式(11)-(12) | models/optimization_model.py | 目标函数 |

## 扩展功能

### 1. 多目标优化
- 帕累托前沿分析
- 权重敏感性分析

### 2. 不确定性建模
- 需求不确定性
- 设备故障建模

### 3. 实时优化
- 滚动时域优化
- 在线学习算法

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 许可证

MIT License

## 引用

如果您在研究中使用此框架，请引用：

```bibtex
@article{etaxi_bss_2025,
    title={E-taxi Fleet Formulation with Battery Swapping Stations},
    author={Your Name},
    journal={Simulation Framework},
    year={2025}
}
```

## 联系方式

- 项目主页: https://github.com/username/bsm
- 问题反馈: https://github.com/username/bsm/issues
- 邮箱: your.email@domain.com 区块网络 (城市区域划分)
│   └── timestep.py       # 时间步管理
├── optimization/         # 优化模块
│   ├── interface.py      # 优化器接口
│   ├── joint_optimizer.py # 联合优化器 (新增)
│   └── charge_scheduler.py # 充电任务调度 (新增)
├── scheduler/            # 调度器
│   ├── bss_scheduler.py  # 站点调度
│   └── taxi_scheduler.py # 出租车调度
├── simulation/           # 模拟核心
│   └── simulation.py     # 主模拟流程
├── utils/                # 工具函数
│   └── visualization.py # 可视化工具
├── models/               # 数学模型 (新增)
│   ├── state_transition.py # 状态转移模型
│   └── optimization_model.py # 优化模型
├── main.py               # 主程序
├── environment.yml       # 环境配置
└── README.md            # 项目说明
```

## 快速开始

### 1. 基本模拟

```bash
# 使用默认配置运行模拟
python main.py

# 使用自定义参数
python main.py --duration 480 --stations 15 --data data/your_data.parquet
```

### 2. 配置文件模式

```python
# 创建配置文件 config.json
{
    "simulation": {
        "duration": 480,
        "time_periods": 96,
        "energy_levels": 10
    },
    "network": {
        "areas": 20,
        "consider_traffic": true
    },
    "taxis": {
        "count": 200,
        "battery_capacity": 100,
        "consumption_rate": 1.0
    },
    "optimization": {
        "solver": "gurobi",
        "objective_weights": {
            "service_quality": 1.0,
            "idle_distance": -0.1
        }
    }
}

# 运行
python main.py --config config.json
```

### 3. 分析结果

```python
from utils.visualization import plot_optimization_results
from utils.analysis import analyze_performance

# 加载结果
results = load_simulation_results('results/simulation_results.json')

# 性能分析
metrics = analyze_performance(results)
print(f"服务质量: {metrics['service_quality']}")
print(f"平均等待时间: {metrics['avg_wait_time']}")

# 可视化
plot_optimization_results(results)
```

## 核心算法

### 1. 联合优化模型 (论文公式11-12)

```python
# 目标函数：最大化服务质量 - β * 空驶距离
max J = J_service + β * J_idle

# 约束条件包括：
# - 状态转移方程 (1)-(7)
# - 距离约束 (10)
# - 电池交换站容量约束
```

### 2. 状态转移模型

实现论文中的关键状态转移方程：
- 车辆重新定位决策变量 X^{t,l}_{i,i'} 和 Y^{t,l}_{i,i'}
- 车队状态转移方程 (2)-(3)
- 电池交换站动态 (4)-(7)

### 3. 充电任务生成

根据论文第III-D节实现充电任务生成算法，考虑：
- 电力价格波动
- 需求预测
- 库存管理策略

## 性能指标

### 服务质量指标
- 平均乘客等待时间
- 服务请求满足率
- 车辆利用率

### 效率指标
- 平均空驶距离
- 电池交换频率
- 充电器利用率

### 经济指标
- 运营成本
- 收入分析
- 能耗成本

## 配置说明

### 模拟参数
```json
{
    "simulation": {
        "duration": 480,          # 模拟时长(分钟)
        "time_periods": 96,       # 时间段数量 T
        "energy_levels": 10,      # 能量等级数量 L
        "start_hour": 6          # 模拟开始时间
    }
}
```

### 网络配置
```json
{
    "network": {
        "areas": 20,             # 城市区域数量 m
        "consider_traffic": true, # 是否考虑交通拥堵
        "max_travel_time": 60    # 最大行驶时间(分钟)
    }
}
```

### 优化配置
```json
{
    "optimization": {
        "solver": "gurobi",      # 求解器选择
        "time_limit": 300,       # 求解时间限制(秒)
        "mip_gap": 0.05,        # MIP gap tolerance
        "objective_weights": {
            "service_quality": 1.0,
            "idle_distance": -0.1
        }
    }
}
```

## 扩展功能

### 1. 多目标优化
- 帕累托前沿分析
- 权重敏感性分析

### 2. 不确定性建模
- 需求不确定性
- 设备故障建模

### 3. 实时优化
- 滚动时域优化
- 在线学习算法

## 贡献指南

1. Fork项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 许可证

MIT License

## 引用

如果您在研究中使用此框架，请引用：

```bibtex
@article{etaxi_bss_2025,
    title={E-taxi Fleet Formulation with Battery Swapping Stations},
    author={Your Name},
    journal={Simulation Framework},
    year={2025}
}
```

## 联系方式

- 项目主页: https://github.com/username/bsm
- 问题反馈: https://github.com/username/bsm/issues
- 邮箱: your.email@domain.com