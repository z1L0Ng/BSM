# 电动出租车电池交换系统模拟框架 (BSM)

本项目实现了一个基于电池交换的电动出租车运营系统的离散事件模拟框架。该框架可以用于研究电池交换站的最优布局、出租车调度策略和充电管理算法，以提高电动出租车车队的服务质量和运营效率。

## 系统架构

系统由以下主要组件构成：

1. **电动出租车(E-taxi)**：在城市网络中行驶并提供乘客服务的自主代理。
2. **电池交换站(BSS)**：提供电池更换服务的设施，包含多个充电器。
3. **E-taxi调度器**：管理出租车车队，为出租车分配行程和换电任务。
4. **BSS充电调度器**：管理站内充电资源，优化电池充电策略。
5. **城市区块网络**：表示城市交通拓扑和行程需求。

## 主要功能

- 基于真实NYC出租车数据进行模拟
- 电池交换站的优化布局
- 动态出租车调度和行程分配
- 基于时间和电价的智能充电策略
- 性能指标分析和可视化

## 安装指南

### 前提条件

- Python 3.8+
- Conda或虚拟环境管理工具

### 环境安装

```bash
# 克隆仓库
git clone https://github.com/username/bsm.git
cd bsm

# 使用Conda创建环境
conda env create -f environment.yml

# 激活环境
conda activate bsm

bsm/
├── agent/             # 代理模型
│   └── taxi.py        # 出租车代理
├── bss/               # 电池交换站
│   └── bss.py         # 换电站模型
├── data/              # 数据处理
│   ├── loaddata.py    # 数据加载
│   └── distance.py    # 距离矩阵计算
├── environment/       # 环境模型
│   ├── blocks.py      # 区块网络
│   └── timestep.py    # 时间步管理
├── optimization/      # 优化模块
│   ├── interface.py   # 优化器接口
│   └── optimal_placement.py # 站点布局优化
├── scheduler/         # 调度器
│   ├── bss_scheduler.py # 站点调度
│   └── taxi_scheduler.py # 出租车调度
├── simulation/        # 模拟核心
│   └── simulation.py  # 主模拟流程
├── utils/             # 工具函数
│   └── visualization.py # 可视化工具
├── main.py            # 主程序
├── environment.yml    # 环境配置
└── README.md          # 项目说明