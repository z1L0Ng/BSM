# config.py
"""
集中管理模拟所需的所有配置参数。
"""
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class SimulationConfig:
    # 模拟基本参数
    m_areas: int = 10
    L_energy_levels: int = 10
    T_periods: int = 100 # 模型中预设的总时间段数 (用于数组初始化，非单次模拟时长)
    simulation_duration: int = 360 # 单次模拟的总时长 (分钟)
    delta_t: int = 20 # 每个离散时间步的长度 (分钟)
    reoptimization_interval: int = 20 # 调度器重新进行优化的时间间隔 (分钟)
    swap_duration: int = 5 # 单次换电操作耗时 (分钟)
    
    # 车辆配置
    num_taxis: int = 50
    initial_energy_range: tuple = (5, 9) # 车辆初始电量等级范围 (0-9)
    
    # 换电站配置 (这是一个静态配置，实际应用中可以从外部文件加载)
    stations: List[Dict] = field(default_factory=lambda: [
        {'id': 'bss_0', 'location': 0, 'capacity': 20, 'initial_charged': 15},
        {'id': 'bss_2', 'location': 2, 'capacity': 20, 'initial_charged': 15},
        {'id': 'bss_4', 'location': 4, 'capacity': 20, 'initial_charged': 15},
        {'id': 'bss_7', 'location': 7, 'capacity': 20, 'initial_charged': 15},
    ])
    bss_capacity: int = 20 # 用于启发式算法的默认站点容量

    # 优化器参数 (对应论文中的 α, β, γ)
    alpha: float = 1.0   # 服务质量权重 (乘客数)
    beta: float = 0.1    # 空驶距离惩罚权重
    gamma: float = 0.05  # 充电成本惩罚权重