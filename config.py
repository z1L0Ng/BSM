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

@dataclass
class CityWideSimulationConfig:
    """
    城市级模拟配置 - 扩展到整个城市的数据规模
    """
    # 城市级模拟基本参数
    m_areas: int = 100  # 扩展到100个区域覆盖整个城市
    L_energy_levels: int = 10
    T_periods: int = 72  # 24小时，每20分钟一个时间段 (24 * 3)
    simulation_duration: int = 1440  # 全天24小时模拟 (分钟)
    delta_t: int = 20  # 每个离散时间步的长度 (分钟)
    reoptimization_interval: int = 20  # 调度器重新进行优化的时间间隔 (分钟)
    swap_duration: int = 5  # 单次换电操作耗时 (分钟)
    
    # 城市级车辆配置
    num_taxis: int = 1000  # 扩展到1000辆出租车
    initial_energy_range: tuple = (5, 9)  # 车辆初始电量等级范围 (0-9)
    
    # 城市级换电站配置 - 自动生成分布在各个区域的换电站
    stations: List[Dict] = field(default_factory=lambda: [
        {'id': f'bss_{i}', 'location': i, 'capacity': 50, 'initial_charged': 35}
        for i in range(0, 100, 5)  # 每5个区域设置一个换电站，共20个站点
    ])
    bss_capacity: int = 50  # 城市级换电站容量

    # 城市级优化器参数 (对应论文中的 α, β, γ)
    alpha: float = 1.0   # 服务质量权重 (乘客数)
    beta: float = 0.05   # 空驶距离惩罚权重 (较小，适应城市级距离)
    gamma: float = 0.02  # 充电成本惩罚权重 (较小，适应规模经济)
    
    # 城市级特定参数
    use_real_data: bool = True  # 使用真实NYC数据
    data_file: str = "data/citywide_sample_data.parquet"
    sample_size: int = 50000  # 数据采样大小
    
    # 性能优化参数
    parallel_processing: bool = True  # 启用并行处理
    solver_time_limit: int = 600  # 优化器求解时间限制 (秒)
    heuristic_mode: bool = True  # 大规模使用启发式算法
    
    # 结果输出配置
    detailed_logging: bool = True  # 详细日志记录
    visualization_enabled: bool = True  # 启用可视化
    performance_tracking: bool = True  # 性能追踪