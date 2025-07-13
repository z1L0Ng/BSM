from dataclasses import dataclass, field
from typing import List, Dict, Tuple

@dataclass
class SimulationConfig:
    # Simulation duration in minutes
    simulation_duration: int = 360  # e.g., 6 hours
    # Time step duration in minutes
    delta_t: int = 20
    # Reoptimization interval in minutes
    reoptimization_interval: int = 20
    # Number of taxis
    num_taxis: int = 50
    # Number of areas in the city
    m_areas: int = 10
    # Number of discrete energy levels
    L_energy_levels: int = 20
    
    # 【逻辑修正】添加缺失的车辆平均速度属性
    avg_speed: float = 30.0 # Vehicle average speed in km/h

    # Initial energy range for taxis (as a tuple of min, max levels)
    initial_energy_range: Tuple[int, int] = (10, 19)
    # Battery swap duration in minutes
    swap_duration: int = 5
    # Battery capacity of stations
    bss_capacity: int = 20

    # Station configurations
    stations: List[Dict] = field(default_factory=lambda: [
        {'id': 'bss_0', 'location': 0, 'capacity': 20, 'initial_charged': 15},
        {'id': 'bss_2', 'location': 2, 'capacity': 20, 'initial_charged': 15},
        {'id': 'bss_4', 'location': 4, 'capacity': 20, 'initial_charged': 15},
        {'id': 'bss_7', 'location': 7, 'capacity': 20, 'initial_charged': 15},
    ])
    # Total number of time periods
    T_periods: int = field(init=False)

    # Objective function weights
    alpha: float = 1.0  # Weight for service quality
    beta: float = 0.1   # Weight for idle distance cost
    gamma: float = 0.1  # Weight for charging cost

    def __post_init__(self):
        self.T_periods = self.simulation_duration // self.delta_t


@dataclass
class CityWideSimulationConfig(SimulationConfig):
    """
    城市级模拟的特定配置
    """
    # 使用真实数据
    use_real_data: bool = False # 改为 True 以使用真实数据
    # 真实数据文件路径 (例如，处理过的 .csv 或 .parquet 文件)
    data_file: str = 'data/processed/nyc_taxi_sample.parquet'
    # 用于数据处理的样本量
    sample_size: int = 50000

    # 日志记录和输出
    detailed_logging: bool = True
    
    # 能耗率 (每公里消耗的能量等级数)
    energy_consumption_rate: float = 0.2

    def __post_init__(self):
        super().__post_init__()
        if self.use_real_data:
            print(f"城市级模拟将尝试使用真实数据: {self.data_file}")