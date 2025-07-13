# models/travel_model.py
import numpy as np

class TravelModel:
    """
    出行模型，负责管理区域间的距离、时间和能耗。
    """
    def __init__(self, m_areas, max_dist=25.0, speed_kph=30.0, energy_consumption_rate=0.2):
        """
        初始化出行模型，并自动生成距离和时间矩阵。

        Args:
            m_areas (int): 城市区域的数量。
            max_dist (float): 区域间的最大可能距离 (km)。
            speed_kph (float): 车辆平均行驶速度 (km/h)。
            energy_consumption_rate (float): 每公里消耗的能量等级数。
        """
        self.m_areas = m_areas
        self.energy_consumption_rate = energy_consumption_rate
        
        # 【逻辑修正】在构造函数中直接调用方法，确保矩阵被初始化
        self.distance_matrix = self._generate_distance_matrix(max_dist)
        self.time_matrix = self._generate_time_matrix(speed_kph)

    def _generate_distance_matrix(self, max_dist: float) -> np.ndarray:
        """
        生成一个模拟的区域间距离矩阵 (单位: km)。
        这是一个私有方法，在初始化时调用。
        """
        print(f"TravelModel: 正在生成 {self.m_areas}x{self.m_areas} 距离矩阵...")
        # 创建一个随机的对称距离矩阵
        rand_matrix = np.random.rand(self.m_areas, self.m_areas) * max_dist
        dist_matrix = (rand_matrix + rand_matrix.T) / 2
        np.fill_diagonal(dist_matrix, 0)
        return dist_matrix

    def _generate_time_matrix(self, speed_kph: float) -> np.ndarray:
        """
        基于距离矩阵和平均速度计算行驶时间矩阵 (单位: 分钟)。
        这是一个私有方法，在初始化时调用。
        """
        print(f"TravelModel: 正在基于速度 {speed_kph} km/h 生成时间矩阵...")
        if speed_kph <= 0:
            return np.zeros((self.m_areas, self.m_areas))
        # 将速度从 km/h 转换为 km/min
        speed_kmpm = speed_kph / 60.0
        # 计算时间 = 距离 / 速度
        time_matrix = self.distance_matrix / speed_kmpm
        return time_matrix

    def get_distance(self, origin_area: int, dest_area: int) -> float:
        """获取两个区域之间的距离 (km)"""
        if 0 <= origin_area < self.m_areas and 0 <= dest_area < self.m_areas:
            return self.distance_matrix[origin_area, dest_area]
        return float('inf') 

    def get_time(self, origin_area: int, dest_area: int) -> float:
        """获取两个区域之间的行驶时间 (分钟)"""
        if 0 <= origin_area < self.m_areas and 0 <= dest_area < self.m_areas:
            return self.time_matrix[origin_area, dest_area]
        return float('inf')
        
    def get_energy_consumption(self, origin_area: int, dest_area: int) -> int:
        """
        计算从一个区域到另一个区域的能量消耗（以离散的能量等级为单位）。
        """
        distance = self.get_distance(origin_area, dest_area)
        if distance == float('inf'):
            return float('inf')
        energy_units_consumed = distance * self.energy_consumption_rate
        return int(np.round(energy_units_consumed))