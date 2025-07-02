# models/travel_model.py
"""
负责管理和提供区域间的距离和行驶时间。
"""
import numpy as np

class TravelModel:
    """
    出行模型，封装了距离和时间矩阵。
    """
    def __init__(self, m_areas):
        self.m_areas = m_areas
        # 距离矩阵
        self.distance_matrix = self._generate_synthetic_distances()
        # 行驶时间矩阵 (可以根据交通状况变得更复杂，此处简化)
        self.time_matrix = self.distance_matrix * 2.0 # 简化假设：时间 = 距离 * 2个单位

    def _generate_synthetic_distances(self):
        """
        生成模拟的距离矩阵。
        在实际应用中，应使用真实的地理数据计算。
        """
        print("正在生成模拟的区域间距离矩阵...")
        # 假设所有区域分布在一个网格上，便于计算距离
        grid_size = int(np.ceil(np.sqrt(self.m_areas)))
        positions = {i: (i // grid_size, i % grid_size) for i in range(self.m_areas)}
        
        dist_matrix = np.zeros((self.m_areas, self.m_areas))
        for i in range(self.m_areas):
            for j in range(self.m_areas):
                pos_i = np.array(positions[i])
                pos_j = np.array(positions[j])
                # 使用曼哈顿距离（格子距离）作为区域间距离
                dist_matrix[i, j] = np.sum(np.abs(pos_i - pos_j))
        return dist_matrix

    def get_distance(self, origin_area: int, dest_area: int) -> float:
        """获取两个区域间的距离。"""
        if 0 <= origin_area < self.m_areas and 0 <= dest_area < self.m_areas:
            return self.distance_matrix[origin_area, dest_area]
        return float('inf')

    def get_travel_time(self, origin_area: int, dest_area: int) -> float:
        """获取两个区域间的行驶时间。"""
        if 0 <= origin_area < self.m_areas and 0 <= dest_area < self.m_areas:
            return self.time_matrix[origin_area, dest_area]
        return float('inf')