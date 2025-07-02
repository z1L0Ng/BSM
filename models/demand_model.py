# models/demand_model.py
"""
负责管理和提供乘客需求数据。
"""
import numpy as np

class ODMatrixDemandModel:
    """
    基于起点-终点（Origin-Destination）矩阵的需求模型。
    """
    def __init__(self, m_areas, t_periods):
        self.m_areas = m_areas
        self.t_periods = t_periods
        # demand_matrix[t, i, j] 代表 t 时刻从区域 i 到区域 j 的乘客需求数
        self.demand_matrix = self._generate_synthetic_demand()
        print(f"已生成覆盖 {t_periods} 个时间段的模拟需求数据。")

    def _generate_synthetic_demand(self):
        """
        生成模拟的需求数据。
        在实际应用中，这里应该从真实数据文件加载和处理。
        """
        demand = np.zeros((self.t_periods, self.m_areas, self.m_areas))
        
        # 创建一个简单的、随时间波动的需求模式
        for t in range(self.t_periods):
            # 假设每20分钟一个时间片，计算当前处于一天中的哪个小时
            hour_of_day = (t * 20 // 60) % 24
            
            # 模拟早晚高峰
            if 7 <= hour_of_day <= 9 or 17 <= hour_of_day <= 19:
                base_demand_rate = 2.0  # 高峰期每对OD平均需求
            else:
                base_demand_rate = 0.5  # 平峰期需求
            
            # 为每个OD对生成随机整数需求
            for i in range(self.m_areas):
                for j in range(self.m_areas):
                    if i != j:
                        # 使用泊松分布生成离散的乘客数量
                        demand[t, i, j] = np.random.poisson(base_demand_rate)
        return demand

    def get_demand(self, t: int):
        """
        获取时间步 t 的需求矩阵。
        
        Args:
            t (int): 当前模拟的时间步。

        Returns:
            np.ndarray: 一个 (m, m) 的需求矩阵 D_ij，代表从i到j的需求数。
        """
        if 0 <= t < self.t_periods:
            return self.demand_matrix[t].copy() # 返回一个拷贝以防原始数据被意外修改
        else:
            # 如果请求的时间超出了预设范围，返回一个全零矩阵
            return np.zeros((self.m_areas, self.m_areas))