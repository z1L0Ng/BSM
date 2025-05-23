"""城市区块间距离矩阵构建。"""
import numpy as np

def compute_distance_matrix(block_positions):
    """
    计算所有区块之间的曼哈顿距离矩阵。
    
    参数:
        block_positions (dict): 区块ID到(x, y)坐标的映射
    返回:
        dict: 嵌套字典表示的距离 {block_i: {block_j: distance, ...}, ...}
    """
    blocks = list(block_positions.keys())
    dist_matrix = {i: {} for i in blocks}
    
    for i in blocks:
        x1, y1 = block_positions[i]
        for j in blocks:
            x2, y2 = block_positions[j]
            # 使用曼哈顿距离作为城市行驶距离的近似值
            dist = abs(x1 - x2) + abs(y1 - y2)
            dist_matrix[i][j] = dist
    
    return dist_matrix


def compute_travel_time_matrix(distance_matrix, avg_speed_km_per_min=0.5, traffic_factor=None):
    """
    基于距离矩阵计算行驶时间矩阵。
    
    参数:
        distance_matrix (dict): 区块间距离矩阵
        avg_speed_km_per_min (float): 平均速度(千米/分钟)
        traffic_factor (dict 或 float, optional): 不同时段的交通拥堵因子或固定因子
    
    返回:
        dict: 行驶时间矩阵 {block_i: {block_j: time_minutes, ...}, ...}
    """
    travel_time_matrix = {}
    
    for origin in distance_matrix:
        travel_time_matrix[origin] = {}
        for dest in distance_matrix[origin]:
            # 基础行驶时间 = 距离 / 速度
            base_time = distance_matrix[origin][dest] / avg_speed_km_per_min
            
            # 处理交通因子，可为字典或单个浮点数
            if isinstance(traffic_factor, dict):
                # 示例使用固定 hour=8，实际调用时可传入动态 hour
                hour = 8
                factor = traffic_factor.get(hour, 1.0)
            elif isinstance(traffic_factor, (int, float)):
                factor = traffic_factor
            else:
                factor = 1.0
            
            travel_time = base_time * factor
            travel_time_matrix[origin][dest] = max(1.0, travel_time)  # 最小行驶时间为1分钟
    
    return travel_time_matrix


def create_traffic_factors():
    """
    创建一天中不同时段的交通拥堵因子。
    较大的值表示交通更拥堵，行驶时间更长。
    
    返回:
        dict: 小时到交通因子的映射
    """
    traffic_factors = {}
    
    # 早高峰 (7-9点)
    for hour in range(7, 10):
        traffic_factors[hour] = 1.5
    
    # 晚高峰 (17-19点)
    for hour in range(17, 20):
        traffic_factors[hour] = 1.7
    
    # 中午时段 (11-16点)
    for hour in range(11, 17):
        traffic_factors[hour] = 1.2
    
    # 夜间/清晨 (其他时间)
    for hour in range(0, 7):
        traffic_factors[hour] = 0.8
    for hour in range(20, 24):
        traffic_factors[hour] = 0.9
    
    return traffic_factors
