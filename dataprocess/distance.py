"""距离矩阵和行驶时间计算模块。"""
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from typing import Dict, Tuple
import pickle
import os

def compute_distance_matrix(block_positions: Dict[int, Tuple[float, float]], 
                          distance_type: str = 'euclidean') -> Dict[int, Dict[int, float]]:
    """
    计算区块间的距离矩阵。
    
    参数:
        block_positions (dict): 区块位置字典 {block_id: (x, y)}
        distance_type (str): 距离类型 ('euclidean', 'manhattan')
    
    返回:
        dict: 嵌套字典格式的距离矩阵 {from_block: {to_block: distance}}
    """
    print(f"计算 {len(block_positions)} 个区块的距离矩阵...")
    
    block_ids = list(block_positions.keys())
    coordinates = np.array([block_positions[bid] for bid in block_ids])
    
    # 计算距离矩阵
    if distance_type == 'euclidean':
        distances = cdist(coordinates, coordinates, metric='euclidean')
    elif distance_type == 'manhattan':
        distances = cdist(coordinates, coordinates, metric='manhattan')
    else:
        raise ValueError(f"不支持的距离类型: {distance_type}")
    
    # 转换为嵌套字典格式
    distance_matrix = {}
    for i, from_block in enumerate(block_ids):
        distance_matrix[from_block] = {}
        for j, to_block in enumerate(block_ids):
            distance_matrix[from_block][to_block] = distances[i, j]
    
    return distance_matrix

def compute_travel_time_matrix(distance_matrix: Dict[int, Dict[int, float]], 
                             avg_speed_km_per_min: float = 0.5,
                             traffic_factor: float = 1.0) -> Dict[int, Dict[int, float]]:
    """
    基于距离矩阵计算行驶时间矩阵。
    
    参数:
        distance_matrix (dict): 距离矩阵
        avg_speed_km_per_min (float): 平均速度 (公里/分钟)
        traffic_factor (float): 交通拥堵因子 (>1表示拥堵)
    
    返回:
        dict: 行驶时间矩阵 {from_block: {to_block: time_minutes}}
    """
    print(f"计算行驶时间矩阵 (速度: {avg_speed_km_per_min} km/min, 交通因子: {traffic_factor})")
    
    time_matrix = {}
    for from_block in distance_matrix:
        time_matrix[from_block] = {}
        for to_block in distance_matrix[from_block]:
            distance = distance_matrix[from_block][to_block]
            # 时间 = 距离 / 速度 * 交通因子
            travel_time = (distance / avg_speed_km_per_min) * traffic_factor
            time_matrix[from_block][to_block] = travel_time
    
    return time_matrix

def create_traffic_factors() -> Dict[int, float]:
    """
    创建24小时的交通拥堵因子。
    
    返回:
        dict: 小时到交通因子的映射 {hour: factor}
    """
    # 基于NYC实际交通模式的拥堵因子
    traffic_factors = {
        0: 0.8,   # 午夜
        1: 0.7,   # 凌晨1点
        2: 0.6,   # 凌晨2点
        3: 0.6,   # 凌晨3点
        4: 0.7,   # 凌晨4点
        5: 0.9,   # 凌晨5点
        6: 1.2,   # 早上6点
        7: 1.8,   # 早高峰开始
        8: 2.2,   # 早高峰高峰
        9: 1.5,   # 早高峰结束
        10: 1.1,  # 上午
        11: 1.2,  # 上午晚些
        12: 1.4,  # 中午
        13: 1.3,  # 下午1点
        14: 1.2,  # 下午2点
        15: 1.3,  # 下午3点
        16: 1.6,  # 下午4点
        17: 2.0,  # 晚高峰开始
        18: 2.3,  # 晚高峰高峰
        19: 1.8,  # 晚高峰结束
        20: 1.4,  # 晚上8点
        21: 1.2,  # 晚上9点
        22: 1.0,  # 晚上10点
        23: 0.9   # 晚上11点
    }
    
    return traffic_factors

def compute_energy_consumption_matrix(distance_matrix: Dict[int, Dict[int, float]], 
                                    consumption_rate: float = 1.0) -> Dict[int, Dict[int, float]]:
    """
    计算区块间移动的能量消耗矩阵。
    
    参数:
        distance_matrix (dict): 距离矩阵
        consumption_rate (float): 能量消耗率 (kWh/km)
    
    返回:
        dict: 能量消耗矩阵 {from_block: {to_block: energy_kwh}}
    """
    print(f"计算能量消耗矩阵 (消耗率: {consumption_rate} kWh/km)")
    
    energy_matrix = {}
    for from_block in distance_matrix:
        energy_matrix[from_block] = {}
        for to_block in distance_matrix[from_block]:
            distance = distance_matrix[from_block][to_block]
            energy = distance * consumption_rate
            energy_matrix[from_block][to_block] = energy
    
    return energy_matrix

def compute_reachability_matrix(travel_time_matrix: Dict[int, Dict[int, float]], 
                               max_time: float) -> Dict[int, Dict[int, bool]]:
    """
    计算在给定时间内的可达性矩阵 (对应论文中的ν^t_{i,i'})。
    
    参数:
        travel_time_matrix (dict): 行驶时间矩阵
        max_time (float): 最大允许行驶时间 (分钟)
    
    返回:
        dict: 可达性矩阵 {from_block: {to_block: is_reachable}}
    """
    print(f"计算可达性矩阵 (最大时间: {max_time} 分钟)")
    
    reachability_matrix = {}
    for from_block in travel_time_matrix:
        reachability_matrix[from_block] = {}
        for to_block in travel_time_matrix[from_block]:
            travel_time = travel_time_matrix[from_block][to_block]
            is_reachable = travel_time <= max_time
            reachability_matrix[from_block][to_block] = is_reachable
    
    return reachability_matrix

def save_matrices(matrices: Dict, filepath: str):
    """
    保存距离和时间矩阵到文件。
    
    参数:
        matrices (dict): 包含各种矩阵的字典
        filepath (str): 保存路径
    """
    print(f"保存矩阵到: {filepath}")
    
    # 确保目录存在
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(matrices, f)

def load_matrices(filepath: str) -> Dict:
    """
    从文件加载距离和时间矩阵。
    
    参数:
        filepath (str): 文件路径
    
    返回:
        dict: 包含各种矩阵的字典
    """
    print(f"从文件加载矩阵: {filepath}")
    
    with open(filepath, 'rb') as f:
        matrices = pickle.load(f)
    
    return matrices

def create_all_matrices(block_positions: Dict[int, Tuple[float, float]], 
                       config: Dict = None) -> Dict:
    """
    创建所有必需的矩阵 (距离、时间、能耗、可达性)。
    
    参数:
        block_positions (dict): 区块位置
        config (dict): 配置参数
    
    返回:
        dict: 包含所有矩阵的字典
    """
    if config is None:
        config = {}
    
    print("创建所有矩阵...")
    
    # 默认参数
    avg_speed = config.get('avg_speed_km_per_min', 0.5)
    consumption_rate = config.get('consumption_rate_kwh_per_km', 1.0)
    max_travel_time = config.get('max_travel_time_minutes', 60)
    
    # 1. 距离矩阵
    distance_matrix = compute_distance_matrix(block_positions)
    
    # 2. 交通因子
    traffic_factors = create_traffic_factors()
    
    # 3. 不同时间的行驶时间矩阵
    travel_time_matrices = {}
    for hour in range(24):
        factor = traffic_factors.get(hour, 1.0)
        travel_time_matrices[hour] = compute_travel_time_matrix(
            distance_matrix, avg_speed, factor
        )
    
    # 4. 能量消耗矩阵
    energy_matrix = compute_energy_consumption_matrix(distance_matrix, consumption_rate)
    
    # 5. 可达性矩阵 (使用平均行驶时间)
    avg_travel_time_matrix = compute_travel_time_matrix(distance_matrix, avg_speed, 1.0)
    reachability_matrix = compute_reachability_matrix(avg_travel_time_matrix, max_travel_time)
    
    matrices = {
        'distance_matrix': distance_matrix,
        'travel_time_matrices': travel_time_matrices,
        'energy_matrix': energy_matrix,
        'reachability_matrix': reachability_matrix,
        'traffic_factors': traffic_factors,
        'config': {
            'avg_speed_km_per_min': avg_speed,
            'consumption_rate_kwh_per_km': consumption_rate,
            'max_travel_time_minutes': max_travel_time
        }
    }
    
    return matrices

def get_shortest_path_distance(from_block: int, to_block: int, 
                              distance_matrix: Dict[int, Dict[int, float]]) -> float:
    """
    获取两个区块间的最短距离。
    
    参数:
        from_block (int): 起始区块
        to_block (int): 目标区块
        distance_matrix (dict): 距离矩阵
    
    返回:
        float: 最短距离
    """
    return distance_matrix.get(from_block, {}).get(to_block, float('inf'))

def get_travel_time(from_block: int, to_block: int, 
                   travel_time_matrices: Dict[int, Dict[int, Dict[int, float]]], 
                   hour: int) -> float:
    """
    获取指定时间的行驶时间。
    
    参数:
        from_block (int): 起始区块
        to_block (int): 目标区块
        travel_time_matrices (dict): 行驶时间矩阵字典
        hour (int): 小时 (0-23)
    
    返回:
        float: 行驶时间 (分钟)
    """
    hour_matrix = travel_time_matrices.get(hour % 24, travel_time_matrices.get(0, {}))
    return hour_matrix.get(from_block, {}).get(to_block, float('inf'))

def is_reachable(from_block: int, to_block: int, 
                reachability_matrix: Dict[int, Dict[int, bool]]) -> bool:
    """
    检查两个区块是否可达。
    
    参数:
        from_block (int): 起始区块
        to_block (int): 目标区块
        reachability_matrix (dict): 可达性矩阵
    
    返回:
        bool: 是否可达
    """
    return reachability_matrix.get(from_block, {}).get(to_block, False)

def analyze_network_connectivity(matrices: Dict) -> Dict:
    """
    分析网络连通性统计信息。
    
    参数:
        matrices (dict): 包含各种矩阵的字典
    
    返回:
        dict: 连通性统计信息
    """
    distance_matrix = matrices['distance_matrix']
    reachability_matrix = matrices['reachability_matrix']
    
    blocks = list(distance_matrix.keys())
    n_blocks = len(blocks)
    
    # 计算平均距离
    total_distance = 0
    count = 0
    for from_block in blocks:
        for to_block in blocks:
            if from_block != to_block:
                dist = distance_matrix[from_block][to_block]
                if dist < float('inf'):
                    total_distance += dist
                    count += 1
    
    avg_distance = total_distance / count if count > 0 else 0
    
    # 计算可达性统计
    reachable_pairs = 0
    for from_block in blocks:
        for to_block in blocks:
            if from_block != to_block and reachability_matrix[from_block][to_block]:
                reachable_pairs += 1
    
    connectivity_ratio = reachable_pairs / (n_blocks * (n_blocks - 1)) if n_blocks > 1 else 0
    
    # 计算每个区块的可达区块数
    reachable_counts = []
    for from_block in blocks:
        count = sum(1 for to_block in blocks 
                   if from_block != to_block and reachability_matrix[from_block][to_block])
        reachable_counts.append(count)
    
    stats = {
        'total_blocks': n_blocks,
        'avg_distance': avg_distance,
        'connectivity_ratio': connectivity_ratio,
        'avg_reachable_blocks': np.mean(reachable_counts),
        'min_reachable_blocks': min(reachable_counts) if reachable_counts else 0,
        'max_reachable_blocks': max(reachable_counts) if reachable_counts else 0
    }
    
    return stats

if __name__ == "__main__":
    # 测试矩阵计算
    print("测试距离矩阵计算...")
    
    # 创建示例区块位置
    test_positions = {
        0: (0, 0),
        1: (10, 0),
        2: (0, 10),
        3: (10, 10),
        4: (5, 5)
    }
    
    # 创建所有矩阵
    matrices = create_all_matrices(test_positions)
    
    # 分析网络连通性
    stats = analyze_network_connectivity(matrices)
    
    print("\n网络连通性统计:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # 测试具体函数
    print(f"\n区块0到区块3的距离: {get_shortest_path_distance(0, 3, matrices['distance_matrix']):.2f}")
    print(f"早高峰(8点)行驶时间: {get_travel_time(0, 3, matrices['travel_time_matrices'], 8):.2f} 分钟")
    print(f"是否可达: {is_reachable(0, 3, matrices['reachability_matrix'])}")
    
    print("\n矩阵计算测试完成!")