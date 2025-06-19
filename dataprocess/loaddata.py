"""NYC出租车数据加载和预处理模块。"""
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import warnings

def load_trip_data(filepath, sample_size=None, random_state=42):
    """
    加载NYC出租车行程数据。
    
    参数:
        filepath (str): 数据文件路径
        sample_size (int, optional): 采样大小
        random_state (int): 随机种子
    
    返回:
        DataFrame: 原始行程数据
    """
    try:
        print(f"正在加载数据: {filepath}")
        
        # 根据文件扩展名选择加载方法
        if filepath.endswith('.parquet'):
            df = pd.read_parquet(filepath)
        elif filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            raise ValueError(f"不支持的文件格式: {filepath}")
        
        print(f"原始数据大小: {len(df)} 行")
        
        # 如果指定了采样大小，进行随机采样
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=random_state)
            print(f"采样后数据大小: {len(df)} 行")
        
        return df
        
    except Exception as e:
        print(f"加载数据时出错: {str(e)}")
        raise

def clean_trip_data(df):
    """
    清洗和预处理出租车行程数据。
    
    参数:
        df (DataFrame): 原始数据
    
    返回:
        DataFrame: 清洗后的数据
    """
    print("开始数据清洗...")
    original_size = len(df)
    
    # 标准化列名（处理不同数据源的列名差异）
    column_mapping = {
        'tpep_pickup_datetime': 'pickup_datetime',
        'tpep_dropoff_datetime': 'dropoff_datetime',
        'lpep_pickup_datetime': 'pickup_datetime',
        'lpep_dropoff_datetime': 'dropoff_datetime',
        'PULocationID': 'pickup_location_id',
        'DOLocationID': 'dropoff_location_id',
        'passenger_count': 'passenger_count',
        'trip_distance': 'trip_distance',
        'total_amount': 'fare_amount',
        'fare_amount': 'fare_amount'
    }
    
    # 重命名列
    df = df.rename(columns=column_mapping)
    
    # 确保必需的列存在
    required_columns = ['pickup_datetime']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"警告: 缺少必需的列: {missing_columns}")
        # 尝试创建时间列
        if 'pickup_datetime' not in df.columns:
            # 如果没有时间列，创建模拟的时间数据
            print("创建模拟时间数据...")
            base_time = pd.Timestamp('2025-01-01 06:00:00')
            df['pickup_datetime'] = [
                base_time + pd.Timedelta(minutes=np.random.randint(0, 1440))
                for _ in range(len(df))
            ]
    
    # 转换时间列
    try:
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
        if 'dropoff_datetime' in df.columns:
            df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])
    except Exception as e:
        print(f"时间转换警告: {e}")
    
    # 数据清洗步骤
    
    # 1. 删除时间异常的行程
    if 'pickup_datetime' in df.columns:
        df = df.dropna(subset=['pickup_datetime'])
        # 过滤合理的时间范围
        min_time = pd.Timestamp('2020-01-01')
        max_time = pd.Timestamp('2025-12-31')
        df = df[(df['pickup_datetime'] >= min_time) & (df['pickup_datetime'] <= max_time)]
    
    # 2. 处理地理位置数据
    if 'pickup_latitude' in df.columns and 'pickup_longitude' in df.columns:
        # 删除经纬度为空的记录
        df = df.dropna(subset=['pickup_latitude', 'pickup_longitude'])
        
        # 过滤纽约市范围内的数据
        nyc_bounds = {
            'lat_min': 40.4, 'lat_max': 41.0,
            'lon_min': -74.3, 'lon_max': -73.7
        }
        
        df = df[
            (df['pickup_latitude'].between(nyc_bounds['lat_min'], nyc_bounds['lat_max'])) &
            (df['pickup_longitude'].between(nyc_bounds['lon_min'], nyc_bounds['lon_max']))
        ]
        
        if 'dropoff_latitude' in df.columns and 'dropoff_longitude' in df.columns:
            df = df.dropna(subset=['dropoff_latitude', 'dropoff_longitude'])
            df = df[
                (df['dropoff_latitude'].between(nyc_bounds['lat_min'], nyc_bounds['lat_max'])) &
                (df['dropoff_longitude'].between(nyc_bounds['lon_min'], nyc_bounds['lon_max']))
            ]
    
    # 3. 处理行程距离和费用
    if 'trip_distance' in df.columns:
        df = df[df['trip_distance'] > 0]  # 删除距离为0的行程
        df = df[df['trip_distance'] <= 100]  # 删除异常长距离行程
    
    if 'fare_amount' in df.columns:
        df = df[df['fare_amount'] > 0]  # 删除费用为0的行程
        df = df[df['fare_amount'] <= 1000]  # 删除异常高费用行程
    
    # 4. 处理乘客数量
    if 'passenger_count' in df.columns:
        df = df[(df['passenger_count'] >= 1) & (df['passenger_count'] <= 6)]
    
    # 5. 创建区域块ID
    df = create_spatial_blocks(df)
    
    # 6. 提取时间特征
    df = extract_temporal_features(df)
    
    # 7. 处理出租车ID
    df = process_taxi_ids(df)
    
    print(f"数据清洗完成: {original_size} -> {len(df)} 行 (保留 {len(df)/original_size*100:.1f}%)")
    
    return df

def create_spatial_blocks(df, n_blocks=50):
    """
    将地理坐标划分为空间区块。
    
    参数:
        df (DataFrame): 包含经纬度的数据
        n_blocks (int): 目标区块数量
    
    返回:
        DataFrame: 添加了block_id列的数据
    """
    if 'pickup_latitude' not in df.columns or 'pickup_longitude' not in df.columns:
        # 如果没有经纬度数据，创建随机区块ID
        print("没有经纬度数据，创建随机区块ID")
        df['pickup_block'] = np.random.randint(0, n_blocks, len(df))
        df['block_id'] = df['pickup_block']
        return df
    
    print(f"创建 {n_blocks} 个空间区块...")
    
    # 使用网格方法划分区块
    grid_size = int(np.ceil(np.sqrt(n_blocks)))
    
    # 计算纬度和经度的分位数
    lat_bins = pd.qcut(df['pickup_latitude'], q=grid_size, labels=False, duplicates='drop')
    lon_bins = pd.qcut(df['pickup_longitude'], q=grid_size, labels=False, duplicates='drop')
    
    # 创建区块ID
    df['pickup_block'] = lat_bins * grid_size + lon_bins
    df['block_id'] = df['pickup_block']
    
    # 处理下车区块
    if 'dropoff_latitude' in df.columns and 'dropoff_longitude' in df.columns:
        # 使用相同的分位数边界处理下车点
        lat_bins_drop = pd.cut(df['dropoff_latitude'], bins=pd.qcut(df['pickup_latitude'], q=grid_size, retbins=True)[1], labels=False)
        lon_bins_drop = pd.cut(df['dropoff_longitude'], bins=pd.qcut(df['pickup_longitude'], q=grid_size, retbins=True)[1], labels=False)
        df['dropoff_block'] = lat_bins_drop * grid_size + lon_bins_drop
    
    # 处理NaN值
    df['block_id'] = df['block_id'].fillna(0)
    df['pickup_block'] = df['pickup_block'].fillna(0)
    
    print(f"创建了 {df['block_id'].nunique()} 个唯一区块")
    
    return df

def extract_temporal_features(df):
    """
    从时间戳中提取时间特征。
    
    参数:
        df (DataFrame): 包含时间列的数据
    
    返回:
        DataFrame: 添加了时间特征的数据
    """
    if 'pickup_datetime' not in df.columns:
        return df
    
    print("提取时间特征...")
    
    # 提取基本时间特征
    df['hour'] = df['pickup_datetime'].dt.hour
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    df['day_of_year'] = df['pickup_datetime'].dt.dayofyear
    df['month'] = df['pickup_datetime'].dt.month
    
    # 创建时间段ID (论文中的时间段t)
    # 假设每个时间段为15分钟
    df['time_period'] = (df['hour'] * 4 + df['pickup_datetime'].dt.minute // 15)
    
    # 分类时间段
    df['is_weekend'] = df['day_of_week'].isin([5, 6])
    
    # 定义高峰时段
    morning_peak = (df['hour'] >= 7) & (df['hour'] <= 9)
    evening_peak = (df['hour'] >= 17) & (df['hour'] <= 19)
    df['is_peak_hour'] = morning_peak | evening_peak
    
    return df

def process_taxi_ids(df):
    """
    处理和标准化出租车ID。
    
    参数:
        df (DataFrame): 原始数据
    
    返回:
        DataFrame: 处理后的数据
    """
    # 寻找可能的出租车ID列
    id_columns = ['taxi_id', 'medallion', 'hack_license', 'vendor_id', 'VendorID']
    taxi_id_col = None
    
    for col in id_columns:
        if col in df.columns:
            taxi_id_col = col
            break
    
    if taxi_id_col:
        print(f"使用 {taxi_id_col} 作为出租车ID")
        df['taxi_id'] = df[taxi_id_col]
    else:
        print("未找到出租车ID列，创建模拟ID")
        # 基于行程模式创建虚拟的出租车ID
        # 假设平均每辆车每天完成20次行程
        n_taxis = max(100, len(df) // 20)
        df['taxi_id'] = np.random.randint(1, n_taxis + 1, len(df))
    
    # 确保taxi_id是整数类型
    df['taxi_id'] = df['taxi_id'].astype('int64')
    
    print(f"识别出 {df['taxi_id'].nunique()} 辆不同的出租车")
    
    return df

def calculate_trip_energy_consumption(df):
    """
    根据行程距离和时间估算能量消耗。
    
    参数:
        df (DataFrame): 行程数据
    
    返回:
        DataFrame: 添加了能量消耗列的数据
    """
    print("计算行程能量消耗...")
    
    # 如果有行程距离，基于距离计算
    if 'trip_distance' in df.columns:
        # 假设电动出租车每英里消耗0.3 kWh
        df['energy_consumption'] = df['trip_distance'] * 0.3
    else:
        # 如果没有距离数据，使用时间估算
        if 'pickup_datetime' in df.columns and 'dropoff_datetime' in df.columns:
            # 计算行程时间（分钟）
            trip_duration = (df['dropoff_datetime'] - df['pickup_datetime']).dt.total_seconds() / 60
            # 假设平均每分钟消耗0.02 kWh
            df['energy_consumption'] = trip_duration * 0.02
        else:
            # 如果都没有，使用平均值
            df['energy_consumption'] = 5.0  # 平均每次行程5 kWh
    
    # 限制能量消耗在合理范围内
    df['energy_consumption'] = df['energy_consumption'].clip(0.5, 30.0)
    
    return df

def aggregate_demand_by_time_space(df):
    """
    按时间和空间聚合需求数据。
    
    参数:
        df (DataFrame): 清洗后的行程数据
    
    返回:
        DataFrame: 聚合的需求数据
    """
    print("聚合时空需求数据...")
    
    # 按区块和时间段聚合
    demand_agg = df.groupby(['block_id', 'time_period', 'hour']).agg({
        'taxi_id': 'count',  # 需求数量
        'fare_amount': 'mean',  # 平均费用
        'trip_distance': 'mean',  # 平均距离
        'energy_consumption': 'sum'  # 总能量消耗
    }).reset_index()
    
    demand_agg.columns = ['block_id', 'time_period', 'hour', 'demand_count', 
                         'avg_fare', 'avg_distance', 'total_energy']
    
    return demand_agg

def prepare_simulation_data(df, config):
    """
    为模拟准备数据，包括需求分布和出租车初始状态。
    
    参数:
        df (DataFrame): 清洗后的数据
        config (dict): 模拟配置
    
    返回:
        dict: 包含模拟所需数据的字典
    """
    print("准备模拟数据...")
    
    # 计算能量消耗
    df = calculate_trip_energy_consumption(df)
    
    # 聚合需求数据
    demand_data = aggregate_demand_by_time_space(df)
    
    # 创建区块位置字典
    if 'pickup_latitude' in df.columns and 'pickup_longitude' in df.columns:
        block_positions = df.groupby('block_id').agg({
            'pickup_latitude': 'mean',
            'pickup_longitude': 'mean'
        }).to_dict('index')
        
        # 转换为 (x, y) 坐标
        block_positions = {
            bid: ((info['pickup_longitude'] + 74.0) * 100, 
                  (info['pickup_latitude'] - 40.5) * 100)
            for bid, info in block_positions.items()
        }
    else:
        # 创建网格布局
        unique_blocks = df['block_id'].unique()
        n = int(np.ceil(np.sqrt(len(unique_blocks))))
        block_positions = {
            bid: (i % n * 10, i // n * 10)
            for i, bid in enumerate(unique_blocks)
        }
    
    # 统计出租车信息
    taxi_info = df.groupby('taxi_id').agg({
        'pickup_datetime': 'min',  # 第一次出现时间
        'block_id': 'first',  # 初始位置
        'trip_distance': 'sum',  # 总行程距离
        'fare_amount': 'sum'  # 总收入
    }).reset_index()
    
    simulation_data = {
        'trip_data': df,
        'demand_data': demand_data,
        'block_positions': block_positions,
        'taxi_info': taxi_info,
        'summary': {
            'total_trips': len(df),
            'unique_taxis': df['taxi_id'].nunique(),
            'unique_blocks': df['block_id'].nunique(),
            'time_span': {
                'start': df['pickup_datetime'].min(),
                'end': df['pickup_datetime'].max()
            } if 'pickup_datetime' in df.columns else None
        }
    }
    
    return simulation_data

if __name__ == "__main__":
    # 测试数据加载和清洗
    test_file = "data/yellow_tripdata_2025-01.parquet"
    
    if os.path.exists(test_file):
        # 加载数据
        raw_data = load_trip_data(test_file, sample_size=10000)
        
        # 清洗数据
        clean_data = clean_trip_data(raw_data)
        
        # 准备模拟数据
        sim_data = prepare_simulation_data(clean_data, {})
        
        print("\n数据处理完成!")
        print(f"总行程数: {sim_data['summary']['total_trips']}")
        print(f"出租车数量: {sim_data['summary']['unique_taxis']}")
        print(f"区块数量: {sim_data['summary']['unique_blocks']}")
    else:
        print(f"测试文件不存在: {test_file}")