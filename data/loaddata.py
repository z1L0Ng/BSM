"""数据加载和清洗模块，用于处理纽约出租车(NYC TLC)行程记录。"""
import pandas as pd
import os
import numpy as np
from datetime import datetime

def load_trip_data(filepath, use_dask=False, chunksize=None):
    """
    从Parquet文件加载NYC TLC出租车行程数据。
    
    参数:
        filepath (str): 包含行程数据的Parquet文件路径
        use_dask (bool): 若为True，使用Dask加载大型数据
        chunksize (int): 每次加载的行数。仅当use_dask=False时使用
    
    返回:
        DataFrame: 包含行程记录的Pandas DataFrame（如果use_dask=True则为Dask DataFrame）
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"数据文件不存在: {filepath}")

    if use_dask:
        try:
            import dask.dataframe as dd
            # 使用Dask处理大型数据集
            df = dd.read_parquet(filepath)
            print(f"使用Dask成功加载数据，分区数量: {df.npartitions}")
            return df
        except ImportError:
            print("Dask未安装，将使用pandas继续。")
            use_dask = False
    
    if chunksize:
        # 对大文件使用分块读取
        return pd.read_parquet(filepath, engine='pyarrow', chunksize=chunksize)
    else:
        # 一次性读取整个文件
        df = pd.read_parquet(filepath, engine='pyarrow')
        print(f"成功加载数据，总行数: {len(df)}")
        return df

def clean_trip_data(df):
    """
    对行程数据DataFrame进行基础清洗。
    
    - 过滤掉包含缺失或无效数据的行程
    - 为模拟准备字段（例如，提取上车/下车坐标或区域）
    - 处理时间戳和地理位置数据
    
    返回:
        DataFrame: 已清洗并准备好进行模拟的DataFrame
    """
    # 检查是否为Dask DataFrame
    is_dask = hasattr(df, 'compute')
    
    # 基本的数据清洗步骤
    # 移除无效的乘客数量记录
    df = df[df.passenger_count > 0]
    
    # 移除异常的行程距离和时间
    df = df[df.trip_distance > 0]
    df = df[df.trip_duration > 0] if 'trip_duration' in df.columns else df
    
    # 确保有效的经纬度坐标
    valid_lat = (df.pickup_latitude >= 40.5) & (df.pickup_latitude <= 41.0)
    valid_lon = (df.pickup_longitude >= -74.1) & (df.pickup_longitude <= -73.7)
    df = df[valid_lat & valid_lon]
    
    # 添加时间特征
    if 'pickup_datetime' in df.columns:
        if is_dask:
            # Dask数据处理
            df['hour'] = df.pickup_datetime.dt.hour
            df['day_of_week'] = df.pickup_datetime.dt.dayofweek
        else:
            # Pandas数据处理
            df['hour'] = pd.to_datetime(df.pickup_datetime).dt.hour
            df['day_of_week'] = pd.to_datetime(df.pickup_datetime).dt.dayofweek
    
    # 添加网格区块ID (将NYC划分为网格)
    df = add_block_ids(df)
    
    if is_dask:
        df = df.compute()  # 将Dask DataFrame转换为Pandas DataFrame
    
    print(f"数据清洗完成，剩余行数: {len(df)}")
    return df

def add_block_ids(df):
    """
    将经纬度坐标映射到网格区块ID。
    
    参数:
        df (DataFrame): 包含pickup_latitude和pickup_longitude的DataFrame
    
    返回:
        DataFrame: 添加了block_id列的DataFrame
    """
    # 定义NYC的地理边界
    lat_min, lat_max = 40.5, 41.0
    lon_min, lon_max = -74.1, -73.7
    
    # 创建10x10的网格
    grid_size = 10
    lat_step = (lat_max - lat_min) / grid_size
    lon_step = (lon_max - lon_min) / grid_size
    
    # 计算区块ID
    df['lat_bin'] = ((df.pickup_latitude - lat_min) / lat_step).astype(int)
    df['lon_bin'] = ((df.pickup_longitude - lon_min) / lon_step).astype(int)
    
    # 限制区块索引范围以防止越界
    df['lat_bin'] = df['lat_bin'].clip(0, grid_size-1)
    df['lon_bin'] = df['lon_bin'].clip(0, grid_size-1)
    
    # 计算唯一区块ID (lat_bin * grid_size + lon_bin)
    df['block_id'] = df['lat_bin'] * grid_size + df['lon_bin']
    
    # 清除临时列
    df = df.drop(['lat_bin', 'lon_bin'], axis=1)
    
    return df