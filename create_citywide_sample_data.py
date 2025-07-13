"""
生成城市级规模的样本数据以支持大规模模拟
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_citywide_sample_data(n_rows=50000, n_areas=100, n_taxis=1000):
    """
    生成城市级规模的样本数据
    
    参数:
        n_rows: 生成的行程记录数
        n_areas: 城市区域数量
        n_taxis: 出租车数量
    """
    print(f"生成城市级样本数据: {n_rows} 条记录, {n_areas} 个区域, {n_taxis} 辆出租车")
    
    # 设置随机种子
    np.random.seed(42)
    
    # 创建24小时的时间分布
    base_date = pd.Timestamp('2025-01-01')
    
    # 生成带有真实时间分布的时间戳
    hours = np.random.choice(24, n_rows, p=create_hourly_distribution())
    minutes = np.random.randint(0, 60, n_rows)
    pickup_times = [base_date + timedelta(hours=int(h), minutes=int(m)) 
                   for h, m in zip(hours, minutes)]
    
    # 生成地理坐标 (模拟NYC区域)
    # 中心点坐标
    center_lat, center_lon = 40.7589, -73.9851
    
    # 生成更真实的地理分布
    pickup_lats = np.random.normal(center_lat, 0.1, n_rows)
    pickup_lons = np.random.normal(center_lon, 0.1, n_rows)
    
    # 生成目的地（相对于起点的偏移）
    dropoff_lats = pickup_lats + np.random.normal(0, 0.05, n_rows)
    dropoff_lons = pickup_lons + np.random.normal(0, 0.05, n_rows)
    
    # 确保坐标在合理范围内
    pickup_lats = np.clip(pickup_lats, 40.4, 41.0)
    pickup_lons = np.clip(pickup_lons, -74.3, -73.7)
    dropoff_lats = np.clip(dropoff_lats, 40.4, 41.0)
    dropoff_lons = np.clip(dropoff_lons, -74.3, -73.7)
    
    # 生成与时间相关的出行距离
    trip_distances = generate_realistic_trip_distances(hours, n_rows)
    
    # 生成与距离相关的费用
    base_fare = 3.0
    per_km_fare = 2.5
    total_amounts = base_fare + trip_distances * per_km_fare + np.random.normal(0, 2, n_rows)
    total_amounts = np.clip(total_amounts, 5, 200)  # 合理的费用范围
    
    # 生成乘客数量（大部分是1人）
    passenger_counts = np.random.choice([1, 2, 3, 4], n_rows, p=[0.7, 0.2, 0.08, 0.02])
    
    # 生成出租车ID
    taxi_ids = np.random.randint(1, n_taxis + 1, n_rows)
    
    # 创建DataFrame
    data = {
        'tpep_pickup_datetime': pickup_times,
        'pickup_latitude': pickup_lats,
        'pickup_longitude': pickup_lons,
        'dropoff_latitude': dropoff_lats,
        'dropoff_longitude': dropoff_lons,
        'passenger_count': passenger_counts,
        'trip_distance': trip_distances,
        'total_amount': total_amounts,
        'taxi_id': taxi_ids,
        'fare_amount': total_amounts * 0.8,  # 基础费用约占80%
        'tip_amount': total_amounts * 0.15,  # 小费约占15%
        'tolls_amount': np.random.exponential(1, n_rows) * 0.3,  # 过路费
        'extra': np.random.choice([0, 0.5, 1], n_rows, p=[0.8, 0.15, 0.05]),  # 额外费用
    }
    
    df = pd.DataFrame(data)
    
    # 检查并处理重复列名
    print(f"列名: {list(df.columns)}")
    if df.columns.duplicated().any():
        print("发现重复列名，正在处理...")
        df = df.loc[:, ~df.columns.duplicated()]
    
    # 确保数据目录存在
    os.makedirs('data', exist_ok=True)
    
    # 保存为parquet格式
    output_file = 'data/citywide_sample_data.parquet'
    df.to_parquet(output_file, index=False)
    
    print(f'城市级样本数据已创建: {output_file}')
    print(f'数据统计:')
    print(f'  - 总记录数: {len(df)}')
    print(f'  - 唯一出租车: {df["taxi_id"].nunique()}')
    print(f'  - 时间范围: {df["tpep_pickup_datetime"].min()} 到 {df["tpep_pickup_datetime"].max()}')
    print(f'  - 平均行程距离: {df["trip_distance"].mean():.2f} 公里')
    print(f'  - 平均费用: {df["total_amount"].mean():.2f} 美元')
    
    return df

def create_hourly_distribution():
    """创建24小时的真实出行需求分布"""
    # 基于真实数据的小时分布模式
    hourly_weights = np.array([
        0.01, 0.01, 0.01, 0.01, 0.01, 0.02,  # 0-5am: 很少
        0.03, 0.06, 0.08, 0.07, 0.05, 0.04,  # 6-11am: 早高峰
        0.05, 0.06, 0.07, 0.06, 0.05, 0.07,  # 12-17pm: 午间和下午
        0.09, 0.08, 0.07, 0.05, 0.04, 0.02   # 18-23pm: 晚高峰
    ])
    
    # 归一化
    return hourly_weights / hourly_weights.sum()

def generate_realistic_trip_distances(hours, n_rows):
    """生成与时间相关的真实出行距离"""
    distances = []
    
    for hour in hours:
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # 高峰时段
            # 高峰时段距离更短（市内通勤）
            distance = np.random.exponential(3.0)
        elif 22 <= hour <= 23 or 0 <= hour <= 5:  # 夜间
            # 夜间距离更长
            distance = np.random.exponential(5.0)
        else:  # 其他时段
            # 正常分布
            distance = np.random.exponential(4.0)
        
        # 限制在合理范围内
        distances.append(min(distance, 50))
    
    return np.array(distances)

def generate_multiple_days_data(days=7, n_rows_per_day=7000):
    """生成多天的数据"""
    print(f"生成 {days} 天的数据，每天 {n_rows_per_day} 条记录")
    
    all_data = []
    base_date = pd.Timestamp('2025-01-01')
    
    for day in range(days):
        current_date = base_date + timedelta(days=day)
        
        # 生成当天的数据
        daily_data = generate_single_day_data(current_date, n_rows_per_day)
        all_data.append(daily_data)
        
        print(f"已生成第 {day + 1} 天的数据")
    
    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 保存合并后的数据
    output_file = 'data/citywide_multiday_data.parquet'
    combined_df.to_parquet(output_file, index=False)
    
    print(f'多天数据已创建: {output_file}')
    print(f'总记录数: {len(combined_df)}')
    
    return combined_df

def generate_single_day_data(date, n_rows):
    """生成单天的数据"""
    np.random.seed(42 + date.day)  # 每天不同的随机种子
    
    # 生成时间戳
    hours = np.random.choice(24, n_rows, p=create_hourly_distribution())
    minutes = np.random.randint(0, 60, n_rows)
    pickup_times = [date + timedelta(hours=int(h), minutes=int(m)) 
                   for h, m in zip(hours, minutes)]
    
    # 生成地理坐标
    center_lat, center_lon = 40.7589, -73.9851
    pickup_lats = np.random.normal(center_lat, 0.1, n_rows)
    pickup_lons = np.random.normal(center_lon, 0.1, n_rows)
    
    dropoff_lats = pickup_lats + np.random.normal(0, 0.05, n_rows)
    dropoff_lons = pickup_lons + np.random.normal(0, 0.05, n_rows)
    
    # 确保坐标在合理范围内
    pickup_lats = np.clip(pickup_lats, 40.4, 41.0)
    pickup_lons = np.clip(pickup_lons, -74.3, -73.7)
    dropoff_lats = np.clip(dropoff_lats, 40.4, 41.0)
    dropoff_lons = np.clip(dropoff_lons, -74.3, -73.7)
    
    # 生成其他字段
    trip_distances = generate_realistic_trip_distances(hours, n_rows)
    total_amounts = 3.0 + trip_distances * 2.5 + np.random.normal(0, 2, n_rows)
    total_amounts = np.clip(total_amounts, 5, 200)
    
    passenger_counts = np.random.choice([1, 2, 3, 4], n_rows, p=[0.7, 0.2, 0.08, 0.02])
    taxi_ids = np.random.randint(1, 1001, n_rows)
    
    data = {
        'tpep_pickup_datetime': pickup_times,
        'pickup_latitude': pickup_lats,
        'pickup_longitude': pickup_lons,
        'dropoff_latitude': dropoff_lats,
        'dropoff_longitude': dropoff_lons,
        'passenger_count': passenger_counts,
        'trip_distance': trip_distances,
        'total_amount': total_amounts,
        'taxi_id': taxi_ids,
        'fare_amount': total_amounts * 0.8,
        'tip_amount': total_amounts * 0.15,
        'tolls_amount': np.random.exponential(1, n_rows) * 0.3,
        'extra': np.random.choice([0, 0.5, 1], n_rows, p=[0.8, 0.15, 0.05]),
    }
    
    df = pd.DataFrame(data)
    
    # 检查并处理重复列名
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    
    return df

if __name__ == "__main__":
    print("开始生成城市级样本数据...")
    
    # 生成单天的大规模数据
    single_day_df = generate_citywide_sample_data(n_rows=50000, n_areas=100, n_taxis=1000)
    
    # 生成多天的数据
    multi_day_df = generate_multiple_days_data(days=7, n_rows_per_day=7000)
    
    print("城市级样本数据生成完成！")
    print("\n可用的数据文件:")
    print("  - data/citywide_sample_data.parquet (单天大规模数据)")
    print("  - data/citywide_multiday_data.parquet (多天数据)")
    print("\n请在配置文件中设置对应的数据文件路径。") 