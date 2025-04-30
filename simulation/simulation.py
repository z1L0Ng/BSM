"""使用SimPy的主模拟流程。"""
import simpy
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from data.loaddata import load_trip_data, clean_trip_data
from environment.blocks import BlockNetwork
from agent.taxi import TaxiAgent
from bss.bss import BatterySwapStation
from scheduler.taxi_scheduler import ETaxiScheduler
from scheduler.bss_scheduler import StationScheduler
import os
import json

def prepare_data(config):
    """
    加载和准备模拟所需的数据。
    
    参数:
        config (dict): 模拟配置
    
    返回:
        tuple: (清洗后的行程数据, 区块位置字典)
    """
    data_config = config.get('data', {})
    data_path = data_config.get('filepath', '')
    use_sample = data_config.get('use_sample', True)
    sample_size = data_config.get('sample_size', 10000) if use_sample else None
    
    print("正在加载NYC出租车数据...")
    try:
        # 加载数据
        if os.path.exists(data_path):
            raw_data = load_trip_data(data_path)
            
            # 如果需要使用样本，抽取样本数据
            if use_sample and sample_size:
                raw_data = raw_data.sample(sample_size, random_state=42)
            
            # 清洗数据
            clean_data = clean_trip_data(raw_data)
            
            # 创建区块位置字典
            block_positions = create_block_positions(clean_data)
            
            print(f"数据准备完成，{len(clean_data)}条行程记录，{len(block_positions)}个区块。")
            return clean_data, block_positions
        else:
            print(f"警告: 数据文件不存在: {data_path}")
            print("将使用模拟数据进行测试。")
            return None, generate_test_blocks(10)
    except Exception as e:
        print(f"数据加载错误: {e}")
        print("将使用模拟数据进行测试。")
        return None, generate_test_blocks(10)

def create_block_positions(data):
    """
    从处理后的数据创建区块位置字典。
    
    参数:
        data (DataFrame): 包含区块ID的预处理行程数据
    
    返回:
        dict: 区块ID到(x,y)坐标的映射
    """
    # 按区块ID分组并计算每个区块的平均坐标
    if 'block_id' not in data.columns:
        print("警告: 数据中没有block_id列，将使用模拟区块。")
        return generate_test_blocks(10)
    
    block_coords = data.groupby('block_id').agg({
        'pickup_latitude': 'mean',
        'pickup_longitude': 'mean'
    }).reset_index()
    
    # 转换为BlockNetwork预期的格式
    block_positions = {}
    for _, row in block_coords.iterrows():
        block_id = int(row['block_id'])
        # 转换为BlockNetwork预期的格式
    block_positions = {}
    for _, row in block_coords.iterrows():
        block_id = int(row['block_id'])
        # 将经纬度坐标缩放到更易于管理的范围
        x = (row['pickup_longitude'] + 74.05) * 100  # 平移和缩放
        y = (row['pickup_latitude'] - 40.5) * 100  # 平移和缩放
        block_positions[block_id] = (x, y)
    
    return block_positions

def generate_test_blocks(grid_size=10):
    """
    生成用于测试的模拟区块网格。
    
    参数:
        grid_size (int): 网格大小
    
    返回:
        dict: 区块ID到(x,y)坐标的映射
    """
    block_positions = {}
    
    # 创建一个grid_size x grid_size的网格
    for i in range(grid_size):
        for j in range(grid_size):
            block_id = i * grid_size + j
            # 将坐标放在0-100的范围内
            x = i * 10
            y = j * 10
            block_positions[block_id] = (x, y)
    
    return block_positions

def optimize_station_locations(block_positions, trip_data=None, num_stations=10):
    """
    确定电池交换站的最佳位置。
    
    参数:
        block_positions (dict): 区块ID到(x,y)坐标的映射
        trip_data (DataFrame): 预处理的行程数据
        num_stations (int): 要放置的站点数量
    
    返回:
        list: 站点放置的区块ID列表
    """
    if trip_data is not None and 'block_id' in trip_data.columns:
        # 将站点放在行程密度最高的区块
        trip_counts = trip_data['block_id'].value_counts().head(num_stations)
        station_blocks = trip_counts.index.tolist()
    else:
        # 如果没有行程数据，均匀分布站点
        all_blocks = list(block_positions.keys())
        if len(all_blocks) < num_stations:
            # 如果可用区块少于所需站点数，重复使用区块
            station_blocks = all_blocks * (num_stations // len(all_blocks) + 1)
            station_blocks = station_blocks[:num_stations]
        else:
            # 随机选择区块放置站点
            station_blocks = random.sample(all_blocks, num_stations)
    
    return station_blocks

def setup_simulation(env, config, trip_data=None):
    """
    初始化模拟实体(网络、站点、出租车)。
    
    参数:
        env (simpy.Environment): SimPy环境
        config (dict): 模拟配置
        trip_data (DataFrame): 预处理的行程数据
    
    返回:
        tuple: (网络对象, 出租车列表, 站点列表, 调度器对象, 站点调度器列表)
    """
    # 创建区块网络
    block_positions = config.get('blocks', {})
    if not block_positions:
        # 如果配置中没有区块位置，使用测试网格
        block_positions = generate_test_blocks(10)
    
    network = BlockNetwork(block_positions, consider_traffic=True)
    
    # 初始化站点
    stations = []
    station_configs = config.get('stations', [])
    
    # 如果没有明确的站点配置但有站点数量，自动生成站点配置
    if not station_configs and 'num_stations' in config:
        num_stations = config['num_stations']
        station_blocks = optimize_station_locations(block_positions, trip_data, num_stations)
        
        # 为每个位置创建默认站点配置
        for i, block_id in enumerate(station_blocks):
            station_configs.append({
                'block_id': block_id,
                'capacity': 20,
                'initial_batteries': 15,
                'swap_time': 5,
                'charge_time': 60,
                'chargers': 5
            })
    
    # 创建站点
    for i, station_conf in enumerate(station_configs):
        station = BatterySwapStation(
            env=env, 
            station_id=i,
            location=station_conf['block_id'],
            capacity=station_conf.get('capacity', 20),
            initial_batteries=station_conf.get('initial_batteries', 15),
            swap_time=station_conf.get('swap_time', 5),
            charge_time=station_conf.get('charge_time', 60),
            n_chargers=station_conf.get('chargers', 5)
        )
        stations.append(station)
    
    # 为每个站点创建站点调度器
    station_schedulers = []
    for station in stations:
        scheduler = StationScheduler(
            station=station,
            check_interval=10,
            min_charged=station.capacity // 4,  # 保持至少1/4的电池充满电
            dynamic_charging=True
        )
        station_schedulers.append(scheduler)
    
    # 初始化出租车
    taxis = []
    taxi_conf = config.get('taxis', {})
    num_taxis = taxi_conf.get('count', 100)
    
    # 确定出租车的初始位置
    if trip_data is not None and 'block_id' in trip_data.columns:
        # 使用行程数据中最常见的区块作为出租车起点
        popular_blocks = trip_data['block_id'].value_counts().head(20).index.tolist()
    else:
        # 如果没有行程数据，使用所有区块
        popular_blocks = list(block_positions.keys())
    
    # 创建出租车
    for i in range(num_taxis):
        # 在热门区块中分配初始位置
        if 'initial_locations' in taxi_conf:
            init_block = taxi_conf['initial_locations'][i % len(taxi_conf['initial_locations'])]
        else:
            init_block = popular_blocks[i % len(popular_blocks)]
        
        taxi = TaxiAgent(
            env=env,
            taxi_id=i,
            initial_block=init_block,
            battery_capacity=taxi_conf.get('battery_capacity', 100),
            consumption_rate=taxi_conf.get('consumption_rate', 0.5),
            swap_threshold=taxi_conf.get('swap_threshold', 20)
        )
        taxis.append(taxi)
        
        # 启动出租车进程
        env.process(taxi.process(network, stations, trip_data))
    
    # 创建中央调度器
    use_real_data = config.get('use_real_data', False) and trip_data is not None
    scheduler = ETaxiScheduler(
        env=env,
        taxis=taxis,
        stations=stations,
        network=network,
        check_interval=5,
        use_real_data=use_real_data,
        trip_data=trip_data
    )
    
    return network, taxis, stations, scheduler, station_schedulers

def collect_results(env, taxis, stations, scheduler, config):
    """
    在模拟结束时收集结果。
    
    参数:
        env (simpy.Environment): SimPy环境
        taxis (list): 出租车对象列表
        stations (list): 站点对象列表
        scheduler (ETaxiScheduler): 调度器对象
        config (dict): 模拟配置
    
    返回:
        dict: 模拟结果和统计数据
    """
    # 收集站点统计
    station_stats = {
        'total_swaps': sum(s.swap_count for s in stations),
        'station_swap_counts': {s.id: s.swap_count for s in stations},
        'station_avg_wait': {s.id: (sum(s.wait_times)/len(s.wait_times) if s.wait_times else 0) for s in stations},
        'station_locations': {s.id: s.location for s in stations},
        'charged_batteries': {s.id: s.charged_batteries.level for s in stations},
        'empty_batteries': {s.id: s.empty_batteries.level for s in stations}
    }
    
    # 收集出租车统计
    taxi_stats = {
        'trips_completed': sum(t.trips_completed for t in taxis),
        'trips_per_taxi': {t.id: t.trips_completed for t in taxis},
        'swap_count': {t.id: t.swap_count for t in taxis},
        'total_revenue': sum(t.total_revenue for t in taxis),
        'revenue_per_taxi': {t.id: t.total_revenue for t in taxis},
        'total_distance': sum(t.total_distance for t in taxis),
        'distance_per_taxi': {t.id: t.total_distance for t in taxis},
        'final_locations': {t.id: t.location for t in taxis},
        'final_battery_levels': {t.id: t.battery_level for t in taxis}
    }
    
    # 计算性能指标
    total_revenue = sum(t.total_revenue for t in taxis)
    total_distance = sum(t.total_distance for t in taxis)
    total_trips = sum(t.trips_completed for t in taxis)
    total_swaps = sum(s.swap_count for s in stations)
    
    performance_metrics = {
        'avg_revenue_per_km': total_revenue / total_distance if total_distance > 0 else 0,
        'avg_trips_per_taxi': total_trips / len(taxis) if taxis else 0,
        'avg_swaps_per_taxi': total_swaps / len(taxis) if taxis else 0,
        'avg_distance_per_trip': total_distance / total_trips if total_trips > 0 else 0,
        'avg_revenue_per_trip': total_revenue / total_trips if total_trips > 0 else 0,
        'simulation_duration': env.now
    }
    
    # 合并所有结果
    results = {
        'station_stats': station_stats,
        'taxi_stats': taxi_stats,
        'performance_metrics': performance_metrics,
        'scheduler_stats': scheduler.get_status()
    }
    
    return results

def visualize_results(results, block_positions, save_path=None):
    """
    可视化模拟结果。
    
    参数:
        results (dict): 模拟结果
        block_positions (dict): 区块位置字典
        save_path (str): 保存图表的路径(如果需要)
    """
    # 创建一个包含多个子图的图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 图1: 包含区块位置和站点的地图
    ax1 = axes[0, 0]
    
    # 绘制所有区块
    for block_id, (x, y) in block_positions.items():
        ax1.scatter(x, y, c='gray', alpha=0.3, s=10)
    
    # 高亮显示站点
    station_locations = results['station_stats']['station_locations']
    station_swap_counts = results['station_stats']['station_swap_counts']
    
    for station_id, block_id in station_locations.items():
        if block_id in block_positions:
            x, y = block_positions[block_id]
            swap_count = station_swap_counts.get(station_id, 0)
            # 根据换电次数调整圆圈大小
            size = max(100, swap_count * 5)
            ax1.scatter(x, y, c='red', s=size, alpha=0.7)
            ax1.annotate(f"S{station_id}: {swap_count}", (x, y), fontsize=8)
    
    ax1.set_title('区块位置和换电站位置')
    ax1.set_xlabel('X坐标')
    ax1.set_ylabel('Y坐标')
    
    # 图2: 每个站点的电池更换次数
    ax2 = axes[0, 1]
    station_ids = list(results['station_stats']['station_swap_counts'].keys())
    swap_counts = [results['station_stats']['station_swap_counts'][i] for i in station_ids]
    ax2.bar(station_ids, swap_counts)
    ax2.set_title('每个站点的电池更换次数')
    ax2.set_xlabel('站点ID')
    ax2.set_ylabel('更换次数')
    
    # 图3: 每个站点的平均等待时间
    ax3 = axes[1, 0]
    station_ids = list(results['station_stats']['station_avg_wait'].keys())
    wait_times = [results['station_stats']['station_avg_wait'][i] for i in station_ids]
    ax3.bar(station_ids, wait_times)
    ax3.set_title('每个站点的平均等待时间')
    ax3.set_xlabel('站点ID')
    ax3.set_ylabel('等待时间(分钟)')
    
    # 图4: 出租车之间的行程分布
    ax4 = axes[1, 1]
    trips = list(results['taxi_stats']['trips_per_taxi'].values())
    ax4.hist(trips, bins=20)
    ax4.set_title('出租车之间的行程分布')
    ax4.set_xlabel('行程数量')
    ax4.set_ylabel('出租车数量')
    
    plt.tight_layout()
    
    # 保存图表(如果指定了路径)
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def run_simulation(config):
    """
    使用给定配置运行模拟。
    
    参数:
        config (dict): 模拟配置
    
    返回:
        dict: 模拟结果和统计数据
    """
    # 准备数据
    trip_data, block_positions = prepare_data(config)
    
    # 更新配置中的区块位置
    config['blocks'] = block_positions
    
    # 创建SimPy环境
    env = simpy.Environment()
    
    # 设置模拟
    network, taxis, stations, scheduler, station_schedulers = setup_simulation(env, config, trip_data)
    
    # 运行指定时长的模拟
    duration = config.get('simulation', {}).get('duration', 1440)  # 默认为1440分钟(24小时)
    
    if duration <= 0:
        print("请在配置中指定一个正的模拟时长。")
        return {}
    
    print(f"开始运行{duration}分钟的模拟...")
    
    # 使用tqdm显示进度
    progress_bar = tqdm(total=duration, desc="模拟进度")
    
    # 分步运行模拟，每10分钟更新一次进度条
    step_size = 10
    for i in range(0, duration, step_size):
        next_step = min(i + step_size, duration)
        env.run(until=next_step)
        progress_bar.update(next_step - i)
    
    progress_bar.close()
    
    # 收集结果
    results = collect_results(env, taxis, stations, scheduler, config)
    
    print("模拟完成。\n结果:")
    print(f"- 完成的行程总数: {results['taxi_stats']['trips_completed']}")
    print(f"- 电池更换总次数: {results['station_stats']['total_swaps']}")
    print(f"- 总收入: ${sum(results['taxi_stats']['revenue_per_taxi'].values()):.2f}")
    print(f"- 总行驶距离: {sum(results['taxi_stats']['distance_per_taxi'].values()):.2f} km")
    
    # 每个站点的详细信息
    print("\n站点详情:")
    for sid, count in results['station_stats']['station_swap_counts'].items():
        print(f"- 站点 {sid}: {count} 次换电, 平均等待 {results['station_stats']['station_avg_wait'][sid]:.2f} 分钟")
    
    # 可视化结果
    visualize_results(results, block_positions, save_path="simulation_results.png")
    
    # 保存结果到JSON文件
    with open("simulation_results.json", "w") as f:
        # 转换不可序列化的对象(如numpy数组)为标准Python类型
        json_results = json.dumps(results, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
        f.write(json_results)
    
    return results

# 如果直接执行此模块，运行一个小型演示模拟
if __name__ == "__main__":
    # 演示配置
    demo_config = {
        'blocks': None,  # 将自动生成测试区块
        'num_stations': 5,  # 要放置的站点数量
        'taxis': {
            'count': 50,  # 出租车数量
            'battery_capacity': 100,  # 电池容量(kWh)
            'consumption_rate': 0.5,  # 消耗率(kWh/分钟)
            'swap_threshold': 20  # 换电阈值
        },
        'data': {
            'filepath': 'data/nyc_taxi_sample.parquet',  # 数据文件路径
            'use_sample': True,
            'sample_size': 5000  # 样本大小
        },
        'use_real_data': False,  # 是否使用真实数据分配行程
        'simulation': {
            'duration': 720  # 模拟持续时间(分钟) - 12小时
        }
    }
    
    # 运行演示模拟
    results = run_simulation(demo_config)