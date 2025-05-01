"""使用SimPy的主模拟流程。"""
import simpy
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataprocess.loaddata import load_trip_data, clean_trip_data
from environment.blocks import BlockNetwork
from agent.taxi import TaxiAgent
from bss.bss import BatterySwapStation
from scheduler.taxi_scheduler import ETaxiScheduler
from scheduler.bss_scheduler import StationScheduler
from optimization.interface import optimize_bss_layout
import os
import json
from datetime import datetime

def prepare_data(config, input_data=None):
    """
    加载和准备模拟所需的数据。
    
    参数:
        config (dict): 模拟配置
        input_data (DataFrame, optional): 已加载的行程数据
    
    返回:
        tuple: (清洗后的行程数据, 区块位置字典)
    """
    if input_data is not None:
        # 如果已经提供了数据，直接使用
        trip_data = input_data
        print(f"使用提供的数据，共 {len(trip_data)} 条记录")
    else:
        # 否则加载数据
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
                trip_data = clean_trip_data(raw_data)
                
                print(f"数据准备完成，{len(trip_data)}条行程记录。")
            else:
                print(f"警告: 数据文件不存在: {data_path}")
                print("将使用模拟数据进行测试。")
                return None, generate_test_blocks(10)
        except Exception as e:
            print(f"数据加载错误: {e}")
            print("将使用模拟数据进行测试。")
            return None, generate_test_blocks(10)
    
    # 创建区块位置字典
    block_positions = create_block_positions(trip_data)
    
    return trip_data, block_positions

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
    
    # 检查是否有坐标列
    if 'pickup_latitude' in data.columns and 'pickup_longitude' in data.columns:
        # 使用经纬度计算区块位置
        block_coords = data.groupby('block_id').agg({
            'pickup_latitude': 'mean',
            'pickup_longitude': 'mean'
        }).reset_index()
        
        # 转换为BlockNetwork预期的格式
        block_positions = {}
        for _, row in block_coords.iterrows():
            block_id = int(row['block_id'])
            # 将经纬度坐标缩放到更易于管理的范围
            x = (row['pickup_longitude'] + 74.05) * 100  # 平移和缩放
            y = (row['pickup_latitude'] - 40.5) * 100  # 平移和缩放
            block_positions[block_id] = (x, y)
    else:
        # 如果没有坐标列，使用区块ID的二维映射
        # 这适用于新的TLC格式，它使用区域ID而不是坐标
        unique_blocks = data['block_id'].unique()
        n = int(np.ceil(np.sqrt(len(unique_blocks))))  # 创建近似正方形网格
        
        block_positions = {}
        for i, block_id in enumerate(unique_blocks):
            row = i // n
            col = i % n
            block_positions[int(block_id)] = (col * 10, row * 10)  # 10单位网格间距
    
    print(f"创建了 {len(block_positions)} 个区块位置")
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

def get_trip_demand_points(trip_data):
    """
    从行程数据提取需求点，用于站点优化。
    
    参数:
        trip_data (DataFrame): 包含行程数据的DataFrame
    
    返回:
        dict: 需求点ID到计数的映射
    """
    if 'block_id' not in trip_data.columns:
        print("警告: 无法从数据提取需求点，使用随机需求。")
        return {}
    
    # 按区块ID计算行程数量
    demand_counts = trip_data['block_id'].value_counts().to_dict()
    return {int(block_id): count for block_id, count in demand_counts.items()}

def setup_simulation(env, config, trip_data=None, block_positions=None):
    """
    初始化模拟实体(网络、站点、出租车)。
    
    参数:
        env (simpy.Environment): SimPy环境
        config (dict): 模拟配置
        trip_data (DataFrame): 预处理的行程数据
        block_positions (dict): 区块ID到(x,y)坐标的映射
    
    返回:
        tuple: (网络对象, 出租车列表, 站点列表, 调度器对象, 站点调度器列表)
    """
    # 创建区块网络
    if block_positions is None:
        block_positions = config.get('blocks', {})
        if not block_positions:
            # 如果配置中没有区块位置，使用测试网格
            block_positions = generate_test_blocks(10)
    
    network = BlockNetwork(block_positions, consider_traffic=config['simulation'].get('consider_traffic', True))
    
    # 初始化站点位置
    stations = []
    station_configs = config.get('stations', [])
    
    # 如果没有明确的站点配置但有站点数量，自动生成站点配置
    if not station_configs and 'num_stations' in config:
        num_stations = config['num_stations']
        
        # 从行程数据提取需求点
        if trip_data is not None:
            demand_points = get_trip_demand_points(trip_data)
            
            # 执行站点位置优化
            candidate_locations = {block_id: pos for block_id, pos in block_positions.items()}
            
            print(f"正在为 {num_stations} 个换电站优化位置...")
            optimization_method = config.get('stations', {}).get('optimization', {}).get('method', 'p-median')
            max_distance = config.get('stations', {}).get('optimization', {}).get('max_distance', None)
            
            if optimization_method == 'max-coverage' and max_distance is not None:
                # 使用最大覆盖模型
                station_blocks = optimize_bss_layout(
                    demand_points, candidate_locations, num_stations, max_distance=max_distance
                )
            else:
                # 使用p-中位点模型
                station_blocks = optimize_bss_layout(
                    demand_points, candidate_locations, num_stations
                )
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
        
        # 如果有唯一出租车ID，则使用它们作为初始位置
        if 'taxi_id' in trip_data.columns and trip_data['taxi_id'].nunique() >= num_taxis:
            # 获取出租车ID和初始位置的对应关系
            taxi_locations = trip_data.groupby('taxi_id')['block_id'].first().iloc[:num_taxis].to_dict()
            
            # 创建实际出租车数量的出租车
            actual_num_taxis = min(num_taxis, len(taxi_locations))
            print(f"根据行程数据创建 {actual_num_taxis} 辆出租车")
            
            for i, (taxi_id, initial_block) in enumerate(taxi_locations.items()):
                if i >= num_taxis:
                    break
                
                taxi = TaxiAgent(
                    env=env,
                    taxi_id=int(taxi_id),
                    initial_block=int(initial_block),
                    battery_capacity=taxi_conf.get('battery_capacity', 100),
                    consumption_rate=taxi_conf.get('consumption_rate', 0.5),
                    swap_threshold=taxi_conf.get('swap_threshold', 20)
                )
                taxis.append(taxi)
                
                # 启动出租车进程
                env.process(taxi.process(network, stations, trip_data))
            
            # 如果需要创建额外的出租车来达到请求的数量
            additional_taxis = num_taxis - actual_num_taxis
            if additional_taxis > 0:
                print(f"创建额外的 {additional_taxis} 辆出租车")
                for i in range(additional_taxis):
                    init_block = popular_blocks[i % len(popular_blocks)]
                    
                    taxi = TaxiAgent(
                        env=env,
                        taxi_id=i + actual_num_taxis,
                        initial_block=int(init_block),
                        battery_capacity=taxi_conf.get('battery_capacity', 100),
                        consumption_rate=taxi_conf.get('consumption_rate', 0.5),
                        swap_threshold=taxi_conf.get('swap_threshold', 20)
                    )
                    taxis.append(taxi)
                    
                    # 启动出租车进程
                    env.process(taxi.process(network, stations, trip_data))
        else:
            # 如果没有唯一出租车ID，使用热门区块
            for i in range(num_taxis):
                # 在热门区块中分配初始位置
                if 'initial_locations' in taxi_conf:
                    init_block = taxi_conf['initial_locations'][i % len(taxi_conf['initial_locations'])]
                else:
                    init_block = popular_blocks[i % len(popular_blocks)]
                
                taxi = TaxiAgent(
                    env=env,
                    taxi_id=i,
                    initial_block=int(init_block),
                    battery_capacity=taxi_conf.get('battery_capacity', 100),
                    consumption_rate=taxi_conf.get('consumption_rate', 0.5),
                    swap_threshold=taxi_conf.get('swap_threshold', 20)
                )
                taxis.append(taxi)
                
                # 启动出租车进程
                env.process(taxi.process(network, stations, trip_data))
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
                initial_block=int(init_block),
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

def collect_results(env, taxis, stations, scheduler, config, block_positions):
    """
    在模拟结束时收集结果。
    
    参数:
        env (simpy.Environment): SimPy环境
        taxis (list): 出租车对象列表
        stations (list): 站点对象列表
        scheduler (ETaxiScheduler): 调度器对象
        config (dict): 模拟配置
        block_positions (dict): 区块位置字典
    
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
        'scheduler_stats': scheduler.get_status(),
        'blocks': block_positions  # 包含区块位置信息
    }
    
    # 如果配置中指定了输出目录，保存详细结果
    if 'output_dir' in config:
        output_dir = config['output_dir']
        
        # 保存每个出租车的详细统计信息
        taxi_details_file = os.path.join(output_dir, 'taxi_details.csv')
        taxi_details = []
        for taxi in taxis:
            taxi_details.append({
                'id': taxi.id,
                'trips_completed': taxi.trips_completed,
                'swap_count': taxi.swap_count,
                'revenue': taxi.total_revenue,
                'distance': taxi.total_distance,
                'final_location': taxi.location,
                'battery_level': taxi.battery_level,
                'idle_time': taxi.idle_time,
                'service_time': taxi.service_time,
                'charging_time': taxi.charging_time,
                'waiting_time': taxi.waiting_time
            })
        
        pd.DataFrame(taxi_details).to_csv(taxi_details_file, index=False)
        print(f"保存出租车详细信息到: {taxi_details_file}")
        
        # 保存每个站点的详细统计信息
        station_details_file = os.path.join(output_dir, 'station_details.csv')
        station_details = []
        for station in stations:
            station_details.append({
                'id': station.id,
                'location': station.location,
                'swap_count': station.swap_count,
                'charged_batteries': station.charged_batteries.level,
                'empty_batteries': station.empty_batteries.level,
                'avg_wait_time': sum(station.wait_times)/len(station.wait_times) if station.wait_times else 0,
                'avg_queue_length': sum(station.queue_lengths)/len(station.queue_lengths) if station.queue_lengths else 0,
                'utilization': sum(station.utilization_samples)/len(station.utilization_samples) if station.utilization_samples else 0
            })
        
        pd.DataFrame(station_details).to_csv(station_details_file, index=False)
        print(f"保存站点详细信息到: {station_details_file}")
    
    return results

def run_simulation(config, input_trip_data=None):
    """
    使用给定配置运行模拟。
    
    参数:
        config (dict): 模拟配置
        input_trip_data (DataFrame, optional): 已经加载的行程数据
    
    返回:
        dict: 模拟结果和统计数据
    """
    # 准备数据
    trip_data, block_positions = prepare_data(config, input_trip_data)
    
    # 更新配置中的区块位置
    config['blocks'] = block_positions
    
    # 创建SimPy环境
    env = simpy.Environment()
    
    # 如果指定了起始小时，设置初始时间
    start_hour = config.get('simulation', {}).get('start_hour', 0)
    env.run(until=start_hour * 60)  # 跳转到指定的起始小时
    
    # 设置模拟
    network, taxis, stations, scheduler, station_schedulers = setup_simulation(
        env, config, trip_data, block_positions
    )
    
    # 运行指定时长的模拟
    duration = config.get('simulation', {}).get('duration', 1440)  # 默认为1440分钟(24小时)
    
    if duration <= 0:
        print("请在配置中指定一个正的模拟时长。")
        return {}
    
    print(f"开始运行{duration}分钟的模拟，从小时 {start_hour} 开始...")
    
    # 使用tqdm显示进度
    progress_bar = tqdm(total=duration, desc="模拟进度")
    
    # 设置记录点，保存中间状态
    record_states = config.get('output', {}).get('save_states', False)
    state_interval = config.get('output', {}).get('state_interval', 60)
    simulation_states = [] if record_states else None
    
    # 分步运行模拟，每10分钟更新一次进度条
    step_size = 10
    for i in range(0, duration, step_size):
        next_step = min(i + step_size, duration)
        env.run(until=start_hour * 60 + next_step)
        progress_bar.update(next_step - i)
        
        # 如果需要记录状态，且到达了记录间隔
        if record_states and i % state_interval == 0:
            # 收集当前状态
            current_state = {
                'time': env.now,
                'hour': int(env.now / 60) % 24,
                'taxis': [taxi.get_status() for taxi in taxis],
                'stations': [station.get_status() for station in stations],
                'total_trips': sum(taxi.trips_completed for taxi in taxis),
                'total_swaps': sum(station.swap_count for station in stations)
            }
            simulation_states.append(current_state)
    
    progress_bar.close()
    
    # 收集结果
    results = collect_results(env, taxis, stations, scheduler, config, block_positions)
    
    # 如果记录了状态，添加到结果中
    if record_states:
        results['simulation_states'] = simulation_states
    
    print("模拟完成。\n结果:")
    print(f"- 完成的行程总数: {results['taxi_stats']['trips_completed']}")
    print(f"- 电池更换总次数: {results['station_stats']['total_swaps']}")
    print(f"- 总收入: ${sum(results['taxi_stats']['revenue_per_taxi'].values()):.2f}")
    print(f"- 总行驶距离: {sum(results['taxi_stats']['distance_per_taxi'].values()):.2f} km")
    
    # 每个站点的详细信息
    print("\n站点详情:")
    for sid, count in results['station_stats']['station_swap_counts'].items():
        print(f"- 站点 {sid}: {count} 次换电, 平均等待 {results['station_stats']['station_avg_wait'][sid]:.2f} 分钟")
    
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
            'filepath': 'data/yellow_tripdata_2025-01.parquet',  # 2025年1月黄色出租车数据
            'use_sample': True,
            'sample_size': 5000  # 样本大小
        },
        'use_real_data': True,  # 使用真实数据分配行程
        'simulation': {
            'duration': 720  # 模拟持续时间(分钟) - 12小时
        }
    }
    
    # 运行演示模拟
    results = run_simulation(demo_config)