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
from datetime import datetime


def prepare_data(config, input_data=None):
    """
    如果传入 input_data，则直接使用；否则从配置里加载并清洗数据。
    返回 (cleaned_data or None, block_positions)。
    """
    try:
        if input_data is not None:
            print(f"使用提供的数据，共 {len(input_data)} 条记录")
            blocks = create_block_positions(input_data)
            return input_data, blocks

        data_cfg = config.get('data', {})
        path = data_cfg.get('filepath', '')
        use_sample = data_cfg.get('use_sample', True)
        sample_size = data_cfg.get('sample_size', None)

        if not path or not os.path.exists(path):
            print(f"数据文件未找到: {path}，将使用测试区块进行模拟。")
            return None, generate_test_blocks(10)

        print(f"正在加载NYC出租车数据: {path}")
        try:
            # 检查文件大小
            file_size = os.path.getsize(path) / (1024 * 1024)  # 转换为MB
            use_dask = False
            
            # 尝试导入Dask（如果安装了的话）
            if file_size > 100:  # 如果文件大于100MB
                try:
                    import dask.dataframe as dd
                    use_dask = True
                    print(f"文件大小为 {file_size:.1f}MB，使用Dask进行数据处理")
                except ImportError:
                    print("未安装Dask，将使用pandas处理数据")
            
            if use_dask:
                raw = dd.read_parquet(path)
                if use_sample and sample_size:
                    # 在Dask中高效地进行随机抽样
                    fraction = sample_size / raw.shape[0].compute()
                    raw = raw.sample(n=fraction)
                # 在计算之前先进行数据清洗
                clean = clean_trip_data(raw)
                # 最后才计算结果
                clean = clean.compute()
            else:
                print("使用pandas直接加载数据")
                raw = pd.read_parquet(path)
                if use_sample and sample_size:
                    raw = raw.sample(n=min(sample_size, len(raw)), random_state=42)
                clean = clean_trip_data(raw)

            print(f"成功加载数据，原始行数: {len(raw) if not use_dask else raw.shape[0].compute()}")
            print(f"数据清洗完成，剩余行数: {len(clean)}")

            # 验证数据的有效性
            if len(clean) == 0:
                print("警告: 清洗后没有剩余数据，将使用测试区块")
                return None, generate_test_blocks(10)

            # 确保必要的列存在
            required_columns = ['block_id']  # 添加其他必需的列
            missing_columns = [col for col in required_columns if col not in clean.columns]
            if missing_columns:
                print(f"警告: 缺少必要的列: {missing_columns}")
                return None, generate_test_blocks(10)

            blocks = create_block_positions(clean)
            return clean, blocks

        except Exception as e:
            print(f"数据加载或清洗错误: {str(e)}")
            print("将使用测试区块进行模拟。")
            return None, generate_test_blocks(10)

    except Exception as e:
        print(f"准备数据时发生错误: {str(e)}")
        # 在出错的情况下也返回一个有效的结果
        return None, generate_test_blocks(10)


def create_block_positions(data):
    """
    根据清洗后的 DataFrame 计算每个 block_id 对应的 (x, y) 坐标。
    如果 DataFrame 中无 block_id，返回测试网格。
    """
    print("\n创建区块位置...")
    
    if data is None:
        print("警告: 数据为None，将使用测试区块。")
        return generate_test_blocks(10)
    
    # 打印数据列以进行调试
    print(f"可用列: {list(data.columns)}")
    
    try:
        # 如果数据中已经有block_id，直接使用
        if 'block_id' in data.columns:
            print("使用已有的block_id")
            unique_blocks = data['block_id'].unique()
        else:
            # 如果没有block_id，但有经纬度，根据经纬度创建区块
            print("根据经纬度创建block_id")
            if 'pickup_latitude' not in data.columns or 'pickup_longitude' not in data.columns:
                print("警告: 缺少经纬度信息，将使用测试区块。")
                return generate_test_blocks(10)
            
            # 使用网格化方法创建区块
            lat_bins = pd.qcut(data['pickup_latitude'], q=10, labels=False)
            lon_bins = pd.qcut(data['pickup_longitude'], q=10, labels=False)
            data['block_id'] = lat_bins * 10 + lon_bins
            unique_blocks = data['block_id'].unique()
        
        print(f"发现 {len(unique_blocks)} 个唯一区块")
        
        # 为每个区块创建位置
        positions = {}
        if 'pickup_latitude' in data.columns and 'pickup_longitude' in data.columns:
            # 使用经纬度的平均值
            coords = data.groupby('block_id').agg({
                'pickup_latitude': 'mean',
                'pickup_longitude': 'mean'
            }).reset_index()
            
            for _, row in coords.iterrows():
                bid = int(row['block_id'])
                # 将经纬度转换为相对坐标
                x = (row['pickup_longitude'] + 74.05) * 100  # 基于纽约的经度范围
                y = (row['pickup_latitude'] - 40.50) * 100   # 基于纽约的纬度范围
                positions[bid] = (x, y)
        else:
            # 如果没有经纬度信息，创建网格布局
            n = int(np.ceil(np.sqrt(len(unique_blocks))))
            for i, bid in enumerate(unique_blocks):
                r = i // n
                c = i % n
                positions[int(bid)] = (c * 10, r * 10)
        
        print(f"成功创建了 {len(positions)} 个区块位置")
        return positions
        
    except Exception as e:
        print(f"创建区块位置时出错: {str(e)}")
        print("将使用测试区块。")
        return generate_test_blocks(10)


def generate_test_blocks(grid_size=10):
    """
    生成一个 grid_size x grid_size 的测试区块网格，Block ID 从 0 到 grid_size^2-1。
    """
    blocks = {}
    for i in range(grid_size):
        for j in range(grid_size):
            bid = i * grid_size + j
            blocks[bid] = (i * 10, j * 10)
    return blocks


def get_trip_demand_points(data):
    """
    从 DataFrame 中统计每个 block_id 的行程数，返回 {block_id: count} 的字典。
    """
    if data is None or 'block_id' not in data.columns:
        print("警告: 无法提取需求点，缺少 block_id 列或数据为 None。")
        return {}
    cnt = data['block_id'].value_counts().to_dict()
    # 确保 key 为 int
    return {int(b): int(v) for b, v in cnt.items()}


def setup_simulation(env, config, trip_data, block_positions):
    """
    在 SimPy 环境中初始化网络、站点、出租车、调度器，并启动对应的进程。
    返回 network, taxis, stations, scheduler, station_schedulers 五元组。
    """
    # 1. 构建城市网络
    network = BlockNetwork(
        block_positions,
        consider_traffic=config.get('simulation', {}).get('consider_traffic', True)
    )

    # 2. 初始化换电站配置
    stations = []
    st_cfg = config.get('stations') or []
    # 如果没有明确 stations 配置，但给了 num_stations，就做优化选址
    if not st_cfg and 'num_stations' in config:
        num = config['num_stations']
        demand = get_trip_demand_points(trip_data)
        locs = block_positions
        print(f"正在为 {num} 个换电站优化位置...")
        sc = config.get('stations', {})
        if isinstance(sc, dict):
            oc = sc.get('optimization', {})
            method = oc.get('method', 'p-median')
            maxd = oc.get('max_distance', None)
        else:
            method, maxd = 'p-median', None
        if method == 'max-coverage' and maxd is not None:
            picks = optimize_bss_layout(demand, locs, num, max_distance=maxd)
        else:
            picks = optimize_bss_layout(demand, locs, num)
        # 统一给优化出的站点一个默认配置
        for bid in picks:
            st_cfg.append({
                'block_id': bid,
                'capacity': 20,
                'initial_batteries': 15,
                'swap_time': 5,
                'charge_time': 60,
                'chargers': 5
            })

    # 根据配置创建 BatterySwapStation 对象
    for i, cfg in enumerate(st_cfg):
        station = BatterySwapStation(
            env=env,
            station_id=i,
            location=cfg['block_id'],
            capacity=cfg.get('capacity', 20),
            initial_batteries=cfg.get('initial_batteries', 15),
            swap_time=cfg.get('swap_time', 5),
            charge_time=cfg.get('charge_time', 60),
            n_chargers=cfg.get('chargers', 5)
        )
        stations.append(station)

    # 3. 为每个换电站创建 StationScheduler，并启动其进程
    station_schedulers = []
    for st in stations:
        ss = StationScheduler(
            station=st,
            check_interval=config.get('station_scheduler', {}).get('check_interval', 10),
            min_charged=st.capacity // 4,
            dynamic_charging=config.get('station_scheduler', {}).get('dynamic_charging', True)
        )
        station_schedulers.append(ss)
        # 假设 StationScheduler 有一个名为 run 或 start 的方法来持续调度充电
        env.process(ss.run())  # 确保名为 run() 的方法在类里实现

    # 4. 初始化出租车列表
    taxis = []
    tc = config.get('taxis', {})
    num_taxis = tc.get('count', 100)

    # 为出租车创建一个内部连续 ID，与真实 taxi_id 保留映射
    if trip_data is not None and 'block_id' in trip_data.columns:
        # 常见热度最高的 20 个区块
        popular_blocks = trip_data['block_id'].value_counts().head(20).index.tolist()

        # 如果数据里存在 taxi_id 且数量足够，就尝试根据真实 taxi_id 初始化
        if 'taxi_id' in trip_data.columns and trip_data['taxi_id'].nunique() >= num_taxis:
            # 取每个 taxi_id 的第一条行程对应的 block 作为起始位置
            loc_map = trip_data.groupby('taxi_id')['block_id'].first().iloc[:num_taxis].to_dict()
            print(f"根据行程数据创建 {len(loc_map)} 辆出租车")
            real_to_sim = {}
            for sim_id, (real_id, bid) in enumerate(loc_map.items()):
                real_to_sim[real_id] = sim_id
                taxi = TaxiAgent(
                    env=env,
                    taxi_id=sim_id,
                    real_id=real_id,
                    initial_block=int(bid),
                    battery_capacity=tc.get('battery_capacity', 100),
                    consumption_rate=tc.get('consumption_rate', 0.5),
                    swap_threshold=tc.get('swap_threshold', 20)
                )
                taxis.append(taxi)
                env.process(taxi.process(network, stations, trip_data))

            extra = num_taxis - len(loc_map)
            if extra > 0:
                print(f"创建额外的 {extra} 辆出租车")
                for i in range(extra):
                    sim_id = len(loc_map) + i
                    bid = popular_blocks[i % len(popular_blocks)]
                    taxi = TaxiAgent(
                        env=env,
                        taxi_id=sim_id,
                        real_id=None,
                        initial_block=int(bid),
                        battery_capacity=tc.get('battery_capacity', 100),
                        consumption_rate=tc.get('consumption_rate', 0.5),
                        swap_threshold=tc.get('swap_threshold', 20)
                    )
                    taxis.append(taxi)
                    env.process(taxi.process(network, stations, trip_data))
        else:
            # 数据里没有足够的 taxi_id，就随机或按设定 initial_locations 分配
            blocks_list = popular_blocks if popular_blocks else list(block_positions.keys())
            for i in range(num_taxis):
                bid = tc.get('initial_locations', blocks_list)[i % len(blocks_list)]
                taxi = TaxiAgent(
                    env=env,
                    taxi_id=i,
                    real_id=None,
                    initial_block=int(bid),
                    battery_capacity=tc.get('battery_capacity', 100),
                    consumption_rate=tc.get('consumption_rate', 0.5),
                    swap_threshold=tc.get('swap_threshold', 20)
                )
                taxis.append(taxi)
                env.process(taxi.process(network, stations, trip_data))
    else:
        # 没有 trip_data，直接随机或按初始 locations 创建出租车
        blocks_list = list(block_positions.keys())
        for i in range(num_taxis):
            bid = tc.get('initial_locations', blocks_list)[i % len(blocks_list)]
            taxi = TaxiAgent(
                env=env,
                taxi_id=i,
                real_id=None,
                initial_block=int(bid),
                battery_capacity=tc.get('battery_capacity', 100),
                consumption_rate=tc.get('consumption_rate', 0.5),
                swap_threshold=tc.get('swap_threshold', 20)
            )
            taxis.append(taxi)
            env.process(taxi.process(network, stations, trip_data))    # 5. 创建调度器实例
    scheduler = ETaxiScheduler(
        env=env,
        taxis=taxis,
        stations=stations,
        network=network,
        check_interval=config.get('scheduler', {}).get('check_interval', 5),
        use_real_data=config.get('use_real_data', False) and (trip_data is not None),
        trip_data=trip_data
    )
    # 假设 ETaxiScheduler 需要显式在环境里注册它的 run 循环
    env.process(scheduler.run_loop())

    # 为每个换电站设置调度器引用
    for station in stations:
        station.scheduler = scheduler

    return network, taxis, stations, scheduler, station_schedulers


def collect_results(env, taxis, stations, scheduler, config, blocks):
    """
    在仿真结束后收集所有统计数据，并根据 output_dir 保存 CSV 文件。
    """
    # 先确保所有 TaxiAgent 和 BatterySwapStation 的统计字段都存在（应在它们的 __init__ 中已初始化）
    station_stats = {
        'total_swaps': sum(getattr(s, 'swap_count', 0) for s in stations),
        'station_swap_counts': {s.id: getattr(s, 'swap_count', 0) for s in stations},
        'station_avg_wait': {
            s.id: (sum(getattr(s, 'wait_times', [])) / len(s.wait_times) if getattr(s, 'wait_times', []) else 0)
            for s in stations
        },
        'station_locations': {s.id: s.location for s in stations},
        'charged_batteries': {s.id: s.charged_batteries.level for s in stations},
        'empty_batteries': {s.id: s.empty_batteries.level for s in stations}
    }

    taxi_stats = {
        'trips_completed': sum(getattr(t, 'trips_completed', 0) for t in taxis),
        'trips_per_taxi': {t.id: getattr(t, 'trips_completed', 0) for t in taxis},
        'swap_count': {t.id: getattr(t, 'swap_count', 0) for t in taxis},
        'total_revenue': sum(getattr(t, 'total_revenue', 0.0) for t in taxis),
        'revenue_per_taxi': {t.id: getattr(t, 'total_revenue', 0.0) for t in taxis},
        'total_distance': sum(getattr(t, 'total_distance', 0.0) for t in taxis),
        'distance_per_taxi': {t.id: getattr(t, 'total_distance', 0.0) for t in taxis},
        'final_locations': {t.id: t.location for t in taxis},
        'final_battery_levels': {t.id: t.battery_level for t in taxis}
    }

    perf = {
        'avg_revenue_per_km': (taxi_stats['total_revenue'] / taxi_stats['total_distance']
                               if taxi_stats['total_distance'] > 0 else 0),
        'avg_trips_per_taxi': (taxi_stats['trips_completed'] / len(taxis)
                               if taxis else 0),
        'avg_swaps_per_taxi': (station_stats['total_swaps'] / len(taxis)
                               if taxis else 0),
        'avg_distance_per_trip': (taxi_stats['total_distance'] / taxi_stats['trips_completed']
                                  if taxi_stats['trips_completed'] > 0 else 0),
        'avg_revenue_per_trip': (taxi_stats['total_revenue'] / taxi_stats['trips_completed']
                                 if taxi_stats['trips_completed'] > 0 else 0),
        'simulation_duration': env.now
    }

    results = {
        'station_stats': station_stats,
        'taxi_stats': taxi_stats,
        'performance_metrics': perf,
        'scheduler_stats': scheduler.get_status() if hasattr(scheduler, 'get_status') else {},
        'blocks': blocks
    }

    od = config.get('output_dir')
    if od:
        # 确保输出目录存在
        if not os.path.isdir(od):
            os.makedirs(od, exist_ok=True)

        # 保存出租车详情
        taxi_details = []
        for t in taxis:
            taxi_details.append({
                'id': t.id,
                'real_id': getattr(t, 'real_id', None),
                'trips_completed': getattr(t, 'trips_completed', 0),
                'swap_count': getattr(t, 'swap_count', 0),
                'revenue': getattr(t, 'total_revenue', 0.0),
                'distance': getattr(t, 'total_distance', 0.0),
                'final_location': t.location,
                'battery_level': t.battery_level,
                'idle_time': getattr(t, 'idle_time', 0.0),
                'service_time': getattr(t, 'service_time', 0.0),
                'charging_time': getattr(t, 'charging_time', 0.0),
                'waiting_time': getattr(t, 'waiting_time', 0.0)
            })
        pd.DataFrame(taxi_details).to_csv(os.path.join(od, 'taxi_details.csv'), index=False)

        # 保存站点详情
        station_details = []
        for s in stations:
            station_details.append({
                'id': s.id,
                'location': s.location,
                'swap_count': getattr(s, 'swap_count', 0),
                'charged_batteries': s.charged_batteries.level,
                'empty_batteries': s.empty_batteries.level,
                'avg_wait_time': (sum(getattr(s, 'wait_times', [])) / len(s.wait_times)
                                  if getattr(s, 'wait_times', []) else 0),
                'avg_queue_length': (sum(getattr(s, 'queue_lengths', [])) / len(s.queue_lengths)
                                     if getattr(s, 'queue_lengths', []) else 0),
                'utilization': (sum(getattr(s, 'utilization_samples', [])) / len(s.utilization_samples)
                                if getattr(s, 'utilization_samples', []) else 0)
            })
        pd.DataFrame(station_details).to_csv(os.path.join(od, 'station_details.csv'), index=False)

    return results


def run_simulation(config, trip_data=None):
    """
    运行电动出租车电池交换系统模拟。
    """
    print("\n=== 开始初始化模拟环境 ===\n")
    try:
        print("\n=== 初始化模拟环境 ===")
        # 创建模拟环境
        env = simpy.Environment()
        
        print("\n=== 准备数据 ===")
        # 准备数据
        trip_data, blocks = prepare_data(config, trip_data)
        if trip_data is not None:
            print(f"成功加载数据: {len(trip_data)} 条记录")
        else:
            print("使用测试数据进行模拟")
            
        print("\n=== 创建区块网络 ===")
        # 创建区块网络
        network = BlockNetwork(blocks)
        print(f"网络创建完成, 共 {len(network.nodes)} 个区块")
        
        print("\n=== 初始化电池交换站 ===")
        # 创建电池交换站
        stations = [
            BatterySwapStation(
                env=env,
                station_id=i,
                location=location,
                capacity=10,
                swap_time=5,
                charge_time=30
            )
            for i, location in enumerate(config['stations'] or range(config['num_stations']))
        ]
        print(f"创建了 {len(stations)} 个换电站")
        
        print("\n=== 初始化出租车 ===")
        # 创建出租车代理
        taxis = [
            TaxiAgent(
                env=env,
                taxi_id=i,
                battery_capacity=config['taxis']['battery_capacity'],
                consumption_rate=config['taxis']['consumption_rate'],
                swap_threshold=config['taxis']['swap_threshold'],
                network=network,
                initial_location=random.choice(network.nodes)
            )
            for i in range(config['taxis']['count'])
        ]
        print(f"创建了 {len(taxis)} 辆出租车")
        
        print("\n=== 初始化调度器 ===")
        # 创建调度器
        scheduler = ETaxiScheduler(
            env=env,
            taxis=taxi,
            stations=stations,
            network=network,
            use_real_data=config['use_real_data'],
            trip_data=trip_data
        )
        print("调度器初始化完成")
        
        # 为每个换电站设置调度器引用
        for station in stations:
            station.scheduler = scheduler
        
        print("\n=== 开始运行模拟 ===")
        # 运行模拟
        duration = config['simulation']['duration']
        env.run(until=duration)
        print(f"\n模拟完成，运行时长: {duration} 分钟")
        
        print("\n=== 收集结果 ===")
        # 收集结果
        results = {
            'simulation_time': duration,
            'num_taxis': len(taxis),
            'num_stations': len(stations),
            'taxi_metrics': [taxi.get_metrics() for taxi in taxis],
            'station_metrics': [station.get_metrics() for station in stations],
            'scheduler_metrics': scheduler.get_status()
        }
        print("结果收集完成")
        
        return results
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        print("\n详细错误信息:")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    demo_config = {
        # blocks 由 prepare_data 自动生成或覆盖
        'num_stations': 5,
        'taxis': {
            'count': 50,
            'battery_capacity': 100,
            'consumption_rate': 0.5,
            'swap_threshold': 20
        },
        'data': {
            'filepath': 'data/yellow_tripdata_2025-01.parquet',
            'use_sample': True,
            'sample_size': 5000
        },
        'use_real_data': True,
        'simulation': {
            'start_hour': 0,
            'duration': 60
        },
        'output': {
            'state_interval': 5,
            'save_states': False
        },
        'output_dir': './results'
    }
    run_simulation(demo_config)

