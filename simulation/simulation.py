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
    if input_data is not None:
        print(f"使用提供的数据，共 {len(input_data)} 条记录")
        return input_data, create_block_positions(input_data)

    data_cfg = config.get('data', {})
    path = data_cfg.get('filepath', '')
    use_sample = data_cfg.get('use_sample', True)
    sample_size = data_cfg.get('sample_size', None)

    print(f"正在加载NYC出租车数据: {path}")
    try:
        raw = load_trip_data(path)
        if use_sample and sample_size:
            raw = raw.sample(sample_size, random_state=42)
        clean = clean_trip_data(raw)
        print(f"Successfully loaded data, total rows: {len(raw)}")
        print(f"Data cleaning complete, remaining rows: {len(clean)}")
    except Exception as e:
        print(f"数据加载错误: {e}")
        print("将使用测试区块进行模拟。")
        return None, generate_test_blocks(10)

    blocks = create_block_positions(clean)
    return clean, blocks


def create_block_positions(data):
    if 'block_id' not in data.columns:
        print("警告: 数据中没有 block_id 列，将使用测试区块。")
        return generate_test_blocks(10)

    if 'pickup_latitude' in data.columns and 'pickup_longitude' in data.columns:
        coords = data.groupby('block_id').agg({
            'pickup_latitude': 'mean',
            'pickup_longitude': 'mean'
        }).reset_index()
        positions = {}
        for _, row in coords.iterrows():
            bid = int(row['block_id'])
            x = (row['pickup_longitude'] + 74.05) * 100
            y = (row['pickup_latitude'] - 40.5) * 100
            positions[bid] = (x, y)
    else:
        unique = data['block_id'].unique()
        n = int(np.ceil(np.sqrt(len(unique))))
        positions = {}
        for i, bid in enumerate(unique):
            r = i // n
            c = i % n
            positions[int(bid)] = (c * 10, r * 10)
    print(f"创建了 {len(positions)} 个区块位置")
    return positions


def generate_test_blocks(grid_size=10):
    blocks = {}
    for i in range(grid_size):
        for j in range(grid_size):
            bid = i * grid_size + j
            blocks[bid] = (i * 10, j * 10)
    return blocks


def get_trip_demand_points(data):
    if 'block_id' not in data.columns:
        print("警告: 无法提取需求点，缺少 block_id 列。")
        return {}
    cnt = data['block_id'].value_counts().to_dict()
    return {int(b): v for b, v in cnt.items()}


def setup_simulation(env, config, trip_data, block_positions):
    network = BlockNetwork(
        block_positions,
        consider_traffic=config['simulation'].get('consider_traffic', True)
    )

    stations = []
    st_cfg = config.get('stations') or []
    if not st_cfg and 'num_stations' in config:
        num = config['num_stations']
        demand = get_trip_demand_points(trip_data) if trip_data is not None else {}
        locs = block_positions
        print(f"正在为 {num} 个换电站优化位置...")
        sc = config.get('stations', {})
        if isinstance(sc, dict):
            oc = sc.get('optimization', {})
            method = oc.get('method', 'p-median')
            maxd = oc.get('max_distance')
        else:
            method, maxd = 'p-median', None
        if method == 'max-coverage' and maxd is not None:
            picks = optimize_bss_layout(demand, locs, num, max_distance=maxd)
        else:
            picks = optimize_bss_layout(demand, locs, num)
        for bid in picks:
            st_cfg.append({
                'block_id': bid,
                'capacity': 20,
                'initial_batteries': 15,
                'swap_time': 5,
                'charge_time': 60,
                'chargers': 5
            })

    for i, cfg in enumerate(st_cfg):
        station = BatterySwapStation(
            env,
            station_id=i,
            location=cfg['block_id'],
            capacity=cfg.get('capacity', 20),
            initial_batteries=cfg.get('initial_batteries', 15),
            swap_time=cfg.get('swap_time', 5),
            charge_time=cfg.get('charge_time', 60),
            n_chargers=cfg.get('chargers', 5)
        )
        stations.append(station)

    station_schedulers = [
        StationScheduler(
            station=st,
            check_interval=10,
            min_charged=st.capacity // 4,
            dynamic_charging=True
        ) for st in stations
    ]

    # 出租车初始化
    taxis = []
    tc = config.get('taxis', {})
    num_taxis = tc.get('count', 100)
    if trip_data is not None and 'block_id' in trip_data.columns:
        popular_blocks = trip_data['block_id'].value_counts().head(20).index.tolist()
        if 'taxi_id' in trip_data.columns and trip_data['taxi_id'].nunique() >= num_taxis:
            loc_map = trip_data.groupby('taxi_id')['block_id'].first().iloc[:num_taxis].to_dict()
            print(f"根据行程数据创建 {len(loc_map)} 辆出租车")
            for idx, (tid, bid) in enumerate(loc_map.items()):
                taxi = TaxiAgent(
                    env=env,
                    taxi_id=int(tid),
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
                    bid = popular_blocks[i % len(popular_blocks)]
                    taxi = TaxiAgent(
                        env=env,
                        taxi_id=i + len(loc_map),
                        initial_block=int(bid),
                        battery_capacity=tc.get('battery_capacity', 100),
                        consumption_rate=tc.get('consumption_rate', 0.5),
                        swap_threshold=tc.get('swap_threshold', 20)
                    )
                    taxis.append(taxi)
                    env.process(taxi.process(network, stations, trip_data))
        else:
            for i in range(num_taxis):
                bid = tc.get('initial_locations', popular_blocks)[i % len(popular_blocks)]
                taxi = TaxiAgent(
                    env=env,
                    taxi_id=i,
                    initial_block=int(bid),
                    battery_capacity=tc.get('battery_capacity', 100),
                    consumption_rate=tc.get('consumption_rate', 0.5),
                    swap_threshold=tc.get('swap_threshold', 20)
                )
                taxis.append(taxi)
                env.process(taxi.process(network, stations, trip_data))
    else:
        blocks_list = list(block_positions.keys())
        for i in range(num_taxis):
            bid = tc.get('initial_locations', blocks_list)[i % len(blocks_list)]
            taxi = TaxiAgent(env=env, taxi_id=i, initial_block=int(bid),
                             battery_capacity=tc.get('battery_capacity', 100),
                             consumption_rate=tc.get('consumption_rate', 0.5),
                             swap_threshold=tc.get('swap_threshold', 20))
            taxis.append(taxi)
            env.process(taxi.process(network, stations, trip_data))

    scheduler = ETaxiScheduler(
        env=env,
        taxis=taxis,
        stations=stations,
        network=network,
        check_interval=5,
        use_real_data=config.get('use_real_data', False) and trip_data is not None,
        trip_data=trip_data
    )

    return network, taxis, stations, scheduler, station_schedulers


def collect_results(env, taxis, stations, scheduler, config, blocks):
    station_stats = {
        'total_swaps': sum(s.swap_count for s in stations),
        'station_swap_counts': {s.id: s.swap_count for s in stations},
        'station_avg_wait': {s.id: (sum(s.wait_times)/len(s.wait_times) if s.wait_times else 0) for s in stations},
        'station_locations': {s.id: s.location for s in stations},
        'charged_batteries': {s.id: s.charged_batteries.level for s in stations},
        'empty_batteries': {s.id: s.empty_batteries.level for s in stations}
    }
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
    perf = {
        'avg_revenue_per_km': taxi_stats['total_revenue'] / taxi_stats['total_distance'] if taxi_stats['total_distance'] > 0 else 0,
        'avg_trips_per_taxi': taxi_stats['trips_completed'] / len(taxis) if taxis else 0,
        'avg_swaps_per_taxi': station_stats['total_swaps'] / len(taxis) if taxis else 0,
        'avg_distance_per_trip': taxi_stats['total_distance'] / taxi_stats['trips_completed'] if taxi_stats['trips_completed'] > 0 else 0,
        'avg_revenue_per_trip': taxi_stats['total_revenue'] / taxi_stats['trips_completed'] if taxi_stats['trips_completed'] > 0 else 0,
        'simulation_duration': env.now
    }
    results = {
        'station_stats': station_stats,
        'taxi_stats': taxi_stats,
        'performance_metrics': perf,
        'scheduler_stats': scheduler.get_status(),
        'blocks': blocks
    }
    od = config.get('output_dir')
    if od:
        # 保存出租车详情
        taxi_details = []
        for t in taxis:
            taxi_details.append({
                'id': t.id,
                'trips_completed': t.trips_completed,
                'swap_count': t.swap_count,
                'revenue': t.total_revenue,
                'distance': t.total_distance,
                'final_location': t.location,
                'battery_level': t.battery_level,
                'idle_time': t.idle_time,
                'service_time': t.service_time,
                'charging_time': t.charging_time,
                'waiting_time': t.waiting_time
            })
        pd.DataFrame(taxi_details).to_csv(os.path.join(od, 'taxi_details.csv'), index=False)
        # 保存站点详情
        station_details = []
        for s in stations:
            station_details.append({
                'id': s.id,
                'location': s.location,
                'swap_count': s.swap_count,
                'charged_batteries': s.charged_batteries.level,
                'empty_batteries': s.empty_batteries.level,
                'avg_wait_time': sum(s.wait_times)/len(s.wait_times) if s.wait_times else 0,
                'avg_queue_length': sum(s.queue_lengths)/len(s.queue_lengths) if s.queue_lengths else 0,
                'utilization': sum(s.utilization_samples)/len(s.utilization_samples) if s.utilization_samples else 0
            })
        pd.DataFrame(station_details).to_csv(os.path.join(od, 'station_details.csv'), index=False)
    return results


def run_simulation(config, input_trip_data=None):
    trip_data, blocks = prepare_data(config, input_trip_data)
    config['blocks'] = blocks
    env = simpy.Environment()
    start = config.get('simulation', {}).get('start_hour', 0)
    # 跳转到指定的起始小时
    env.run(until=start * 60)

    net, taxis, stations, scheduler, station_schedulers = setup_simulation(
        env, config, trip_data, blocks
    )

    duration = config.get('simulation', {}).get('duration', 1440)
    print(f"开始运行{duration}分钟的模拟，从小时 {start} 开始...")

    progress_bar = tqdm(total=duration, desc="模拟进度")
    step = config.get('output', {}).get('state_interval', 5)
    record_states = config.get('output', {}).get('save_states', False)
    simulation_states = [] if record_states else None

    elapsed = 0
    while elapsed < duration:
        delta = min(step, duration - elapsed)
        next_time = env.now + delta
        env.run(until=next_time)
        progress_bar.update(delta)
        elapsed += delta
        if record_states:
            simulation_states.append({
                'time': env.now,
                'hour': int(env.now / 60) % 24,
                'taxis': [tx.get_status() for tx in taxis],
                'stations': [st.get_status() for st in stations],
                'total_trips': sum(tx.trips_completed for tx in taxis),
                'total_swaps': sum(st.swap_count for st in stations)
            })

    progress_bar.close()

    results = collect_results(env, taxis, stations, scheduler, config, blocks)
    if record_states:
        results['simulation_states'] = simulation_states

    print("模拟完成。结果摘要:")
    print(f"- 完成的行程总数: {results['taxi_stats']['trips_completed']}")
    print(f"- 总换电次数: {results['station_stats']['total_swaps']}")
    print(f"- 总收入: ${sum(results['taxi_stats']['revenue_per_taxi'].values()):.2f}")
    print(f"- 总行驶距离: {sum(results['taxi_stats']['distance_per_taxi'].values()):.2f} km")
    for sid, cnt in results['station_stats']['station_swap_counts'].items():
        print(f"站点 {sid}: {cnt} 次换电, 平均等待 {results['station_stats']['station_avg_wait'][sid]:.2f} 分钟")

    return results


if __name__ == "__main__":
    demo_config = {
        'blocks': None,
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
            'duration': 60
        }
    }
    run_simulation(demo_config)
