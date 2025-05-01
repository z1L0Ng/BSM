"""
电动出租车电池交换系统模拟主程序
=================================

该程序模拟基于电池交换的电动出租车运营系统，包括：
- 从NYC出租车数据中加载真实行程
- 构建城市区块网络和行驶距离/时间矩阵
- 优化电池交换站的位置
- 模拟出租车运营、电池更换和充电管理
- 分析和可视化系统性能
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import random
from datetime import datetime
from simulation.simulation import run_simulation
from utils.visualization import (
    plot_station_metrics, 
    plot_taxi_metrics, 
    create_nyc_map
)
from dataprocess.loaddata import load_trip_data, clean_trip_data

def create_default_config():
    """
    创建默认模拟配置。
    
    返回:
        dict: 包含默认设置的配置字典
    """
    config = {
        # 区块网络
        'blocks': None,  # 将从数据推断或生成
        
        # 站点配置
        'num_stations': 20,  # 要放置的站点数量
        'stations': [],  # 将由优化器填充
        
        # 出租车配置 - 将从数据中确定实际数量
        'taxis': {
            'count': None,  # 将从数据确定
            'battery_capacity': 100,  # 电池容量(kWh)
            'consumption_rate': 0.5,  # 消耗率(kWh/分钟)
            'swap_threshold': 20  # 换电阈值
        },
        
        # 数据配置
        'data': {
            'filepath': 'data/yellow_tripdata_2025-01.parquet',  # 2025年1月黄色出租车数据
            'use_sample': False,  # 使用所有数据
            'sample_size': None  # 不限制样本大小
        },
        
        # 行程调度
        'use_real_data': True,  # 使用真实数据分配行程
        
        # 模拟参数
        'simulation': {
            'duration': 1440,  # 模拟时长(分钟) - 24小时
            'start_hour': 8,    # 从早上8点开始模拟
            'random_seed': 42,  # 随机种子，确保可重复性
            'consider_traffic': True  # 考虑交通状况
        }
    }
    
    return config

def parse_arguments():
    """
    解析命令行参数。
    
    返回:
        argparse.Namespace: 解析后的参数对象
    """
    parser = argparse.ArgumentParser(description='电动出租车电池交换系统模拟')
    
    parser.add_argument('--config', type=str, help='配置文件路径(JSON格式)')
    parser.add_argument('--data', type=str, help='NYC出租车数据文件路径')
    parser.add_argument('--duration', type=int, help='模拟时长(分钟)')
    parser.add_argument('--stations', type=int, help='换电站数量')
    parser.add_argument('--output', type=str, help='结果输出目录')
    parser.add_argument('--use-real-data', action='store_true', help='使用真实数据进行行程调度')
    parser.add_argument('--seed', type=int, help='随机种子')
    
    return parser.parse_args()

def determine_taxi_count(trip_data, hour_range=3):
    """
    根据行程数据确定合适的出租车数量。
    
    参数:
        trip_data (DataFrame): 出租车行程数据
        hour_range (int): 用于确定出租车数量的小时范围
    
    返回:
        int: 推荐的出租车数量
    """
    try:
        # 检查数据是否包含时间列
        if 'tpep_pickup_datetime' in trip_data.columns:
            time_col = 'tpep_pickup_datetime'
        elif 'lpep_pickup_datetime' in trip_data.columns:
            time_col = 'lpep_pickup_datetime'
        elif 'pickup_datetime' in trip_data.columns:
            time_col = 'pickup_datetime'
        elif 'hour' in trip_data.columns:
            # 如果已经有小时列，则使用它
            busiest_hour = trip_data.groupby('hour').size().idxmax()
            busiest_trips = trip_data[trip_data['hour'] == busiest_hour]
            # 使用唯一的出租车ID数量，如果没有，则使用行程数的1/3作为估计
            if 'taxi_id' in trip_data.columns:
                taxi_count = busiest_trips['taxi_id'].nunique()
            else:
                taxi_count = len(busiest_trips) // 3
            
            return max(100, min(1000, taxi_count))  # 确保数量在合理范围内
        else:
            # 如果没有时间列，使用默认值
            return 500
        
        # 将时间列转换为datetime
        trip_data[time_col] = pd.to_datetime(trip_data[time_col])
        
        # 提取小时
        trip_data['hour'] = trip_data[time_col].dt.hour
        
        # 找出最繁忙的小时
        hourly_counts = trip_data.groupby('hour').size()
        busiest_hour = hourly_counts.idxmax()
        
        # 获取最繁忙时段的行程
        busy_hours = [
            (busiest_hour + i) % 24 for i in range(hour_range)
        ]
        busiest_trips = trip_data[trip_data['hour'].isin(busy_hours)]
        
        # 估计需要的出租车数量：使用行程数除以每辆车每小时能完成的行程数估计
        trips_per_taxi_per_hour = 2  # 假设每辆车每小时完成2次行程
        estimated_taxis = len(busiest_trips) // (trips_per_taxi_per_hour * hour_range)
        
        # 确保数量在合理范围内
        taxi_count = max(100, min(2000, estimated_taxis))
        
        print(f"基于数据估计的出租车数量: {taxi_count} (最繁忙时段: {busiest_hour}时)")
        return taxi_count
        
    except Exception as e:
        print(f"估计出租车数量时出错: {e}")
        print("使用默认出租车数量: 500")
        return 500

def main():
    """主程序入口点"""
    print("="*80)
    print("电动出租车电池交换系统模拟")
    print("="*80)
    
    # 解析命令行参数
    args = parse_arguments()
    
    # 获取基础配置
    if args.config and os.path.exists(args.config):
        # 从文件加载配置
        with open(args.config, 'r') as f:
            config = json.load(f)
        print(f"从文件加载配置: {args.config}")
    else:
        # 使用默认配置
        config = create_default_config()
        print("使用默认配置")
    
    # 更新配置中的命令行参数
    if args.data:
        config['data']['filepath'] = args.data
    if args.duration:
        config['simulation']['duration'] = args.duration
    if args.stations:
        config['num_stations'] = args.stations
    if args.use_real_data:
        config['use_real_data'] = True
    if args.seed:
        config['simulation']['random_seed'] = args.seed
    
    # 固定随机种子，确保可重复性
    random_seed = config['simulation']['random_seed']
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # 创建输出目录
    output_dir = args.output if args.output else 'results'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载行程数据
    data_config = config.get('data', {})
    data_path = data_config.get('filepath', '')
    
    print(f"\n正在加载NYC出租车数据: {data_path}")
    if not os.path.exists(data_path):
        print(f"错误: 数据文件不存在: {data_path}")
        print("请确保数据文件位于正确位置并重试。")
        sys.exit(1)
    
    try:
        # 加载数据
        raw_data = load_trip_data(data_path)
        
        # 清洗数据
        print("正在处理出租车数据...")
        trip_data = clean_trip_data(raw_data)
        print(f"数据处理完成。共有 {len(trip_data)} 条行程记录。")
        
        # 确定出租车数量，基于数据
        if config['taxis']['count'] is None:
            config['taxis']['count'] = determine_taxi_count(trip_data)
        
        # 更新配置中的区块位置 - 将在run_simulation中处理
        
        # 保存配置到输出目录
        config_file = os.path.join(output_dir, 'config.json')
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        # 显示当前配置
        print("\n当前配置:")
        print(f"- 数据文件: {config['data']['filepath']}")
        print(f"- 模拟时长: {config['simulation']['duration']}分钟")
        print(f"- 出租车数量: {config['taxis']['count']}")
        print(f"- 换电站数量: {config['num_stations']}")
        print(f"- 使用真实数据调度: {'是' if config.get('use_real_data', False) else '否'}")
        print(f"- 输出目录: {output_dir}")
        
        # 运行模拟
        print("\n开始运行模拟...")
        config['output_dir'] = output_dir  # 将输出目录添加到配置中
        results = run_simulation(config, trip_data)
        
        # 保存结果到JSON文件
        results_file = os.path.join(output_dir, 'simulation_results.json')
        with open(results_file, 'w') as f:
            # 转换不可序列化的对象，如numpy数组
            json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float16, np.float32, np.float64)) else str(x) if isinstance(x, (np.int64, pd.Timestamp)) else x)
        print(f"\n结果已保存到: {results_file}")
        
        # 创建可视化
        print("\n生成可视化结果...")
        
        # 1. 站点指标图表
        station_metrics_file = os.path.join(output_dir, 'station_metrics.png')
        plot_station_metrics(results, save_path=station_metrics_file)
        print(f"站点指标图表已保存到: {station_metrics_file}")
        
        # 2. 出租车指标图表
        taxi_metrics_file = os.path.join(output_dir, 'taxi_metrics.png')
        plot_taxi_metrics(results, save_path=taxi_metrics_file)
        print(f"出租车指标图表已保存到: {taxi_metrics_file}")
        
        # 3. NYC地图(如果有区块位置)
        if 'blocks' in results:
            map_file = os.path.join(output_dir, 'nyc_map.html')
            nyc_map = create_nyc_map(results, results['blocks'])
            nyc_map.save(map_file)
            print(f"NYC地图可视化已保存到: {map_file}")
        
        print("\n模拟完成!")
        print("="*80)
        
    except Exception as e:
        print(f"\n模拟过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()