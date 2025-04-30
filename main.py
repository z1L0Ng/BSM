"""
电动出租车电池交换系统模拟主程序
=================================

该程序模拟基于电池交换的电动出租车运营系统，包括：
- 从NYC出租车数据中加载真实行程
- 构建城市区块网络和行驶距离/时间矩阵
- 优化电池交换站的位置
- 模拟出租车运营、电池更换和充电管理
- 分析和可视化系统性能

作者: [您的名字]
日期: [当前日期]
"""

import os
import sys
import json
import argparse
import pandas as pd
from simulation.simulation import run_simulation
from utils.visualization import (
    plot_station_metrics, 
    plot_taxi_metrics, 
    create_nyc_map
)

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
        'num_stations': 10,  # 要放置的站点数量
        'stations': [],  # 将由优化器填充
        
        # 出租车配置
        'taxis': {
            'count': 100,  # 出租车数量
            'battery_capacity': 100,  # 电池容量(kWh)
            'consumption_rate': 0.5,  # 消耗率(kWh/分钟)
            'swap_threshold': 20  # 换电阈值
        },
        
        # 数据配置
        'data': {
            'filepath': 'data/nyc_taxi_sample.parquet',  # 数据文件路径
            'use_sample': True,
            'sample_size': 10000  # 样本大小
        },
        
        # 行程调度
        'use_real_data': True,  # 是否使用真实数据分配行程
        
        # 模拟参数
        'simulation': {
            'duration': 1440  # 模拟时长(分钟) - 24小时
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
    parser.add_argument('--taxis', type=int, help='出租车数量')
    parser.add_argument('--stations', type=int, help='换电站数量')
    parser.add_argument('--output', type=str, help='结果输出目录')
    parser.add_argument('--sample', type=int, help='使用的数据样本大小')
    parser.add_argument('--real-data', action='store_true', help='使用真实数据进行行程调度')
    
    return parser.parse_args()

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
    if args.taxis:
        config['taxis']['count'] = args.taxis
    if args.stations:
        config['num_stations'] = args.stations
    if args.sample:
        config['data']['sample_size'] = args.sample
        config['data']['use_sample'] = True
    if args.real_data:
        config['use_real_data'] = True
    
    # 创建输出目录
    output_dir = args.output if args.output else 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 显示当前配置
    print("\n当前配置:")
    print(f"- 数据文件: {config['data']['filepath']}")
    print(f"- 模拟时长: {config['simulation']['duration']}分钟")
    print(f"- 出租车数量: {config['taxis']['count']}")
    print(f"- 换电站数量: {config['num_stations']}")
    print(f"- 使用样本: {'是' if config['data']['use_sample'] else '否'}, " + 
          f"样本大小: {config['data']['sample_size'] if config['data']['use_sample'] else 'N/A'}")
    print(f"- 使用真实数据调度: {'是' if config.get('use_real_data', False) else '否'}")
    print(f"- 输出目录: {output_dir}")
    
    # 运行模拟
    try:
        print("\n开始运行模拟...")
        results = run_simulation(config)
        
        # 保存结果到JSON文件
        results_file = os.path.join(output_dir, 'simulation_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
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
        if 'blocks' in config and config['blocks']:
            map_file = os.path.join(output_dir, 'nyc_map.html')
            nyc_map = create_nyc_map(results, config['blocks'])
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