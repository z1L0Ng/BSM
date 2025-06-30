"""
电动出租车电池交换系统模拟主程序 (基于论文实现)
====================================================

该程序实现论文《E-taxi Fleet Formulation》中的数学模型，包括：
- 基于NYC出租车数据的真实需求建模
- 时空状态转移模型 (公式1-7)
- 联合优化模型 (公式11-12)
- 充电任务生成 (第III-D节)
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import random
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional

# 导入新的模块
from dataprocess.loaddata import load_trip_data, clean_trip_data, prepare_simulation_data
from dataprocess.distance import create_all_matrices
from models.state_transition import ETaxiStateModel, BSSStateModel
from models.optimization_model import JointOptimizationModel
from optimization.joint_optimizer import JointOptimizer, OptimizationConfig
from optimization.charge_scheduler import ChargeTaskGenerator
from optimization.interface import optimize_bss_layout
from utils.visualization import (
    plot_station_metrics, 
    plot_taxi_metrics, 
    create_nyc_map
)
from utils.analysis import analyze_performance, plot_optimization_results

@dataclass
class SimulationConfig:
    """模拟配置类。"""
    # 基本参数
    m_areas: int = 20
    L_energy_levels: int = 10
    T_periods: int = 54  # 24小时，每15分钟一个时间段
    period_length_minutes: int = 20
    
    # 数据配置
    data_filepath: str = 'data/yellow_tripdata_2025-01.parquet'
    use_sample: bool = True
    sample_size: int = 10000
    
    # 出租车配置
    num_taxis: int = 500
    battery_capacity: float = 100.0
    consumption_rate: float = 1.0  # kWh per energy level
    swap_threshold: float = 20.0
    
    # 换电站配置
    num_stations: int = 10
    station_capacity: int = 30
    station_chargers: int = 8
    
    # 优化配置
    beta: float = -0.1  # 空驶距离权重
    solver_method: str = 'auto'
    time_limit: int = 300
    mip_gap: float = 0.05
    
    # 模拟配置
    simulation_duration: int = 1080  # 6小时
    start_hour: int = 6
    random_seed: int = 42
    
    # 输出配置
    output_dir: str = 'results'
    save_detailed_results: bool = True

def create_default_config() -> SimulationConfig:
    """创建默认配置。"""
    return SimulationConfig()

def parse_arguments():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description='电动出租车电池交换系统模拟 (基于论文实现)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 基本参数
    parser.add_argument('--config', type=str, help='配置文件路径(JSON格式)')
    parser.add_argument('--data', type=str, help='NYC出租车数据文件路径')
    parser.add_argument('--duration', type=int, help='模拟时长(分钟)')
    parser.add_argument('--areas', type=int, help='城市区域数量')
    parser.add_argument('--energy-levels', type=int, help='电池能量等级数量')
    parser.add_argument('--periods', type=int, help='时间段数量')
    
    # 车辆和站点
    parser.add_argument('--taxis', type=int, help='出租车数量')
    parser.add_argument('--stations', type=int, help='换电站数量')
    
    # 优化参数
    parser.add_argument('--solver', choices=['gurobi', 'heuristic', 'auto'], 
                       help='优化求解器')
    parser.add_argument('--beta', type=float, help='空驶距离权重')
    
    # 其他
    parser.add_argument('--output', type=str, help='结果输出目录')
    parser.add_argument('--seed', type=int, help='随机种子')
    parser.add_argument('--verbose', action='store_true', help='详细输出')
    
    return parser.parse_args()

def load_and_validate_config(args) -> SimulationConfig:
    """加载和验证配置。"""
    # 从默认配置开始
    config = create_default_config()
    
    # 从配置文件加载
    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            # 更新配置
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
            
            print(f"从配置文件加载: {args.config}")
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            sys.exit(1)
    
    # 从命令行参数更新
    if args.data:
        config.data_filepath = args.data
    if args.duration:
        config.simulation_duration = args.duration
    if args.areas:
        config.m_areas = args.areas
    if args.energy_levels:
        config.L_energy_levels = args.energy_levels
    if args.periods:
        config.T_periods = args.periods
    if args.taxis:
        config.num_taxis = args.taxis
    if args.stations:
        config.num_stations = args.stations
    if args.solver:
        config.solver_method = args.solver
    if args.beta is not None:
        config.beta = args.beta
    if args.output:
        config.output_dir = args.output
    if args.seed:
        config.random_seed = args.seed
    
    # 验证配置
    _validate_config(config)
    
    return config

def _validate_config(config: SimulationConfig):
    """验证配置的有效性。"""
    if config.m_areas <= 0:
        raise ValueError("区域数量必须大于0")
    if config.L_energy_levels <= 0:
        raise ValueError("能量等级数量必须大于0")
    if config.T_periods <= 0:
        raise ValueError("时间段数量必须大于0")
    if config.simulation_duration <= 0:
        raise ValueError("模拟时长必须大于0")
    if config.num_taxis <= 0:
        raise ValueError("出租车数量必须大于0")
    if config.num_stations <= 0:
        raise ValueError("换电站数量必须大于0")

def prepare_data_and_network(config: SimulationConfig) -> Dict:
    """准备数据和网络信息。"""
    print("\n=== 数据准备阶段 ===")
    
    # 加载NYC出租车数据
    if os.path.exists(config.data_filepath):
        print(f"加载数据: {config.data_filepath}")
        
        sample_size = config.sample_size if config.use_sample else None
        raw_data = load_trip_data(config.data_filepath, sample_size, config.random_seed)
        
        print("清洗和处理数据...")
        clean_data = clean_trip_data(raw_data)
        
        print("准备模拟数据...")
        sim_data = prepare_simulation_data(clean_data, config.__dict__)
        
        # 调整区域数量以匹配实际数据
        actual_areas = len(sim_data['block_positions'])
        if actual_areas != config.m_areas:
            print(f"调整区域数量: {config.m_areas} -> {actual_areas}")
            config.m_areas = actual_areas
        
    else:
        print(f"数据文件不存在: {config.data_filepath}")
        print("生成模拟数据...")
        sim_data = _generate_synthetic_data(config)
    
    # 创建网络矩阵
    print("创建网络矩阵...")
    network_config = {
        'avg_speed_km_per_min': 0.5,
        'consumption_rate_kwh_per_km': config.consumption_rate,
        'max_travel_time_minutes': 60
    }
    
    matrices = create_all_matrices(sim_data['block_positions'], network_config)
    
    return {
        'trip_data': sim_data.get('trip_data'),
        'demand_data': sim_data.get('demand_data'),
        'block_positions': sim_data['block_positions'],
        'taxi_info': sim_data.get('taxi_info'),
        'matrices': matrices,
        'summary': sim_data.get('summary', {})
    }

def _generate_synthetic_data(config: SimulationConfig) -> Dict:
    """生成合成数据。"""
    print("生成合成数据...")
    
    # 创建网格布局的区块位置
    grid_size = int(np.ceil(np.sqrt(config.m_areas)))
    block_positions = {}
    
    for i in range(config.m_areas):
        row = i // grid_size
        col = i % grid_size
        block_positions[i] = (col * 10.0, row * 10.0)
    
    # 生成合成需求数据
    demand_data = []
    for t in range(config.T_periods):
        for area in range(config.m_areas):
            # 基于时间的需求模式
            hour = (t * config.period_length_minutes // 60) % 24
            base_demand = 10
            
            # 早晚高峰
            if 7 <= hour <= 9 or 17 <= hour <= 19:
                demand = base_demand * 2
            elif 22 <= hour or hour <= 5:
                demand = base_demand * 0.5
            else:
                demand = base_demand
            
            demand_data.append({
                'time_period': t,
                'hour': hour,
                'block_id': area,
                'demand_count': int(demand + np.random.normal(0, 2))
            })
    
    return {
        'block_positions': block_positions,
        'demand_data': pd.DataFrame(demand_data),
        'summary': {
            'total_trips': len(demand_data),
            'unique_blocks': config.m_areas,
            'synthetic_data': True
        }
    }

def optimize_station_placement(config: SimulationConfig, data: Dict) -> List[Dict]:
    """优化换电站布局。"""
    print("\n=== 换电站布局优化 ===")
    
    # 从需求数据提取需求分布
    if 'demand_data' in data and data['demand_data'] is not None:
        demand_by_block = data['demand_data'].groupby('block_id')['demand_count'].sum().to_dict()
    else:
        # 使用均匀分布
        demand_by_block = {i: 100 for i in range(config.m_areas)}
    
    # 候选位置（所有区块）
    candidate_locations = data['block_positions']
    
    print(f"优化 {config.num_stations} 个换电站的位置...")
    
    # 使用优化器选择站点位置
    selected_locations = optimize_bss_layout(
        demand_by_block, 
        candidate_locations, 
        config.num_stations
    )
    
    # 创建站点配置
    stations = []
    for i, location in enumerate(selected_locations):
        station_config = {
            'id': i,
            'location': location,
            'capacity': config.station_capacity,
            'chargers': config.station_chargers,
            'initial_batteries': config.station_capacity // 2
        }
        stations.append(station_config)
    
    print(f"选定站点位置: {selected_locations}")
    
    return stations

def create_optimization_models(config: SimulationConfig, data: Dict, stations: List[Dict]) -> Dict:
    """创建优化模型。"""
    print("\n=== 创建优化模型 ===")
    
    # 创建需求字典
    demand_dict = {}
    if 'demand_data' in data and data['demand_data'] is not None:
        for _, row in data['demand_data'].iterrows():
            t = int(row['time_period']) if 'time_period' in row else int(row['hour']) // (config.period_length_minutes // 60)
            area = int(row['block_id'])
            demand = float(row['demand_count'])
            demand_dict[(t, area)] = demand
    
    # 创建联合优化器配置
    opt_config = OptimizationConfig(
        m_areas=config.m_areas,
        L_energy_levels=config.L_energy_levels,
        T_periods=config.T_periods,
        period_length_minutes=config.period_length_minutes,
        beta=config.beta,
        solver_method=config.solver_method,
        time_limit=config.time_limit,
        mip_gap=config.mip_gap
    )
    
    # 网络数据
    matrices = data['matrices']
    network_data = {
        'distance_matrix': matrices['distance_matrix'],
        'reachability_matrix': matrices['reachability_matrix'],
        'travel_time_matrices': matrices['travel_time_matrices']
    }
    
    # 创建联合优化器
    joint_optimizer = JointOptimizer(opt_config, network_data)
    
    # 添加换电站
    for station in stations:
        joint_optimizer.add_battery_swap_station(station)
    
    # 设置需求预测
    joint_optimizer.set_demand_forecast(demand_dict)
    
    # 创建充电任务生成器
    charge_config = {
        'L_energy_levels': config.L_energy_levels,
        'period_length_minutes': config.period_length_minutes,
        'pricing': {
            'base_price': 0.12,
            'peak_hours': [(7, 10), (17, 20)],
            'peak_multiplier': 1.8,
            'off_peak_discount': 0.7
        }
    }
    
    charge_generator = ChargeTaskGenerator(charge_config)
    
    return {
        'joint_optimizer': joint_optimizer,
        'charge_generator': charge_generator,
        'demand_dict': demand_dict
    }

def initialize_system_state(config: SimulationConfig, stations: List[Dict]) -> Dict:
    """初始化系统状态。"""
    print("\n=== 初始化系统状态 ===")
    
    # 初始化出租车分布
    vacant_taxis = np.zeros((config.m_areas, config.L_energy_levels))
    occupied_taxis = np.zeros((config.m_areas, config.L_energy_levels))
    
    # 随机分布出租车
    np.random.seed(config.random_seed)
    
    for _ in range(config.num_taxis):
        area = np.random.randint(0, config.m_areas)
        energy_level = np.random.randint(config.L_energy_levels // 2, config.L_energy_levels)  # 初始电量较高
        
        if np.random.random() < 0.7:  # 70%概率为空闲
            vacant_taxis[area, energy_level] += 1
        else:
            occupied_taxis[area, energy_level] += 1
    
    # 初始化换电站库存
    bss_inventories = {}
    for station in stations:
        station_id = station['id']
        capacity = station['capacity']
        
        # 初始库存分布（更多高电量电池）
        inventory = np.zeros(config.L_energy_levels)
        for _ in range(station['initial_batteries']):
            level = np.random.randint(config.L_energy_levels // 2, config.L_energy_levels)
            inventory[level] += 1
        
        bss_inventories[station_id] = inventory
    
    system_state = {
        'vacant_taxis': vacant_taxis,
        'occupied_taxis': occupied_taxis,
        'bss_inventories': bss_inventories,
        'current_time': 0
    }
    
    print(f"初始化 {config.num_taxis} 辆出租车")
    print(f"初始化 {len(stations)} 个换电站")
    
    return system_state

def run_simulation(config: SimulationConfig, models: Dict, initial_state: Dict, stations: List[Dict]) -> Dict:
    """运行主模拟循环。"""
    print("\n=== 开始模拟 ===")
    
    joint_optimizer = models['joint_optimizer']
    charge_generator = models['charge_generator']
    
    # 模拟结果存储
    results = {
        'config': config.__dict__,
        'stations': stations,
        'simulation_history': [],
        'optimization_decisions': [],
        'charging_decisions': [],
        'performance_metrics': []
    }
    
    # 当前状态
    current_state = initial_state.copy()
    
    # 计算模拟时间段数
    simulation_periods = config.simulation_duration // config.period_length_minutes
    simulation_periods = min(simulation_periods, config.T_periods)
    
    print(f"模拟 {simulation_periods} 个时间段 ({config.simulation_duration} 分钟)")
    
    for t in range(simulation_periods):
        print(f"\n--- 时间段 {t} ---")
        
        # 1. 优化当前时间段的决策
        optimization_start = datetime.now()
        
        decisions = joint_optimizer.optimize_single_period(t, current_state)
        
        optimization_time = (datetime.now() - optimization_start).total_seconds()
        
        # 2. 优化充电调度
        charging_decisions = {}
        for station in stations:
            station_id = station['id']
            inventory = current_state['bss_inventories'][station_id]
            
            charging_decision = joint_optimizer.optimize_charging_schedule(
                t, {station_id: inventory}
            )
            charging_decisions[station_id] = charging_decision
        
        # 3. 更新系统状态
        current_state = _update_system_state(
            current_state, decisions, charging_decisions, config
        )
        current_state['current_time'] = t + 1
        
        # 4. 记录结果
        period_results = {
            'time_period': t,
            'optimization_time': optimization_time,
            'decisions': decisions,
            'charging_decisions': charging_decisions,
            'system_state': _serialize_state(current_state),
            'metrics': _calculate_period_metrics(current_state, decisions)
        }
        
        results['simulation_history'].append(period_results)
        
        # 5. 输出进度
        if t % 10 == 0 or t == simulation_periods - 1:
            total_served = sum(np.sum(d['served_passengers']) for d in 
                             [r['decisions'] for r in results['simulation_history']])
            print(f"已完成 {t+1}/{simulation_periods} 时间段，累计服务乘客: {total_served}")
    
    # 6. 计算最终性能指标
    results['final_metrics'] = _calculate_final_metrics(results, config)
    
    # 7. 获取优化器性能摘要
    results['optimizer_summary'] = joint_optimizer.get_performance_summary()
    
    print(f"\n模拟完成！总优化时间: {sum(r['optimization_time'] for r in results['simulation_history']):.2f}秒")
    
    return results

def _update_system_state(current_state: Dict, decisions: Dict, 
                        charging_decisions: Dict, config: SimulationConfig) -> Dict:
    """更新系统状态。"""
    new_state = {
        'vacant_taxis': current_state['vacant_taxis'].copy(),
        'occupied_taxis': current_state['occupied_taxis'].copy(),
        'bss_inventories': {}
    }
    
    # 简化的状态更新（实际应用中需要更复杂的状态转移）
    
    # 更新出租车分布
    served_passengers = decisions.get('served_passengers', np.zeros(config.m_areas))
    for area in range(config.m_areas):
        # 简单的状态转移：一部分空闲车变为载客车
        if served_passengers[area] > 0:
            for level in range(config.L_energy_levels-1, -1, -1):
                available = new_state['vacant_taxis'][area, level]
                if available > 0:
                    transfer = min(available, served_passengers[area])
                    new_state['vacant_taxis'][area, level] -= transfer
                    new_state['occupied_taxis'][area, level] += transfer
                    served_passengers[area] -= transfer
                    if served_passengers[area] <= 0:
                        break
    
    # 更新换电站库存
    for station_id, charging_decision in charging_decisions.items():
        if station_id in current_state['bss_inventories']:
            inventory = current_state['bss_inventories'][station_id].copy()
            
            # 应用充电决策（简化）
            if 'charging_schedule' in charging_decision:
                schedule = charging_decision['charging_schedule']
                for level in range(len(schedule)):
                    charge_amount = schedule[level]
                    if charge_amount > 0 and level < config.L_energy_levels - 1:
                        # 从低能量等级充电到高能量等级
                        actual_charge = min(charge_amount, inventory[level])
                        inventory[level] -= actual_charge
                        inventory[level + 1] += actual_charge
            
            new_state['bss_inventories'][station_id] = inventory
    
    return new_state

def _serialize_state(state: Dict) -> Dict:
    """序列化状态以便保存。"""
    serialized = {}
    for key, value in state.items():
        if isinstance(value, np.ndarray):
            serialized[key] = value.tolist()
        elif isinstance(value, dict):
            serialized[key] = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                             for k, v in value.items()}
        else:
            serialized[key] = value
    return serialized

def _calculate_period_metrics(state: Dict, decisions: Dict) -> Dict:
    """计算单个时间段的性能指标。"""
    metrics = {
        'total_vacant_taxis': np.sum(state['vacant_taxis']),
        'total_occupied_taxis': np.sum(state['occupied_taxis']),
        'total_served_passengers': np.sum(decisions.get('served_passengers', 0)),
        'utilization_rate': 0,
        'avg_battery_level': 0
    }
    
    total_taxis = metrics['total_vacant_taxis'] + metrics['total_occupied_taxis']
    if total_taxis > 0:
        metrics['utilization_rate'] = metrics['total_occupied_taxis'] / total_taxis
    
    # 计算平均电池等级
    total_weighted_level = 0
    total_taxis = 0
    for area in range(state['vacant_taxis'].shape[0]):
        for level in range(state['vacant_taxis'].shape[1]):
            count = state['vacant_taxis'][area, level] + state['occupied_taxis'][area, level]
            total_weighted_level += level * count
            total_taxis += count
    
    if total_taxis > 0:
        metrics['avg_battery_level'] = total_weighted_level / total_taxis
    
    return metrics

def _calculate_final_metrics(results: Dict, config: SimulationConfig) -> Dict:
    """计算最终性能指标。"""
    history = results['simulation_history']
    
    final_metrics = {
        'total_passengers_served': sum(r['metrics']['total_served_passengers'] for r in history),
        'avg_utilization_rate': np.mean([r['metrics']['utilization_rate'] for r in history]),
        'avg_optimization_time': np.mean([r['optimization_time'] for r in history]),
        'total_simulation_time': len(history) * config.period_length_minutes,
        'service_efficiency': 0,  # 乘客数/时间
        'system_efficiency': 0   # 乘客数/车辆数
    }
    
    if final_metrics['total_simulation_time'] > 0:
        final_metrics['service_efficiency'] = (final_metrics['total_passengers_served'] / 
                                             final_metrics['total_simulation_time'])
    
    if config.num_taxis > 0:
        final_metrics['system_efficiency'] = (final_metrics['total_passengers_served'] / 
                                            config.num_taxis)
    
    return final_metrics

def save_results(results: Dict, config: SimulationConfig):
    """保存模拟结果。"""
    print("\n=== 保存结果 ===")
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{config.output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存主要结果
    results_file = os.path.join(output_dir, 'simulation_results.json')
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"结果已保存到: {results_file}")
    
    # 保存配置
    config_file = os.path.join(output_dir, 'config.json')
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config.__dict__, f, indent=2)
    
    # 生成性能报告
    _generate_performance_report(results, output_dir)
    
    # 生成可视化
    if config.save_detailed_results:
        _generate_visualizations(results, output_dir)
    
    return output_dir

def _generate_performance_report(results: Dict, output_dir: str):
    """生成性能报告。"""
    try:
        report_file = os.path.join(output_dir, 'performance_report.txt')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("电动出租车电池交换系统模拟性能报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 基本信息
            config = results['config']
            f.write(f"模拟参数:\n")
            f.write(f"  区域数量: {config['m_areas']}\n")
            f.write(f"  能量等级: {config['L_energy_levels']}\n")
            f.write(f"  时间段数: {config['T_periods']}\n")
            f.write(f"  出租车数量: {config['num_taxis']}\n")
            f.write(f"  换电站数量: {config['num_stations']}\n\n")
            
            # 性能指标
            metrics = results['final_metrics']
            f.write(f"性能指标:\n")
            f.write(f"  总服务乘客数: {metrics['total_passengers_served']}\n")
            f.write(f"  平均利用率: {metrics['avg_utilization_rate']:.2%}\n")
            f.write(f"  平均优化时间: {metrics['avg_optimization_time']:.2f}秒\n")
            f.write(f"  服务效率: {metrics['service_efficiency']:.2f} 乘客/分钟\n")
            f.write(f"  系统效率: {metrics['system_efficiency']:.2f} 乘客/车辆\n\n")
            
            # 优化器统计
            if 'optimizer_summary' in results:
                opt_summary = results['optimizer_summary']
                f.write(f"优化器统计:\n")
                f.write(f"  优化周期数: {opt_summary.get('total_periods_optimized', 0)}\n")
                f.write(f"  平均响应时间: {opt_summary.get('avg_response_time', 0):.2f}秒\n")
                f.write(f"  平均利用率: {opt_summary.get('avg_utilization', 0):.2%}\n")
        
        print(f"性能报告已保存到: {report_file}")
        
    except Exception as e:
        print(f"生成性能报告时出错: {e}")

def _generate_visualizations(results: Dict, output_dir: str):
    """生成可视化图表。"""
    try:
        # 可视化需要特定的数据格式，这里创建兼容的格式
        viz_data = {
            'station_stats': {
                'station_locations': {s['id']: s['location'] for s in results['stations']},
                'station_swap_counts': {s['id']: 10 for s in results['stations']},  # 示例数据
                'station_avg_wait': {s['id']: 5.0 for s in results['stations']},
                'charged_batteries': {s['id']: 15 for s in results['stations']},
                'empty_batteries': {s['id']: 5 for s in results['stations']}
            },
            'taxi_stats': {
                'trips_completed': results['final_metrics']['total_passengers_served'],
                'trips_per_taxi': {i: 10 for i in range(results['config']['num_taxis'])},
                'swap_count': {i: 3 for i in range(results['config']['num_taxis'])},
                'revenue_per_taxi': {i: 200.0 for i in range(results['config']['num_taxis'])}
            },
            'performance_metrics': results['final_metrics']
        }
        
        # 生成站点指标图
        station_plot = os.path.join(output_dir, 'station_metrics.png')
        plot_station_metrics(viz_data, save_path=station_plot)
        
        # 生成出租车指标图
        taxi_plot = os.path.join(output_dir, 'taxi_metrics.png')
        plot_taxi_metrics(viz_data, save_path=taxi_plot)
        
        # 生成优化结果图
        opt_plot = os.path.join(output_dir, 'optimization_results.png')
        plot_optimization_results(results, save_path=opt_plot)
        
        print(f"可视化图表已保存到: {output_dir}")
        
    except Exception as e:
        print(f"生成可视化时出错: {e}")

def main():
    """主程序入口。"""
    print("电动出租车电池交换系统模拟 (基于论文实现)")
    print("=" * 60)
    
    try:
        # 1. 解析参数和加载配置
        args = parse_arguments()
        config = load_and_validate_config(args)
        
        # 设置随机种子
        np.random.seed(config.random_seed)
        random.seed(config.random_seed)
        
        print(f"模拟配置:")
        print(f"  区域数量: {config.m_areas}")
        print(f"  能量等级: {config.L_energy_levels}")
        print(f"  时间段数: {config.T_periods}")
        print(f"  出租车数量: {config.num_taxis}")
        print(f"  换电站数量: {config.num_stations}")
        print(f"  优化方法: {config.solver_method}")
        print(f"  数据文件: {config.data_filepath}")
        
        # 2. 准备数据和网络
        data = prepare_data_and_network(config)
        print(f"数据摘要: {data['summary']}")
        
        # 3. 优化换电站布局
        stations = optimize_station_placement(config, data)
        
        # 4. 创建优化模型
        models = create_optimization_models(config, data, stations)
        
        # 5. 初始化系统状态
        initial_state = initialize_system_state(config, stations)
        
        # 6. 运行模拟
        results = run_simulation(config, models, initial_state, stations)
        
        # 7. 保存结果
        output_dir = save_results(results, config)
        
        # 8. 显示最终结果
        print("\n" + "=" * 60)
        print("模拟完成！")
        print("=" * 60)
        
        metrics = results['final_metrics']
        print(f"总服务乘客数: {metrics['total_passengers_served']}")
        print(f"平均利用率: {metrics['avg_utilization_rate']:.2%}")
        print(f"服务效率: {metrics['service_efficiency']:.2f} 乘客/分钟")
        print(f"系统效率: {metrics['system_efficiency']:.2f} 乘客/车辆")
        print(f"平均优化时间: {metrics['avg_optimization_time']:.2f} 秒")
        
        print(f"\n结果已保存到: {output_dir}")
        print(f"详细报告: {output_dir}/performance_report.txt")
        
        if 'optimizer_summary' in results:
            opt_summary = results['optimizer_summary']
            print(f"\n优化器性能:")
            print(f"  优化周期数: {opt_summary.get('total_periods_optimized', 0)}")
            print(f"  总服务乘客: {opt_summary.get('total_passengers_served', 0)}")
            print(f"  总换电次数: {opt_summary.get('total_swaps_completed', 0)}")
        
    except KeyboardInterrupt:
        print("\n\n用户中断执行。")
        sys.exit(1)
    except Exception as e:
        print(f"\n发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

def main():
    """主程序入口。"""
    print("电动出租车电池交换系统模拟 (基于论文实现)")
    print("=" * 60)
    
    try:
        # 1. 解析参数和加载配置
        args = parse_arguments()
        config = load_and_validate_config(args)
        
        # 设置随机种子
        np.random.seed(config.random_seed)
        random.seed(config.random_seed)
        
        print(f"模拟配置:")
        print(f"  区域数量: {config.m_areas}")
        print(f"  能量等级: {config.L_energy_levels}")
        print(f"  时间段数: {config.T_periods}")
        print(f"  出租车数量: {config.num_taxis}")
        print(f"  换电站数量: {config.num_stations}")
        print(f"  优化方法: {config.solver_method}")
        print(f"  数据文件: {config.data_filepath}")
        
        # 2. 准备数据和网络
        data = prepare_data_and_network(config)
        print(f"数据摘要: {data['summary']}")
        
        # 3. 优化换电站布局
        stations = optimize_station_placement(config, data)
        
        # 4. 创建优化模型
        models = create_optimization_models(config, data, stations)
        
        # 5. 初始化系统状态
        initial_state = initialize_system_state(config, stations)
        
        # 6. 运行模拟
        results = run_simulation(config, models, initial_state, stations)
        
        # 7. 保存结果
        output_dir = save_results(results, config)
        
        # 8. 显示最终结果
        print("\n" + "=" * 60)
        print("模拟完成！")
        print("=" * 60)
        
        metrics = results['final_metrics']
        print(f"总服务乘客数: {metrics['total_passengers_served']}")
        print(f"平均利用率: {metrics['avg_utilization_rate']:.2%}")
        print(f"服务效率: {metrics['service_efficiency']:.2f} 乘客/分钟")
        print(f"系统效率: {metrics['system_efficiency']:.2f} 乘客/车辆")
        print(f"平均优化时间: {metrics['avg_optimization_time']:.2f} 秒")
        
        print(f"\n结果已保存到: {output_dir}")
        print(f"详细报告: {output_dir}/performance_report.txt")
        
        if 'optimizer_summary' in results:
            opt_summary = results['optimizer_summary']
            print(f"\n优化器性能:")
            print(f"  优化周期数: {opt_summary.get('total_periods_optimized', 0)}")
            print(f"  总服务乘客: {opt_summary.get('total_passengers_served', 0)}")
            print(f"  总换电次数: {opt_summary.get('total_swaps_completed', 0)}")
        
    except KeyboardInterrupt:
        print("\n\n用户中断执行。")
        sys.exit(1)
    except Exception as e:
        print(f"\n发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()