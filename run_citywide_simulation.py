# run_citywide_simulation.py
"""
城市级BSM模拟运行脚本 - 扩展到整个城市数据规模
"""
import os
import sys
import time
import logging
from datetime import datetime
from config import CityWideSimulationConfig
from models.demand_model import ODMatrixDemandModel
from models.travel_model import TravelModel
from models.optimization_model import JointOptimizer
from utils.visualization import plot_station_metrics, plot_taxi_metrics, plot_performance_metrics
from simulation.simulation import run_simulation
from dataprocess.loaddata import load_trip_data, clean_trip_data, prepare_simulation_data
import pandas as pd
import numpy as np

def setup_logging(config):
    """设置日志记录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"results_citywide_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "simulation.log")
    
    logging.basicConfig(
        level=logging.INFO if config.detailed_logging else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return log_dir

def load_and_prepare_data(config):
    """加载和准备城市级数据"""
    logging.info("开始加载城市级数据...")
    
    if config.use_real_data and os.path.exists(config.data_file):
        logging.info(f"使用真实数据文件: {config.data_file}")
        
        # 加载真实数据
        df = load_trip_data(config.data_file, sample_size=config.sample_size)
        
        # 简化的数据清洗 - 避免复杂操作
        df = simple_clean_data(df)
        
        # 准备模拟数据
        simulation_data = prepare_simulation_data(df, config.__dict__)
        
        logging.info(f"数据准备完成: {len(df)} 条记录")
        return simulation_data
    else:
        logging.info("使用模拟数据生成需求...")
        return None

def simple_clean_data(df):
    """简化的数据清洗函数，避免复杂操作"""
    print("开始简化数据清洗...")
    
    # 创建副本
    df = df.copy()
    df = df.reset_index(drop=True)
    
    # 确保需要的列存在
    if 'pickup_datetime' not in df.columns:
        if 'tpep_pickup_datetime' in df.columns:
            df['pickup_datetime'] = df['tpep_pickup_datetime']
    
    # 转换时间列
    if 'pickup_datetime' in df.columns:
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    
    # 只保留白天的行程
    if 'pickup_datetime' in df.columns:
        df = df[df['pickup_datetime'].dt.hour >= 6]
        df = df.reset_index(drop=True)
    
    # 基本的数据过滤
    if 'trip_distance' in df.columns:
        df = df[df['trip_distance'] > 0]
        df = df.reset_index(drop=True)
    
    # 创建区域块ID
    if 'pickup_latitude' in df.columns and 'pickup_longitude' in df.columns:
        # 简单的区域划分
        df['pickup_block'] = ((df['pickup_latitude'] - 40.4) * 100).astype(int) % 100
        df['block_id'] = df['pickup_block']
    else:
        # 随机分配区域
        df['block_id'] = np.random.randint(0, 100, len(df))
    
    # 处理出租车ID
    if 'taxi_id' not in df.columns:
        df['taxi_id'] = np.random.randint(1, 1001, len(df))
    
    # 创建时间特征
    if 'pickup_datetime' in df.columns:
        df['hour'] = df['pickup_datetime'].dt.hour
        df['time_period'] = (df['hour'] * 3 + df['pickup_datetime'].dt.minute // 20) % 72
    
    print(f"数据清洗完成: {len(df)} 条记录")
    return df

def create_citywide_demand_model(config, simulation_data=None):
    """创建城市级需求模型"""
    if simulation_data is not None:
        # 使用真实数据创建需求模型
        logging.info("基于真实数据创建需求模型...")
        demand_model = ODMatrixDemandModel(config.m_areas, config.T_periods)
        
        # 这里可以集成真实数据的需求矩阵
        # TODO: 实现基于真实数据的需求矩阵生成
        
        return demand_model
    else:
        logging.info("创建城市级模拟需求模型...")
        return ODMatrixDemandModel(config.m_areas, config.T_periods)

def create_citywide_travel_model(config, simulation_data=None):
    """创建城市级出行模型"""
    if simulation_data is not None:
        logging.info("基于真实数据创建出行模型...")
        travel_model = TravelModel(config.m_areas)
        
        # 这里可以集成真实数据的距离和时间矩阵
        # TODO: 实现基于真实数据的距离矩阵生成
        
        return travel_model
    else:
        logging.info("创建城市级模拟出行模型...")
        return TravelModel(config.m_areas)

def _generate_citywide_visualizations(results: dict, output_dir: str, config):
    """生成城市级模拟结果的可视化图表"""
    logging.info("开始生成城市级可视化图表...")
    
    if not config.visualization_enabled:
        logging.info("可视化功能已禁用")
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        # 从真实的模拟结果中提取数据
        stations_config = results['stations_config']
        station_ids = [s['id'] for s in stations_config]

        # 1. 站点指标
        station_swap_counts = results['station_swap_counts']
        swap_counts_full = {sid: station_swap_counts.get(sid, 0) for sid in station_ids}

        # 最终电池库存
        final_inventories = results['final_station_inventories']
        
        # 计算每个站点的电池状态
        station_capacities = {s['id']: s['capacity'] for s in config.stations}
        charged_batteries = {sid: final_inventories.get(sid, 0) for sid in station_ids}
        empty_batteries = {sid: station_capacities.get(sid, 0) - charged_batteries.get(sid, 0) for sid in station_ids}
        
        # 平均等待时间
        avg_wait_time = results.get('average_wait_time', 0)
        station_avg_wait = {sid: avg_wait_time for sid in station_ids}

        station_stats = {
            'station_locations': {s['id']: s['location'] for s in stations_config},
            'station_swap_counts': swap_counts_full,
            'station_avg_wait': station_avg_wait,
            'charged_batteries': charged_batteries,
            'empty_batteries': empty_batteries,
        }

        # 2. 性能指标
        performance_metrics = {
            'average_wait_time': results.get('average_wait_time', 0),
            'total_passengers_served': results.get('total_passengers_served', 0),
            'total_idle_distance': results.get('total_idle_distance', 0),
        }
        
        # 3. 构建 viz_data
        viz_data = {
            'station_stats': station_stats,
            'performance_metrics': performance_metrics,
            'taxi_stats': {}, 
        }

        # 调用绘图函数
        plot_station_metrics(viz_data, os.path.join(output_dir, "citywide_station_metrics.png"))
        plot_performance_metrics(viz_data, os.path.join(output_dir, "citywide_performance_metrics.png"))

        logging.info(f"城市级可视化图表已保存到: {output_dir}")

    except Exception as e:
        logging.error(f"生成城市级可视化时出错: {e}", exc_info=True)

def generate_performance_report(results: dict, output_dir: str, config):
    """生成性能报告"""
    logging.info("生成性能报告...")
    
    report_file = os.path.join(output_dir, "citywide_performance_report.txt")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=== 城市级BSM模拟性能报告 ===\n\n")
        f.write(f"模拟时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("=== 配置参数 ===\n")
        f.write(f"城市区域数量: {config.m_areas}\n")
        f.write(f"出租车数量: {config.num_taxis}\n")
        f.write(f"换电站数量: {len(config.stations)}\n")
        f.write(f"模拟时长: {config.simulation_duration} 分钟\n")
        f.write(f"使用真实数据: {'是' if config.use_real_data else '否'}\n\n")
        
        f.write("=== 模拟结果 ===\n")
        f.write(f"总服务乘客数: {results.get('total_passengers_served', 0)}\n")
        f.write(f"平均等待时间: {results.get('average_wait_time', 0):.2f} 分钟\n")
        f.write(f"总空驶距离: {results.get('total_idle_distance', 0):.2f} 公里\n")
        f.write(f"总换电次数: {sum(results.get('station_swap_counts', {}).values())}\n")
        
        # 换电站利用率
        f.write("\n=== 换电站利用率 ===\n")
        station_swaps = results.get('station_swap_counts', {})
        for station_id, swaps in station_swaps.items():
            f.write(f"{station_id}: {swaps} 次换电\n")
        
        f.write("\n=== 系统效率指标 ===\n")
        total_swaps = sum(station_swaps.values())
        if total_swaps > 0:
            f.write(f"平均每个换电站换电次数: {total_swaps/len(config.stations):.2f}\n")
        
        if results.get('total_passengers_served', 0) > 0:
            f.write(f"每次服务的平均空驶距离: {results.get('total_idle_distance', 0)/results.get('total_passengers_served', 1):.2f} 公里\n")
    
    logging.info(f"性能报告已保存到: {report_file}")

def main():
    """城市级模拟主函数"""
    print("=" * 60)
    print("启动城市级BSM模拟系统")
    print("=" * 60)
    
    # 1. 加载城市级配置
    config = CityWideSimulationConfig()
    
    # 2. 设置日志和输出目录
    output_dir = setup_logging(config)
    
    logging.info("开始城市级BSM模拟...")
    logging.info(f"配置: {config.m_areas}个区域, {config.num_taxis}辆出租车, {len(config.stations)}个换电站")
    
    start_time = time.time()
    
    try:
        # 3. 加载和准备数据
        simulation_data = load_and_prepare_data(config)
        
        # 4. 初始化模型
        demand_model = create_citywide_demand_model(config, simulation_data)
        travel_model = create_citywide_travel_model(config, simulation_data)
        
        # 5. 创建优化器
        optimizer = JointOptimizer(config, demand_model, travel_model)
        
        # 6. 运行城市级模拟
        logging.info("开始运行城市级模拟...")
        results = run_simulation(config, optimizer)
        
        # 7. 生成结果和可视化
        if results:
            _generate_citywide_visualizations(results, output_dir, config)
            generate_performance_report(results, output_dir, config)
            
            end_time = time.time()
            simulation_time = end_time - start_time
            
            print("\n" + "=" * 60)
            print("城市级模拟完成！")
            print(f"模拟时间: {simulation_time:.2f} 秒")
            print(f"结果保存在: {output_dir}")
            print("=" * 60)
            
        else:
            logging.error("城市级模拟未返回结果")
            
    except Exception as e:
        logging.error(f"城市级模拟过程中发生错误: {e}", exc_info=True)
        print(f"模拟失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 