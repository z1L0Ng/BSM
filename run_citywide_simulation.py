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
from dataprocess.distance import create_all_matrices # 导入距离计算工具
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
        
        # 清洗数据
        df = clean_trip_data(df)
        
        # 准备模拟数据
        simulation_data = prepare_simulation_data(df, config.__dict__)
        
        logging.info(f"数据准备完成: {len(df)} 条记录")
        return simulation_data
    else:
        logging.info("使用模拟数据生成需求...")
        return None

def create_citywide_demand_model(config, simulation_data=None):
    """创建城市级需求模型"""
    demand_model = ODMatrixDemandModel(config.m_areas, config.T_periods)
    
    if simulation_data and 'demand_data' in simulation_data:
        logging.info("基于真实数据创建需求模型...")
        # 从聚合的需求数据中填充需求矩阵
        demand_df = simulation_data['demand_data']
        
        # 创建一个 O-D 矩阵
        od_matrix = np.zeros((config.T_periods, config.m_areas, config.m_areas))
        
        # 从 trip_data 中计算 OD 对
        trip_df = simulation_data['trip_data']
        if 'pickup_block' in trip_df.columns and 'dropoff_block' in trip_df.columns:
            # 按时间、起点、终点聚合
            od_counts = trip_df.groupby(['time_period', 'pickup_block', 'dropoff_block']).size().reset_index(name='count')
            
            for _, row in od_counts.iterrows():
                t = int(row['time_period'])
                origin = int(row['pickup_block'])
                dest = int(row['dropoff_block'])
                count = int(row['count'])
                
                if 0 <= t < config.T_periods and 0 <= origin < config.m_areas and 0 <= dest < config.m_areas:
                    od_matrix[t, origin, dest] = count

        demand_model.demand_matrix = od_matrix
        logging.info("真实数据需求模型创建完成。")
        
    else:
        logging.info("创建城市级模拟需求模型...")
        # 如果没有真实数据，这里可以调用 demand_model 内部的模拟数据生成方法
        demand_model.simulate_demand()

    return demand_model


def create_citywide_travel_model(config, simulation_data=None):
    """创建城市级出行模型"""
    # 【修改处】在初始化TravelModel时，传入能耗率参数
    travel_model = TravelModel(config.m_areas, config.energy_consumption_rate)

    if simulation_data and simulation_data.get('block_positions'):
        logging.info("基于真实数据创建出行模型...")
        # 使用区块的平均坐标计算距离和时间矩阵
        block_positions = simulation_data['block_positions']
        
        # 确保所有区域ID都在区块位置字典中
        for i in range(config.m_areas):
            if i not in block_positions:
                # 如果有缺失，提供一个默认或随机的位置
                block_positions[i] = (np.random.uniform(-74, -73.8), np.random.uniform(40.7, 40.8))

        matrices = create_all_matrices(block_positions)
        
        # 更新 travel_model 的距离和时间矩阵
        dist_matrix_np = np.zeros((config.m_areas, config.m_areas))
        time_matrix_np = np.zeros((config.m_areas, config.m_areas))

        # 使用字典的 get 方法安全地访问
        dist_dict = matrices.get('distance_matrix', {})
        # 使用平均时间作为代表 (例如使用中午12点作为平均交通状况)
        time_dict = matrices.get('travel_time_matrices', {}).get(12, {}) 

        for i in range(config.m_areas):
            for j in range(config.m_areas):
                dist_matrix_np[i, j] = dist_dict.get(i, {}).get(j, 0)
                time_matrix_np[i, j] = time_dict.get(i, {}).get(j, 0)
        
        travel_model.distance_matrix = dist_matrix_np
        travel_model.time_matrix = time_matrix_np
        logging.info("真实数据出行模型创建完成。")
        
    else:
        logging.info("创建城市级模拟出行模型...")
        # 如果没有真实数据，调用 travel_model 内部的模拟数据生成方法
        travel_model.simulate_distances()
        # 假设一个平均速度 (km/h)
        avg_speed_kmh = 30 
        # 将速度转换为 km/min
        avg_speed_kmpm = avg_speed_kmh / 60
        # 避免除以零
        travel_model.time_matrix = travel_model.distance_matrix / avg_speed_kmpm if avg_speed_kmpm > 0 else np.zeros_like(travel_model.distance_matrix)


    return travel_model


def _generate_citywide_visualizations(results: dict, output_dir: str, config):
    """生成城市级模拟结果的可视化图表"""
    logging.info("开始生成可视化图表...")
    try:
        # 提取站点和出租车历史数据
        station_history = results.get('station_history', {})
        taxi_history = results.get('taxi_history', {})
        performance_history = results.get('performance_history', [])

        if station_history:
            plot_station_metrics(station_history, output_dir)
        
        if taxi_history:
            plot_taxi_metrics(taxi_history, output_dir)

        if performance_history:
            plot_performance_metrics(performance_history, output_dir)
            
        logging.info(f"可视化图表已成功保存到: {output_dir}")
        
    except Exception as e:
        logging.error(f"生成可视化图表时发生错误: {e}", exc_info=True)


def generate_performance_report(results: dict, output_dir: str, config):
    """生成并保存性能报告"""
    logging.info("生成性能报告...")
    report_path = os.path.join(output_dir, "performance_report.txt")
    
    perf_hist = results.get('performance_history', [])
    if not perf_hist:
        logging.warning("性能历史数据为空，无法生成报告。")
        return
        
    # 计算平均指标
    avg_service_quality = np.mean([p['service_quality'] for p in perf_hist])
    total_idle_distance = np.sum([p['idle_distance'] for p in perf_hist])
    total_charging_cost = np.sum([p['charging_cost'] for p in perf_hist])
    avg_objective = np.mean([p['objective_value'] for p in perf_hist])

    # 模拟结束时的最终状态
    final_taxi_states = results['taxi_history'][max(results['taxi_history'].keys())]
    final_station_states = results['station_history'][max(results['station_history'].keys())]
    
    report_content = f"""
=================================================
       城市级 BSM 模拟性能报告
=================================================

模拟配置:
- 区域数量: {config.m_areas}
- 出租车数量: {config.num_taxis}
- 换电站数量: {len(config.stations)}
- 时间段总数: {config.T_periods}
- 优化算法: {'Gurobi' if config.use_gurobi else 'Heuristic'}

-------------------------------------------------
核心性能指标:
-------------------------------------------------
- 平均服务质量 (服务乘客数/时段): {avg_service_quality:.2f}
- 总空驶距离 (km): {total_idle_distance:.2f}
- 总充电成本: {total_charging_cost:.2f}
- 平均目标函数值: {avg_objective:.2f}

-------------------------------------------------
模拟结束时状态:
-------------------------------------------------
- 空闲出租车总数: {final_taxi_states['vacant']}
- 载客出租车总数: {final_taxi_states['occupied']}
- 换电中出租车总数: {final_taxi_states['swapping']}
- 总可用充满电池数: {sum(s['charged'] for s in final_station_states.values())}
- 总可用空电池数: {sum(s['empty'] for s in final_station_states.values())}

=================================================
    """
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
        
    logging.info(f"性能报告已保存到: {report_path}")


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
        # 假设主模拟函数在 simulation/simulation.py 中
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