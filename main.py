# main.py
import os
import logging
import numpy as np
from config import SimulationConfig
from models.demand_model import ODMatrixDemandModel
from models.travel_model import TravelModel
from models.optimization_model import JointOptimizer
from utils.visualization import plot_station_metrics, plot_taxi_metrics, plot_performance_metrics
from simulation.simulation import run_simulation

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _generate_visualizations(results: dict, output_dir: str):
    """使用真实的模拟结果生成可视化图表。"""
    if not results:
        logging.warning("模拟结果为空，跳过可视化生成。")
        return
        
    logging.info("开始生成可视化图表...")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        stations_config = results.get('stations_config', [])
        if not stations_config:
            logging.warning("可视化数据中缺少 'stations_config'。")
            return

        station_ids = [s['id'] for s in stations_config]
        config = SimulationConfig() # 加载配置以获取容量信息
        station_capacities = {s['id']: s['capacity'] for s in config.stations}

        # 1. 站点指标
        station_swap_counts = results.get('station_swap_counts', {})
        swap_counts_full = {sid: station_swap_counts.get(sid, 0) for sid in station_ids}

        final_inventories = results.get('final_station_inventories', {})
        
        charged_batteries = {sid: final_inventories.get(sid, 0) for sid in station_ids}
        
        # 【逻辑修正】修复了 .get() 方法的参数错误
        # 使用 station_capacities.get(sid, 20) 来获取容量，如果找不到则默认容量为20
        empty_batteries = {
            sid: station_capacities.get(sid, 20) - charged_batteries.get(sid, 0) 
            for sid in station_ids
        }
        
        # 简化处理：目前全局等待时间作为每个站点的等待时间
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
        
        # 3. 构建传递给绘图函数的数据结构
        viz_data = {
            'station_stats': station_stats,
            'performance_metrics': performance_metrics,
            'taxi_stats': {}, # 出租车相关统计目前未详细实现
        }

        plot_station_metrics(viz_data, os.path.join(output_dir, "station_metrics.png"))
        plot_performance_metrics(viz_data, os.path.join(output_dir, "performance_metrics.png"))

        logging.info(f"可视化图表已成功保存到: {output_dir}")

    except Exception as e:
        logging.error(f"生成可视化时出错: {e}", exc_info=True)


def main():
    """主执行函数。"""
    # 1. 加载配置
    config = SimulationConfig()
    output_dir = "output"
    np.random.seed(42) # 设置随机种子以保证结果可复现

    # 2. 初始化模型
    logging.info("正在初始化模型...")
    demand_model = ODMatrixDemandModel(config.m_areas, config.T_periods)
    travel_model = TravelModel(config.m_areas, speed_kph=config.avg_speed) 
    optimizer = JointOptimizer(config, demand_model, travel_model)

    # 3. 运行模拟
    results = run_simulation(config, optimizer)

    # 4. 生成并保存可视化结果
    if results:
        _generate_visualizations(results, output_dir)
    else:
        logging.error("模拟未返回结果，无法生成可视化。")

if __name__ == "__main__":
    main()