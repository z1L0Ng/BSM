# main.py
import os
import logging
from config import SimulationConfig
from models.demand_model import ODMatrixDemandModel
from models.travel_model import TravelModel
from models.optimization_model import JointOptimizer
from utils.visualization import plot_station_metrics, plot_taxi_metrics, plot_performance_metrics
from simulation.simulation import run_simulation # 新增: 导入新的模拟运行器

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _generate_visualizations(results: dict, output_dir: str):
    """使用真实的模拟结果生成可视化图表。"""
    logging.info("开始生成可视化图表...")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        # --- 从真实的模拟结果中提取数据 ---
        stations_config = results['stations_config']
        station_ids = [s['id'] for s in stations_config]

        # 1. 站点指标
        station_swap_counts = results['station_swap_counts']
        # 确保所有站点都有一个值，即使是0
        swap_counts_full = {sid: station_swap_counts.get(sid, 0) for sid in station_ids}

        # 最终电池库存
        final_inventories = results['final_station_inventories']
        
        # 计算每个站点的电池状态
        # 注意：这里需要从 config 中获取站点的总容量
        station_capacities = {s['id']: s['capacity'] for s in SimulationConfig.stations}
        charged_batteries = {sid: final_inventories.get(sid, 0) for sid in station_ids}
        empty_batteries = {sid: station_capacities.get(sid, 0) - charged_batteries.get(sid, 0) for sid in station_ids}
        
        # TODO: 计算每个站点的平均等待时间，这需要更详细的结果收集
        # 暂时使用全局平均等待时间作为所有站点的近似值
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
            # taxi_stats 可以类似地从 history 中构建，这里简化
            'taxi_stats': {}, 
        }

        # --- 调用绘图函数 ---
        plot_station_metrics(viz_data, os.path.join(output_dir, "station_metrics.png"))
        plot_performance_metrics(viz_data, os.path.join(output_dir, "performance_metrics.png"))
        # plot_taxi_metrics(viz_data, os.path.join(output_dir, "taxi_metrics.png")) # 如果需要

        logging.info(f"可视化图表已成功保存到: {output_dir}")

    except Exception as e:
        logging.error(f"生成可视化时出错: {e}", exc_info=True)


def main():
    """主执行函数。"""
    # 1. 加载配置
    config = SimulationConfig()
    output_dir = "simulation_results"

    # 2. 初始化模型
    demand_model = ODMatrixDemandModel(config.m_areas, config.T_periods)
    travel_model = TravelModel(config.m_areas)
    optimizer = JointOptimizer(config, demand_model, travel_model)

    # 3. 运行新的基于 SimPy 的模拟
    # 这个函数现在返回一个包含真实模拟历史和统计数据的字典
    results = run_simulation(config, optimizer)

    # 4. 生成并保存可视化结果
    if results:
        _generate_visualizations(results, output_dir)
    else:
        logging.error("模拟未返回结果，无法生成可视化。")

if __name__ == "__main__":
    main()