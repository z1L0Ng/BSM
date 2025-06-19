"""结果分析工具模块。"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_performance(results: Dict) -> Dict:
    """
    分析模拟性能。
    
    参数:
        results (dict): 模拟结果
    
    返回:
        dict: 性能分析结果
    """
    analysis = {
        'service_quality': {},
        'operational_efficiency': {},
        'optimization_performance': {},
        'recommendations': []
    }
    
    # 服务质量分析
    history = results.get('simulation_history', [])
    if history:
        total_served = sum(r['metrics']['total_served_passengers'] for r in history)
        avg_utilization = np.mean([r['metrics']['utilization_rate'] for r in history])
        
        analysis['service_quality'] = {
            'total_passengers_served': total_served,
            'avg_utilization_rate': avg_utilization,
            'service_consistency': np.std([r['metrics']['utilization_rate'] for r in history]),
            'peak_utilization': max(r['metrics']['utilization_rate'] for r in history),
            'min_utilization': min(r['metrics']['utilization_rate'] for r in history)
        }
    
    # 运营效率分析
    if 'final_metrics' in results:
        metrics = results['final_metrics']
        analysis['operational_efficiency'] = {
            'service_efficiency': metrics.get('service_efficiency', 0),
            'system_efficiency': metrics.get('system_efficiency', 0),
            'avg_wait_time': calculate_avg_wait_time(results),
            'station_utilization': calculate_station_utilization(results)
        }
    
    # 优化性能分析
    if history:
        opt_times = [r['optimization_time'] for r in history]
        analysis['optimization_performance'] = {
            'avg_optimization_time': np.mean(opt_times),
            'max_optimization_time': max(opt_times),
            'optimization_stability': np.std(opt_times),
            'total_optimization_time': sum(opt_times)
        }
    
    # 生成建议
    analysis['recommendations'] = generate_recommendations(analysis)
    
    return analysis

def calculate_avg_wait_time(results: Dict) -> float:
    """计算平均等待时间。"""
    # 简化计算，实际应该从详细的等待时间数据中计算
    if 'station_stats' in results:
        wait_times = results['station_stats'].get('station_avg_wait', {})
        if wait_times:
            return np.mean(list(wait_times.values()))
    return 0.0

def calculate_station_utilization(results: Dict) -> Dict:
    """计算站点利用率。"""
    utilization = {}
    
    if 'stations' in results:
        for station in results['stations']:
            station_id = station['id']
            # 简化计算
            utilization[station_id] = 0.75  # 示例值
    
    return utilization

def generate_recommendations(analysis: Dict) -> List[str]:
    """基于分析结果生成建议。"""
    recommendations = []
    
    # 服务质量建议
    sq = analysis['service_quality']
    if sq.get('avg_utilization_rate', 0) < 0.6:
        recommendations.append("车辆利用率较低，建议减少车队规模或增加营销推广")
    elif sq.get('avg_utilization_rate', 0) > 0.85:
        recommendations.append("车辆利用率过高，建议增加车队规模以改善服务质量")
    
    # 运营效率建议
    oe = analysis['operational_efficiency']
    if oe.get('avg_wait_time', 0) > 10:
        recommendations.append("平均等待时间过长，建议增加换电站数量或容量")
    
    # 优化性能建议
    op = analysis['optimization_performance']
    if op.get('avg_optimization_time', 0) > 30:
        recommendations.append("优化求解时间过长，建议使用启发式算法或减少问题规模")
    
    return recommendations

def plot_optimization_results(results: Dict, save_path: Optional[str] = None):
    """
    绘制优化结果图表。
    
    参数:
        results (dict): 模拟结果
        save_path (str, optional): 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('优化结果分析', fontsize=16)
    
    history = results.get('simulation_history', [])
    if not history:
        plt.text(0.5, 0.5, '无历史数据', ha='center', va='center', transform=fig.transFigure)
        if save_path:
            plt.savefig(save_path)
        plt.show()
        return
    
    # 1. 服务乘客数时间序列
    ax1 = axes[0, 0]
    time_periods = [r['time_period'] for r in history]
    passengers_served = [r['metrics']['total_served_passengers'] for r in history]
    
    ax1.plot(time_periods, passengers_served, 'b-', linewidth=2)
    ax1.set_title('每时间段服务乘客数')
    ax1.set_xlabel('时间段')
    ax1.set_ylabel('服务乘客数')
    ax1.grid(True, alpha=0.3)
    
    # 2. 车辆利用率时间序列
    ax2 = axes[0, 1]
    utilization_rates = [r['metrics']['utilization_rate'] for r in history]
    
    ax2.plot(time_periods, utilization_rates, 'g-', linewidth=2)
    ax2.set_title('车辆利用率变化')
    ax2.set_xlabel('时间段')
    ax2.set_ylabel('利用率')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # 3. 优化时间分布
    ax3 = axes[1, 0]
    opt_times = [r['optimization_time'] for r in history]
    
    ax3.hist(opt_times, bins=20, alpha=0.7, color='orange')
    ax3.set_title('优化求解时间分布')
    ax3.set_xlabel('求解时间 (秒)')
    ax3.set_ylabel('频次')
    ax3.grid(True, alpha=0.3)
    
    # 4. 累计性能指标
    ax4 = axes[1, 1]
    cumulative_passengers = np.cumsum(passengers_served)
    
    ax4.plot(time_periods, cumulative_passengers, 'r-', linewidth=2)
    ax4.set_title('累计服务乘客数')
    ax4.set_xlabel('时间段')
    ax4.set_ylabel('累计乘客数')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def compare_scenarios(results_list: List[Dict], scenario_names: List[str], 
                     save_path: Optional[str] = None):
    """
    比较多个场景的结果。
    
    参数:
        results_list (List[Dict]): 多个模拟结果
        scenario_names (List[str]): 场景名称列表
        save_path (str, optional): 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('多场景对比分析', fontsize=16)
    
    # 提取关键指标
    metrics_data = []
    for i, results in enumerate(results_list):
        final_metrics = results.get('final_metrics', {})
        metrics_data.append({
            'scenario': scenario_names[i],
            'total_passengers': final_metrics.get('total_passengers_served', 0),
            'avg_utilization': final_metrics.get('avg_utilization_rate', 0),
            'service_efficiency': final_metrics.get('service_efficiency', 0),
            'avg_opt_time': final_metrics.get('avg_optimization_time', 0)
        })
    
    df = pd.DataFrame(metrics_data)
    
    # 1. 总服务乘客数对比
    ax1 = axes[0, 0]
    bars1 = ax1.bar(df['scenario'], df['total_passengers'], color='skyblue')
    ax1.set_title('总服务乘客数对比')
    ax1.set_ylabel('乘客数')
    ax1.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    # 2. 平均利用率对比
    ax2 = axes[0, 1]
    bars2 = ax2.bar(df['scenario'], df['avg_utilization'], color='lightgreen')
    ax2.set_title('平均利用率对比')
    ax2.set_ylabel('利用率')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='x', rotation=45)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}', ha='center', va='bottom')
    
    # 3. 服务效率对比
    ax3 = axes[1, 0]
    bars3 = ax3.bar(df['scenario'], df['service_efficiency'], color='salmon')
    ax3.set_title('服务效率对比')
    ax3.set_ylabel('乘客/分钟')
    ax3.tick_params(axis='x', rotation=45)
    
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')
    
    # 4. 平均优化时间对比
    ax4 = axes[1, 1]
    bars4 = ax4.bar(df['scenario'], df['avg_opt_time'], color='gold')
    ax4.set_title('平均优化时间对比')
    ax4.set_ylabel('时间 (秒)')
    ax4.tick_params(axis='x', rotation=45)
    
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def export_analysis_report(analysis: Dict, results: Dict, filepath: str):
    """
    导出详细分析报告。
    
    参数:
        analysis (dict): 分析结果
        results (dict): 原始模拟结果
        filepath (str): 报告文件路径
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("电动出租车电池交换系统详细分析报告\n")
        f.write("=" * 60 + "\n\n")
        
        # 1. 执行摘要
        f.write("1. 执行摘要\n")
        f.write("-" * 20 + "\n")
        
        sq = analysis['service_quality']
        f.write(f"总服务乘客数: {sq.get('total_passengers_served', 0)}\n")
        f.write(f"平均车辆利用率: {sq.get('avg_utilization_rate', 0):.2%}\n")
        f.write(f"服务效率: {analysis['operational_efficiency'].get('service_efficiency', 0):.2f} 乘客/分钟\n\n")
        
        # 2. 服务质量分析
        f.write("2. 服务质量分析\n")
        f.write("-" * 20 + "\n")
        
        for key, value in sq.items():
            if isinstance(value, float):
                if 'rate' in key or 'utilization' in key:
                    f.write(f"  {key}: {value:.2%}\n")
                else:
                    f.write(f"  {key}: {value:.2f}\n")
            else:
                f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        # 3. 运营效率分析
        f.write("3. 运营效率分析\n")
        f.write("-" * 20 + "\n")
        
        oe = analysis['operational_efficiency']
        for key, value in oe.items():
            if isinstance(value, dict):
                f.write(f"  {key}:\n")
                for sub_key, sub_value in value.items():
                    f.write(f"    {sub_key}: {sub_value:.2f}\n")
            elif isinstance(value, float):
                f.write(f"  {key}: {value:.2f}\n")
            else:
                f.write(f"  {key}: {value}\n")
        f.write("\n")
        
        # 4. 优化性能分析
        f.write("4. 优化性能分析\n")
        f.write("-" * 20 + "\n")
        
        op = analysis['optimization_performance']
        for key, value in op.items():
            f.write(f"  {key}: {value:.2f} 秒\n")
        f.write("\n")
        
        # 5. 建议和改进措施
        f.write("5. 建议和改进措施\n")
        f.write("-" * 20 + "\n")
        
        for i, recommendation in enumerate(analysis['recommendations'], 1):
            f.write(f"  {i}. {recommendation}\n")
        f.write("\n")
        
        # 6. 配置参数
        f.write("6. 模拟配置参数\n")
        f.write("-" * 20 + "\n")
        
        config = results.get('config', {})
        for key, value in config.items():
            f.write(f"  {key}: {value}\n")
    
    print(f"详细分析报告已保存到: {filepath}")

if __name__ == "__main__":
    # 测试分析功能
    print("测试结果分析模块...")
    
    # 创建模拟的结果数据
    test_results = {
        'simulation_history': [
            {
                'time_period': i,
                'optimization_time': 5.0 + np.random.normal(0, 1),
                'metrics': {
                    'total_served_passengers': 20 + np.random.randint(-5, 5),
                    'utilization_rate': 0.7 + np.random.normal(0, 0.1)
                }
            }
            for i in range(24)
        ],
        'final_metrics': {
            'total_passengers_served': 500,
            'avg_utilization_rate': 0.75,
            'service_efficiency': 2.1,
            'system_efficiency': 1.0,
            'avg_optimization_time': 5.2
        },
        'config': {
            'num_taxis': 500,
            'num_stations': 10,
            'm_areas': 20
        }
    }
    
    # 执行分析
    analysis = analyze_performance(test_results)
    print("分析完成!")
    
    # 生成可视化
    plot_optimization_results(test_results)
    
    print("结果分析模块测试完成!")