# utils/visualization.py
"""
用于生成模拟结果可视化图表的工具函数。
"""
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def plot_station_metrics(data: dict, save_path: str):
    """绘制换电站相关指标的图表。"""
    if 'station_stats' not in data:
        print("警告: 可视化数据中缺少 'station_stats'。")
        return

    stats = data['station_stats']
    station_ids = list(stats['station_swap_counts'].keys())
    num_stations = len(station_ids)
    
    if num_stations == 0:
        print("警告: 没有可供可视化的站点数据。")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('换电站性能指标', fontsize=16)

    # 1. 每个站点的电池更换次数
    ax1 = axes[0, 0]
    swap_counts = list(stats['station_swap_counts'].values())
    ax1.bar(station_ids, swap_counts, color='skyblue')
    ax1.set_title('每个站点的电池更换次数')
    ax1.set_ylabel('更换次数')
    ax1.tick_params(axis='x', rotation=45)

    # 2. 每个站点的平均等待时间
    ax2 = axes[0, 1]
    avg_wait = list(stats['station_avg_wait'].values())
    ax2.bar(station_ids, avg_wait, color='salmon')
    ax2.set_title('每个站点的平均等待时间')
    ax2.set_ylabel('等待时间 (分钟)')
    ax2.tick_params(axis='x', rotation=45)

    # 3. 模拟结束时各站点电池状态
    ax3 = axes[1, 0]
    charged = np.array(list(stats['charged_batteries'].values()))
    empty = np.array(list(stats['empty_batteries'].values()))
    ax3.bar(station_ids, charged, color='green', label='已充电池')
    ax3.bar(station_ids, empty, bottom=charged, color='orange', label='待充电池')
    ax3.set_title('模拟结束时各站点电池状态')
    ax3.set_ylabel('电池数量')
    ax3.legend()
    ax3.tick_params(axis='x', rotation=45)

    # 4. 站点使用率分布
    ax4 = axes[1, 1]
    total_swaps = sum(swap_counts)
    if total_swaps > 0:
        usage_percent = [(c / total_swaps) * 100 for c in swap_counts]
        ax4.pie(usage_percent, labels=station_ids, autopct='%1.1f%%', startangle=90)
    ax4.set_title('站点使用率分布')
    ax4.axis('equal')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.close()
    print(f"站点指标图表已保存到: {save_path}")

def plot_taxi_metrics(data: dict, save_path: str):
    """绘制出租车相关指标的图表。"""
    # 此函数可以根据从 simulation_history 收集的数据进行扩展
    print(f"出租车指标图表生成功能待实现。将保存一个空图像于: {save_path}")
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, '出租车指标待实现', ha='center', va='center')
    plt.savefig(save_path)
    plt.close()

def plot_performance_metrics(data: dict, save_path: str):
    """绘制整体性能指标的图表。"""
    if 'performance_metrics' not in data:
        print("警告: 可视化数据中缺少 'performance_metrics'。")
        return
        
    metrics = data['performance_metrics']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    labels = list(metrics.keys())
    values = list(metrics.values())
    
    bars = ax.bar(labels, values, color=['#4e79a7', '#f28e2b', '#e15759'])
    
    ax.set_ylabel('数值')
    ax.set_title('系统整体性能指标')
    
    # 在柱状图上显示数值
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', va='bottom', ha='center')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"性能指标图表已保存到: {save_path}")