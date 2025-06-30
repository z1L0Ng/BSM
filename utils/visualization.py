"""可视化工具，用于图形化展示模拟结果。"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.animation import FuncAnimation
import folium
from folium.plugins import HeatMap

# --- 新增代码：解决中文显示问题 ---
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为 SimHei
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
# ------------------------------------


def plot_station_metrics(results, save_path=None):
    """
    绘制站点相关的指标图表。
    
    参数:
        results (dict): 模拟结果字典
        save_path (str, optional): 保存图像的路径
    """
    station_stats = results['station_stats']
    
    # 创建一个2x2的子图网格
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('换电站性能指标', fontsize=16)
    
    # 1. 每个站点的电池更换次数
    ax1 = axes[0, 0]
    station_ids = list(station_stats['station_swap_counts'].keys())
    swap_counts = [station_stats['station_swap_counts'][i] for i in station_ids]
    
    # 生成站点ID的标签
    labels = [f"站点 {sid}" for sid in station_ids]
    
    # 创建条形图
    bars = ax1.bar(range(len(station_ids)), swap_counts, color='skyblue')
    ax1.set_xticks(range(len(station_ids)))
    ax1.set_xticklabels(labels, rotation=45)
    ax1.set_title('每个站点的电池更换次数')
    ax1.set_ylabel('更换次数')
    
    # 在条形顶部添加数值标签
    for bar, count in zip(bars, swap_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{count}', ha='center', va='bottom')
    
    # 2. 每个站点的平均等待时间
    ax2 = axes[0, 1]
    wait_times = [station_stats['station_avg_wait'][i] for i in station_ids]
    
    bars = ax2.bar(range(len(station_ids)), wait_times, color='salmon')
    ax2.set_xticks(range(len(station_ids)))
    ax2.set_xticklabels(labels, rotation=45)
    ax2.set_title('每个站点的平均等待时间')
    ax2.set_ylabel('等待时间 (分钟)')
    
    for bar, time in zip(bars, wait_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{time:.2f}', ha='center', va='bottom')
    
    # 3. 各站点剩余已充电电池的分布
    ax3 = axes[1, 0]
    charged_batteries = [station_stats['charged_batteries'][i] for i in station_ids]
    empty_batteries = [station_stats['empty_batteries'][i] for i in station_ids]
    
    # 堆叠条形图
    width = 0.35
    ax3.bar(range(len(station_ids)), charged_batteries, width, label='已充电电池', color='green')
    ax3.bar(range(len(station_ids)), empty_batteries, width, bottom=charged_batteries, 
           label='待充电电池', color='orange')
    
    ax3.set_xticks(range(len(station_ids)))
    ax3.set_xticklabels(labels, rotation=45)
    ax3.set_title('模拟结束时各站点电池状态')
    ax3.set_ylabel('电池数量')
    ax3.legend()
    
    # 4. 站点使用率饼图
    ax4 = axes[1, 1]
    usage_percentage = [count / sum(swap_counts) * 100 if sum(swap_counts) > 0 else 0 for count in swap_counts]

    
    # 只显示使用率大于等于1%的站点
    significant_indices = [i for i, percentage in enumerate(usage_percentage) if percentage >= 1]
    significant_labels = [labels[i] for i in significant_indices]
    significant_usage = [usage_percentage[i] for i in significant_indices]
    
    # 如果有使用率小于1%的站点，将它们合并为"其他"
    if len(significant_indices) < len(usage_percentage):
        other_percentage = 100 - sum(significant_usage)
        significant_labels.append('其他')
        significant_usage.append(other_percentage)
    
    # 绘制饼图
    wedges, texts, autotexts = ax4.pie(significant_usage, autopct='%1.1f%%', 
                                      startangle=90, shadow=True)
    ax4.axis('equal')  # 确保饼图是圆形的
    ax4.set_title('站点使用率分布')
    ax4.legend(wedges, significant_labels, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # 保存图像
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_taxi_metrics(results, save_path=None):
    """
    绘制出租车相关的指标图表。
    """
    taxi_stats = results['taxi_stats']
    performance = results['performance_metrics']
    
    # 创建一个2x2的子图网格
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('出租车性能指标', fontsize=16)
    
    # 1. 出租车行程分布直方图
    ax1 = axes[0, 0]
    trips = list(taxi_stats['trips_per_taxi'].values())
    
    # 绘制直方图
    n, bins, patches = ax1.hist(trips, bins=20, color='skyblue', edgecolor='black')
    
    # 添加平均线
    ax1.axvline(x=performance['avg_trips_per_taxi'], color='red', linestyle='--', 
                label=f'平均: {performance["avg_trips_per_taxi"]:.2f}')
    
    ax1.set_title('出租车行程分布')
    ax1.set_xlabel('行程数量')
    ax1.set_ylabel('出租车数量')
    ax1.legend()
    
    # 2. 出租车收入分布直方图
    ax2 = axes[0, 1]
    revenues = list(taxi_stats['revenue_per_taxi'].values())
    
    n, bins, patches = ax2.hist(revenues, bins=20, color='salmon', edgecolor='black')
    
    # 添加平均收入线
    avg_revenue = sum(revenues) / len(revenues) if revenues else 0
    ax2.axvline(x=avg_revenue, color='red', linestyle='--', 
                label=f'平均: ${avg_revenue:.2f}')
    
    ax2.set_title('出租车收入分布')
    ax2.set_xlabel('收入 ($)')
    ax2.set_ylabel('出租车数量')
    ax2.legend()
    
    # 3. 行程与换电次数的散点图
    ax3 = axes[1, 0]
    
    # 提取所有出租车的行程数和换电次数
    taxi_ids = list(taxi_stats['trips_per_taxi'].keys())
    trips_list = [taxi_stats['trips_per_taxi'][tid] for tid in taxi_ids]
    swaps_list = [taxi_stats['swap_count'][tid] for tid in taxi_ids]
    
    # 计算每次换电平均完成的行程数
    trips_per_swap = [t/s if s > 0 else 0 for t, s in zip(trips_list, swaps_list)]
    
    # 绘制散点图
    ax3.scatter(trips_list, swaps_list, alpha=0.5)
    
    # 添加回归线
    if len(trips_list) > 1:
        z = np.polyfit(trips_list, swaps_list, 1)
        p = np.poly1d(z)
        x_range = np.linspace(min(trips_list), max(trips_list), 100)
        ax3.plot(x_range, p(x_range), "r--", label=f'趋势线: y = {z[0]:.2f}x + {z[1]:.2f}')
    
    ax3.set_title('行程数量与换电次数关系')
    ax3.set_xlabel('行程数量')
    ax3.set_ylabel('换电次数')
    ax3.legend()
    
    # 4. 行程数与收入的散点图
    ax4 = axes[1, 1]
    
    # 提取所有出租车的行程数和收入
    revenues = [taxi_stats['revenue_per_taxi'][tid] for tid in taxi_ids]
    
    # 绘制散点图
    ax4.scatter(trips_list, revenues, alpha=0.5)
    
    # 添加回归线
    if len(trips_list) > 1:
        z = np.polyfit(trips_list, revenues, 1)
        p = np.poly1d(z)
        x_range = np.linspace(min(trips_list), max(trips_list), 100) # 复用x_range
        ax4.plot(x_range, p(x_range), "r--", label=f'趋势线: y = {z[0]:.2f}x + {z[1]:.2f}')
    
    ax4.set_title('行程数量与收入关系')
    ax4.set_xlabel('行程数量')
    ax4.set_ylabel('收入 ($)')
    ax4.legend()
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # 保存图像
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

# ... (文件中剩余的其他函数 create_nyc_map, create_animation 保持不变) ...
def create_nyc_map(results, block_positions, latitude_offset=40.5, longitude_offset=-74.05, scale=0.01):
    """
    在纽约地图上可视化模拟结果。
    
    参数:
        results (dict): 模拟结果字典
        block_positions (dict): 区块位置字典
        latitude_offset (float): 纬度偏移
        longitude_offset (float): 经度偏移
        scale (float): 坐标缩放因子
    
    返回:
        folium.Map: 含有可视化结果的Folium地图对象
    """
    # 创建以纽约为中心的地图
    nyc_map = folium.Map(location=[40.7128, -74.0060], zoom_start=12)
    
    # 添加换电站位置
    station_locations = results['station_stats']['station_locations']
    station_swap_counts = results['station_stats']['station_swap_counts']
    
    # 创建站点图层
    station_layer = folium.FeatureGroup(name="换电站")
    
    for station_id, block_id in station_locations.items():
        if block_id in block_positions:
            x, y = block_positions[block_id]
            # 将坐标转换回经纬度
            lat = y * scale + latitude_offset
            lon = x * scale + longitude_offset
            
            swap_count = station_swap_counts.get(station_id, 0)
            
            # 创建站点标记，大小和颜色基于换电次数
            radius = max(10, min(30, swap_count / 5))  # 限制半径在10-30之间
            
            # 创建圆形标记
            folium.CircleMarker(
                location=[lat, lon],
                radius=radius,
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.6,
                popup=f"站点 {station_id}: {swap_count} 次换电"
            ).add_to(station_layer)
    
    # 添加站点图层到地图
    station_layer.add_to(nyc_map)
    
    # 添加热力图显示行程密度
    if 'final_locations' in results['taxi_stats']:
        # 提取出租车最终位置
        taxi_locations = []
        
        for taxi_id, block_id in results['taxi_stats']['final_locations'].items():
            if block_id in block_positions:
                x, y = block_positions[block_id]
                # 将坐标转换回经纬度
                lat = y * scale + latitude_offset
                lon = x * scale + longitude_offset
                taxi_locations.append([lat, lon])
        
        # 创建热力图
        if taxi_locations:
            heat_layer = folium.FeatureGroup(name="出租车分布热图")
            HeatMap(taxi_locations).add_to(heat_layer)
            heat_layer.add_to(nyc_map)
    
    # 添加图层控制
    folium.LayerControl().add_to(nyc_map)
    
    return nyc_map

def create_animation(simulation_data, block_positions, save_path=None):
    """
    创建模拟动画，显示出租车和站点状态的变化。
    """
    # 创建图表和坐标轴
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 函数用于更新每一帧
    def update(frame):
        ax.clear()
        
        # 获取当前帧的数据
        frame_data = simulation_data[frame]
        
        # 绘制区块
        for block_id, (x, y) in block_positions.items():
            ax.scatter(x, y, c='lightgray', s=5, alpha=0.3)
        
        # 绘制站点
        for station in frame_data['stations']:
            x, y = block_positions[station['location']]
            # 根据已充电电池比例确定颜色
            charge_ratio = station['charged_batteries'] / station['total_capacity']
            color = plt.cm.RdYlGn(charge_ratio)  # 红-黄-绿色映射
            
            # 绘制站点
            ax.scatter(x, y, c=[color], s=100, edgecolor='black', zorder=3)
            ax.annotate(f"S{station['id']}", (x, y), 
                       textcoords="offset points", xytext=(0, 5), 
                       ha='center', fontsize=8)
        
        # 绘制出租车
        for taxi in frame_data['taxis']:
            x, y = block_positions[taxi['location']]
            
            # 根据出租车状态确定颜色
            if taxi['state'] == '待命':
                color = 'blue'
            elif taxi['state'] == '服务中':
                color = 'green'
            elif taxi['state'] in ['前往换电', '换电中']:
                color = 'red'
            else:
                color = 'gray'
            
            # 绘制出租车
            ax.scatter(x, y, c=color, s=20, alpha=0.7, zorder=2)
        
        # 设置标题和时间
        ax.set_title(f"E-出租车模拟 - 时间: {frame_data['time']:.1f}分钟")
        
        # 添加颜色图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='待命出租车'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='服务中出租车'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='换电中出租车'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', markersize=8, label='区块')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # 添加状态信息
        status_text = (
            f"总出租车: {len(frame_data['taxis'])}\n"
            f"总行程: {frame_data['total_trips']}\n"
            f"总换电: {frame_data['total_swaps']}"
        )
        ax.text(0.02, 0.97, status_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # 设置坐标轴范围
        ax.set_xlim(min(x for x, _ in block_positions.values()) - 5, 
                   max(x for x, _ in block_positions.values()) + 5)
        ax.set_ylim(min(y for _, y in block_positions.values()) - 5, 
                   max(y for _, y in block_positions.values()) + 5)
        
        return ax,
    
    # 创建动画
    ani = FuncAnimation(fig, update, frames=len(simulation_data), 
                        interval=200, blit=False)
    
    # 保存动画
    if save_path:
        ani.save(save_path, writer='pillow', fps=10)
    
    plt.tight_layout()
    plt.show()
    
    return ani