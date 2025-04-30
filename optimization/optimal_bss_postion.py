"""电池交换站的最优布局优化。"""
import numpy as np
from scipy.spatial.distance import cdist
try:
    import gurobipy as gp
except ImportError:
    gp = None

def optimize_bss_layout(demand_points, candidate_locations, num_stations, max_distance=None):
    """
    优化电池交换站的布局，以最大化覆盖率或最小化距离。
    
    参数:
        demand_points (dict): 需求点位置 {point_id: (x, y)}
        candidate_locations (dict): 候选站点位置 {loc_id: (x, y)}
        num_stations (int): 要选择的站点数量
        max_distance (float, optional): 最大服务距离(用于覆盖模型)
    
    返回:
        list: 选定的站点ID列表
    """
    if gp is None:
        print("Gurobi未安装。将使用启发式方法进行优化。")
        return greedy_optimization(demand_points, candidate_locations, num_stations)
    
    try:
        # 将字典转换为列表，以便处理
        demand_ids = list(demand_points.keys())
        candidate_ids = list(candidate_locations.keys())
        
        # 计算需求点到候选站点的距离矩阵
        demand_coords = np.array([demand_points[i] for i in demand_ids])
        candidate_coords = np.array([candidate_locations[i] for i in candidate_ids])
        
        distances = cdist(demand_coords, candidate_coords, 'euclidean')
        
        # 创建优化模型
        model = gp.Model("BSS_Optimal_Placement")
        
        # 决策变量: 是否在候选地点j建立站点
        x = model.addVars(len(candidate_ids), vtype=gp.GRB.BINARY, name="x")
        
        # 如果指定了最大距离，使用最大覆盖模型
        if max_distance is not None:
            # 决策变量: 需求点i是否被覆盖
            y = model.addVars(len(demand_ids), vtype=gp.GRB.BINARY, name="y")
            
            # 覆盖约束: 如果需求点i到任何一个站点的距离 <= max_distance，则y[i]=1
            for i in range(len(demand_ids)):
                model.addConstr(
                    gp.quicksum(x[j] for j in range(len(candidate_ids)) 
                              if distances[i,j] <= max_distance) >= y[i]
                )
            
            # 目标: 最大化覆盖的需求点数量
            model.setObjective(gp.quicksum(y), gp.GRB.MAXIMIZE)
        else:
            # 使用p-中位点模型
            # 决策变量: 需求点i是否分配给站点j
            z = model.addVars(len(demand_ids), len(candidate_ids), vtype=gp.GRB.BINARY, name="z")
            
            # 每个需求点必须分配给一个站点
            for i in range(len(demand_ids)):
                model.addConstr(gp.quicksum(z[i,j] for j in range(len(candidate_ids))) == 1)
            
            # 只能分配给已建立的站点
            for i in range(len(demand_ids)):
                for j in range(len(candidate_ids)):
                    model.addConstr(z[i,j] <= x[j])
            
            # 目标: 最小化总距离
            model.setObjective(
                gp.quicksum(distances[i,j] * z[i,j] 
                           for i in range(len(demand_ids)) 
                           for j in range(len(candidate_ids))),
                gp.GRB.MINIMIZE
            )
        
        # 站点数量约束
        model.addConstr(gp.quicksum(x) == num_stations)
        
        # 设置Gurobi参数
        model.setParam('OutputFlag', 1)  # 显示输出
        model.setParam('TimeLimit', 300)  # 设置300秒时间限制
        
        # 解决模型
        model.optimize()
        
        # 检查解决状态
        if model.status == gp.GRB.Status.OPTIMAL or model.status == gp.GRB.Status.TIME_LIMIT:
            # 获取选定的站点
            selected_locations = [candidate_ids[j] for j in range(len(candidate_ids)) 
                                 if x[j].x > 0.5]  # 取值 > 0.5 认为是1
            
            # 打印覆盖率或总距离
            if max_distance is not None:
                coverage = sum(y[i].x for i in range(len(demand_ids)))
                print(f"覆盖的需求点: {coverage}/{len(demand_ids)} ({coverage/len(demand_ids)*100:.1f}%)")
            else:
                total_dist = sum(distances[i,j] * z[i,j].x 
                                for i in range(len(demand_ids)) 
                                for j in range(len(candidate_ids)))
                print(f"总距离: {total_dist:.1f}")
            
            return selected_locations
        else:
            print(f"优化失败，状态: {model.status}. 使用启发式方法。")
            return greedy_optimization(demand_points, candidate_locations, num_stations)
        
    except Exception as e:
        print(f"优化过程中发生错误: {e}")
        print("使用启发式方法代替。")
        return greedy_optimization(demand_points, candidate_locations, num_stations)

def greedy_optimization(demand_points, candidate_locations, num_stations):
    """
    使用贪心算法为换电站找到近似最优的位置。
    
    参数:
        demand_points (dict): 需求点位置 {point_id: (x, y)}
        candidate_locations (dict): 候选站点位置 {loc_id: (x, y)}
        num_stations (int): 要选择的站点数量
    
    返回:
        list: 选定的站点ID列表
    """
    # 将字典转换为列表
    demand_ids = list(demand_points.keys())
    candidate_ids = list(candidate_locations.keys())
    
    # 计算需求点到候选站点的距离矩阵
    demand_coords = np.array([demand_points[i] for i in demand_ids])
    candidate_coords = np.array([candidate_locations[i] for i in candidate_ids])
    
    distances = cdist(demand_coords, candidate_coords, 'euclidean')
    
    # 贪心选择站点
    selected_indices = []
    remaining_indices = list(range(len(candidate_ids)))
    
    # 选择每个需求点的最小距离(初始为无穷大)
    min_distances = np.full(len(demand_ids), np.inf)
    
    for _ in range(min(num_stations, len(candidate_ids))):
        # 计算添加每个候选站点的收益
        max_improvement = -np.inf
        best_idx = -1
        
        for idx in remaining_indices:
            # 计算如果添加此站点，每个需求点的新最小距离
            new_min_distances = np.minimum(min_distances, distances[:, idx])
            
            # 计算总距离改善
            improvement = np.sum(min_distances - new_min_distances)
            
            if improvement > max_improvement:
                max_improvement = improvement
                best_idx = idx
        
        if best_idx != -1:
            # 添加最佳站点
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
            
            # 更新最小距离
            min_distances = np.minimum(min_distances, distances[:, best_idx])
    
    # 转换回原始ID
    selected_locations = [candidate_ids[idx] for idx in selected_indices]
    
    # 打印站点选择信息
    print(f"使用贪心算法选择了 {len(selected_locations)} 个站点")
    print(f"平均最小距离: {np.mean(min_distances):.2f}")
    
    return selected_locations

def calculate_station_capacities(selected_locations, demand_points, distances=None, 
                                min_capacity=10, max_capacity=30):
    """
    根据需求分布计算每个站点的合适容量。
    
    参数:
        selected_locations (list): 选定的站点ID列表
        demand_points (dict): 需求点位置 {point_id: (x, y)}
        distances (ndarray, optional): 预计算的距离矩阵
        min_capacity (int): 最小站点容量
        max_capacity (int): 最大站点容量
    
    返回:
        dict: 站点ID到推荐容量的映射
    """
    # 如果没有提供距离矩阵，计算需求点到选定站点的距离
    if distances is None:
        demand_coords = np.array([demand_points[i] for i in demand_points])
        selected_coords = np.array([demand_points[i] if i in demand_points else (0,0) 
                                  for i in selected_locations])
        distances = cdist(demand_coords, selected_coords, 'euclidean')
    
    # 为每个需求点找到最近的站点
    nearest_station = np.argmin(distances, axis=1)
    
    # 计算分配给每个站点的需求点数量
    demand_counts = np.bincount(nearest_station, minlength=len(selected_locations))
    
    # 计算站点容量(按需求比例缩放，但在最小和最大容量之间)
    total_demand = np.sum(demand_counts)
    capacities = {}
    
    for i, loc_id in enumerate(selected_locations):
        if total_demand > 0:
            # 按需求比例缩放容量
            capacity = min_capacity + (max_capacity - min_capacity) * (demand_counts[i] / total_demand)
            # 取整
            capacities[loc_id] = max(min_capacity, min(max_capacity, int(round(capacity))))
        else:
            # 如果没有需求，使用最小容量
            capacities[loc_id] = min_capacity
    
    return capacities