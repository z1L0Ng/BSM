"""实现论文中的联合优化模型。"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import logging
try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    print("Gurobi未安装，将使用启发式算法")

# 导入时使用相对路径，运行时使用绝对路径
try:
    from .state_transition import ETaxiStateModel, BSSStateModel
except ImportError:
    from models.state_transition import ETaxiStateModel, BSSStateModel

class JointOptimizationModel:
    """
    联合优化模型，实现论文中的公式(11)-(12)。
    结合出租车调度和电池交换站充电的整体优化。
    """
    
    def __init__(self, config: Dict):
        """
        初始化联合优化模型。
        
        参数:
            config (dict): 优化配置参数
        """
        # 模型参数
        self.m = config['m_areas']  # 区域数量
        self.L = config['L_energy_levels']  # 能量等级数量
        self.T = config['T_periods']  # 时间段数量
        
        # 目标函数权重
        self.beta = config.get('beta', -0.1)  # 空驶距离权重
        
        # 约束参数
        self.demand = config['demand']  # 需求矩阵 D^t_i
        self.reachability = config['reachability']  # 可达性矩阵 ν^t_{i,i'}
        self.distance_weights = config.get('distance_weights', None)  # 距离权重 ω_{i,i'}
        
        # 状态模型
        self.taxi_model = ETaxiStateModel(self.m, self.L, self.T)
        self.bss_models = {}  # 换电站模型字典
        
        # 优化器设置
        self.solver_config = config.get('solver', {})
        self.time_limit = self.solver_config.get('time_limit', 300)
        self.mip_gap = self.solver_config.get('mip_gap', 0.05)
        
        # 日志配置
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def add_bss_station(self, station_config: Dict):
        """
        添加电池交换站。
        
        参数:
            station_config (dict): 站点配置
        """
        station_id = station_config['id']
        self.bss_models[station_id] = BSSStateModel(
            station_id=station_id,
            location=station_config['location'],
            capacity=station_config['capacity'],
            chargers=station_config['chargers'],
            T_periods=self.T,
            L_energy_levels=self.L
        )
        
        self.logger.info(f"添加换电站 {station_id} 于区域 {station_config['location']}")
    
    def set_initial_state(self, taxi_state: Dict, bss_states: Dict):
        """
        设置初始状态。
        
        参数:
            taxi_state (dict): 出租车初始状态
            bss_states (dict): 换电站初始状态
        """
        # 设置出租车初始状态
        self.taxi_model.set_initial_state(
            taxi_state['vacant_taxis'],
            taxi_state['occupied_taxis']
        )
        
        # 设置换电站初始状态
        for station_id, inventory in bss_states.items():
            if station_id in self.bss_models:
                self.bss_models[station_id].set_initial_inventory(inventory)
    
    def solve_gurobi(self, current_state: Dict) -> Optional[Dict]:
        """
        使用Gurobi求解优化问题。
        
        参数:
            current_state (dict): 当前系统状态
        
        返回:
            dict: 优化解，包含调度决策
        """
        if not GUROBI_AVAILABLE:
            return None
        
        try:
            # 创建模型
            model = gp.Model("ETaxi_Joint_Optimization")
            model.setParam('OutputFlag', 1)
            model.setParam('TimeLimit', self.time_limit)
            model.setParam('MIPGap', self.mip_gap)
            
            # 决策变量
            # X^{t,l}_{i,i'}: 乘客服务调度
            X = model.addVars(
                self.T, self.m, self.m, self.L,
                vtype=GRB.INTEGER,
                name="X"
            )
            
            # Y^{t,l}_{i,i'}: 换电调度
            Y = model.addVars(
                self.T, self.m, self.m, self.L,
                vtype=GRB.INTEGER,
                name="Y"
            )
            
            # S^t_{i,l}: 服务乘客的出租车数量
            S = model.addVars(
                self.T, self.m, self.L,
                vtype=GRB.INTEGER,
                name="S"
            )
            
            # 服务的乘客数量
            served_passengers = model.addVars(
                self.T, self.m,
                vtype=GRB.INTEGER,
                name="served"
            )
            
            # 约束条件
            
            # 1. 服务乘客数量约束 (公式8)
            for t in range(self.T):
                for i in range(self.m):
                    total_supply = gp.quicksum(S[t, i, l] for l in range(self.L))
                    demand_i = self.demand.get((t, i), 0)
                    
                    # 服务乘客数不能超过供给和需求的最小值
                    model.addConstr(served_passengers[t, i] <= total_supply)
                    model.addConstr(served_passengers[t, i] <= demand_i)
            
            # 2. 状态转移约束 (公式1)
            for t in range(self.T):
                for i in range(self.m):
                    for l in range(self.L):
                        # B^t_{i,l} = Σ_{i'} Y^{t,l}_{i',i}
                        swapping_taxis = gp.quicksum(Y[t, i_prime, i, l] for i_prime in range(self.m))
                        
                        # S^t_{i,l} = Σ_{i'} X^{t,l}_{i',i}
                        model.addConstr(S[t, i, l] == gp.quicksum(X[t, i_prime, i, l] for i_prime in range(self.m)))
            
            # 3. 可达性约束 (公式10)
            for t in range(self.T):
                for i in range(self.m):
                    for i_prime in range(self.m):
                        for l in range(self.L):
                            reachable = self.reachability.get((t, i, i_prime), True)
                            if not reachable:
                                model.addConstr(X[t, i, i_prime, l] == 0)
                                model.addConstr(Y[t, i, i_prime, l] == 0)
            
            # 4. 车辆数量平衡约束
            for t in range(self.T):
                for i in range(self.m):
                    for l in range(self.L):
                        # 当前可用的空闲出租车数量
                        available_vacant = current_state['vacant_taxis'][t, i, l]
                        
                        # 调度出去的出租车总数不能超过可用数量
                        total_dispatched = (gp.quicksum(X[t, i, i_prime, l] for i_prime in range(self.m)) +
                                          gp.quicksum(Y[t, i, i_prime, l] for i_prime in range(self.m)))
                        
                        model.addConstr(total_dispatched <= available_vacant)
            
            # 目标函数 (公式11)
            
            # 服务质量项 J_service
            service_quality = gp.quicksum(
                served_passengers[t, i] 
                for t in range(self.T) 
                for i in range(self.m)
            )
            
            # 空驶距离项 J_idle
            idle_distance = 0
            if self.distance_weights is not None:
                idle_distance = gp.quicksum(
                    self.distance_weights.get((i, i_prime), 1.0) * (X[t, i, i_prime, l] + Y[t, i, i_prime, l])
                    for t in range(self.T)
                    for i in range(self.m)
                    for i_prime in range(self.m)
                    for l in range(self.L)
                )
            
            # 总目标函数
            objective = service_quality + self.beta * idle_distance
            model.setObjective(objective, GRB.MAXIMIZE)
            
            # 求解
            model.optimize()
            
            # 检查求解状态
            if model.status == GRB.OPTIMAL:
                self.logger.info("找到最优解")
            elif model.status == GRB.TIME_LIMIT:
                self.logger.info("达到时间限制，返回当前最优解")
            else:
                self.logger.warning(f"求解失败，状态: {model.status}")
                return None
            
            # 提取解
            solution = {
                'passenger_dispatch': np.zeros((self.T, self.m, self.m, self.L)),
                'swap_dispatch': np.zeros((self.T, self.m, self.m, self.L)),
                'service_taxis': np.zeros((self.T, self.m, self.L)),
                'served_passengers': np.zeros((self.T, self.m)),
                'objective_value': model.objVal,
                'service_quality': service_quality.getValue(),
                'idle_distance': idle_distance.getValue() if idle_distance != 0 else 0
            }
            
            # 提取决策变量值
            for t in range(self.T):
                for i in range(self.m):
                    solution['served_passengers'][t, i] = served_passengers[t, i].x
                    for l in range(self.L):
                        solution['service_taxis'][t, i, l] = S[t, i, l].x
                        for i_prime in range(self.m):
                            solution['passenger_dispatch'][t, i, i_prime, l] = X[t, i, i_prime, l].x
                            solution['swap_dispatch'][t, i, i_prime, l] = Y[t, i, i_prime, l].x
            
            return solution
            
        except Exception as e:
            self.logger.error(f"Gurobi求解过程中出错: {str(e)}")
            return None
    
    def solve_heuristic(self, current_state: Dict) -> Dict:
        """
        使用启发式算法求解优化问题。
        
        参数:
            current_state (dict): 当前系统状态
        
        返回:
            dict: 启发式解
        """
        self.logger.info("使用启发式算法求解")
        
        solution = {
            'passenger_dispatch': np.zeros((self.T, self.m, self.m, self.L)),
            'swap_dispatch': np.zeros((self.T, self.m, self.m, self.L)),
            'service_taxis': np.zeros((self.T, self.m, self.L)),
            'served_passengers': np.zeros((self.T, self.m)),
            'objective_value': 0,
            'service_quality': 0,
            'idle_distance': 0
        }
        
        for t in range(self.T):
            # 1. 换电调度：为低电量出租车安排换电
            self._heuristic_swap_dispatch(t, current_state, solution)
            
            # 2. 乘客服务调度：为剩余出租车安排乘客服务
            self._heuristic_passenger_dispatch(t, current_state, solution)
            
            # 3. 更新状态
            self._update_state_heuristic(t, current_state, solution)
        
        # 计算目标函数值
        solution['objective_value'] = solution['service_quality'] + self.beta * solution['idle_distance']
        
        return solution
    
    def _heuristic_swap_dispatch(self, t: int, current_state: Dict, solution: Dict):
        """启发式换电调度。"""
        low_battery_threshold = 2  # 低电量阈值
        
        for i in range(self.m):
            for l in range(low_battery_threshold):  # 只考虑低电量车辆
                available_taxis = current_state['vacant_taxis'][t, i, l]
                
                if available_taxis > 0:
                    # 寻找最近的有电池库存的换电站
                    best_station_area = self._find_best_swap_station(i, t)
                    
                    if best_station_area is not None:
                        # 分配换电任务
                        dispatch_count = min(available_taxis, 3)  # 限制每次最多3辆车
                        solution['swap_dispatch'][t, i, best_station_area, l] = dispatch_count
                        
                        # 更新可用车辆数
                        current_state['vacant_taxis'][t, i, l] -= dispatch_count
    
    def _heuristic_passenger_dispatch(self, t: int, current_state: Dict, solution: Dict):
        """启发式乘客服务调度。"""
        for i in range(self.m):
            demand_i = self.demand.get((t, i), 0)
            
            if demand_i > 0:
                # 优先使用高电量车辆服务乘客
                total_assigned = 0
                
                for l in range(self.L-1, -1, -1):  # 从高电量到低电量
                    available_taxis = current_state['vacant_taxis'][t, i, l]
                    
                    if available_taxis > 0 and total_assigned < demand_i:
                        # 本地服务（不需要移动）
                        local_assignment = min(available_taxis, demand_i - total_assigned)
                        solution['passenger_dispatch'][t, i, i, l] = local_assignment
                        solution['service_taxis'][t, i, l] += local_assignment
                        total_assigned += local_assignment
                        
                        # 更新可用车辆数
                        current_state['vacant_taxis'][t, i, l] -= local_assignment
                
                # 如果本地车辆不足，从邻近区域调度
                if total_assigned < demand_i:
                    self._dispatch_from_nearby_areas(t, i, demand_i - total_assigned, 
                                                   current_state, solution)
                
                solution['served_passengers'][t, i] = total_assigned
                solution['service_quality'] += total_assigned
    
    def _dispatch_from_nearby_areas(self, t: int, target_area: int, remaining_demand: int, 
                                  current_state: Dict, solution: Dict):
        """从邻近区域调度车辆。"""
        dispatched = 0
        
        # 按距离排序邻近区域
        area_distances = []
        for i in range(self.m):
            if i != target_area:
                distance = self.distance_weights.get((i, target_area), float('inf'))
                if distance < float('inf') and self.reachability.get((t, i, target_area), True):
                    area_distances.append((i, distance))
        
        area_distances.sort(key=lambda x: x[1])
        
        # 从最近的区域开始调度
        for source_area, distance in area_distances:
            if dispatched >= remaining_demand:
                break
            
            for l in range(self.L-1, -1, -1):  # 优先使用高电量车辆
                available_taxis = current_state['vacant_taxis'][t, source_area, l]
                
                if available_taxis > 0:
                    dispatch_count = min(available_taxis, remaining_demand - dispatched)
                    
                    # 记录调度决策
                    solution['passenger_dispatch'][t, source_area, target_area, l] = dispatch_count
                    solution['service_taxis'][t, target_area, l] += dispatch_count
                    
                    # 更新状态
                    current_state['vacant_taxis'][t, source_area, l] -= dispatch_count
                    dispatched += dispatch_count
                    
                    # 累计空驶距离
                    solution['idle_distance'] += dispatch_count * distance
                    
                    if dispatched >= remaining_demand:
                        break
    
    def _find_best_swap_station(self, area: int, t: int) -> Optional[int]:
        """寻找最佳换电站。"""
        best_station = None
        best_score = -1
        
        for station_id, bss_model in self.bss_models.items():
            station_area = bss_model.location
            
            # 检查可达性
            if not self.reachability.get((t, area, station_area), True):
                continue
            
            # 检查电池库存
            total_batteries = np.sum(bss_model.battery_inventory[t])
            if total_batteries <= 0:
                continue
            
            # 计算评分（考虑距离和库存）
            distance = self.distance_weights.get((area, station_area), 1.0)
            score = total_batteries / (1.0 + distance)
            
            if score > best_score:
                best_score = score
                best_station = station_area
        
        return best_station
    
    def _update_state_heuristic(self, t: int, current_state: Dict, solution: Dict):
        """更新启发式算法的状态。"""
        # 这里可以添加状态更新逻辑
        # 例如更新出租车位置、电池消耗等
        pass
    
    def solve(self, current_state: Dict, method: str = 'auto') -> Dict:
        """
        求解优化问题。
        
        参数:
            current_state (dict): 当前系统状态
            method (str): 求解方法 ('gurobi', 'heuristic', 'auto')
        
        返回:
            dict: 优化解
        """
        if method == 'auto':
            method = 'gurobi' if GUROBI_AVAILABLE else 'heuristic'
        
        self.logger.info(f"使用 {method} 方法求解优化问题")
        
        if method == 'gurobi' and GUROBI_AVAILABLE:
            solution = self.solve_gurobi(current_state)
            if solution is None:
                self.logger.warning("Gurobi求解失败，转为启发式算法")
                solution = self.solve_heuristic(current_state)
        else:
            solution = self.solve_heuristic(current_state)
        
        # 记录求解结果
        self.logger.info(f"求解完成 - 目标值: {solution['objective_value']:.2f}, "
                        f"服务质量: {solution['service_quality']:.2f}, "
                        f"空驶距离: {solution['idle_distance']:.2f}")
        
        return solution
    
    def validate_solution(self, solution: Dict, current_state: Dict) -> bool:
        """
        验证解的可行性。
        
        参数:
            solution (dict): 优化解
            current_state (dict): 当前状态
        
        返回:
            bool: 解是否可行
        """
        try:
            # 1. 检查车辆数量平衡
            for t in range(self.T):
                for i in range(self.m):
                    for l in range(self.L):
                        available = current_state['vacant_taxis'][t, i, l]
                        dispatched = (np.sum(solution['passenger_dispatch'][t, i, :, l]) + 
                                    np.sum(solution['swap_dispatch'][t, i, :, l]))
                        
                        if dispatched > available + 1e-6:  # 允许小的数值误差
                            self.logger.warning(f"车辆数量不平衡: t={t}, i={i}, l={l}")
                            return False
            
            # 2. 检查可达性约束
            for t in range(self.T):
                for i in range(self.m):
                    for i_prime in range(self.m):
                        if not self.reachability.get((t, i, i_prime), True):
                            for l in range(self.L):
                                if (solution['passenger_dispatch'][t, i, i_prime, l] > 1e-6 or
                                    solution['swap_dispatch'][t, i, i_prime, l] > 1e-6):
                                    self.logger.warning(f"违反可达性约束: t={t}, i={i}, i'={i_prime}")
                                    return False
            
            # 3. 检查服务乘客数量约束
            for t in range(self.T):
                for i in range(self.m):
                    served = solution['served_passengers'][t, i]
                    supply = np.sum(solution['service_taxis'][t, i, :])
                    demand = self.demand.get((t, i), 0)
                    
                    if served > supply + 1e-6 or served > demand + 1e-6:
                        self.logger.warning(f"服务约束违反: t={t}, i={i}")
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"解验证过程中出错: {str(e)}")
            return False
    
    def get_optimization_statistics(self, solution: Dict) -> Dict:
        """
        计算优化统计信息。
        
        参数:
            solution (dict): 优化解
        
        返回:
            dict: 统计信息
        """
        stats = {
            'total_served_passengers': np.sum(solution['served_passengers']),
            'total_swaps': np.sum(solution['swap_dispatch']),
            'total_relocations': np.sum(solution['passenger_dispatch']),
            'avg_service_rate': 0,
            'avg_utilization': 0,
            'peak_demand_period': 0,
            'peak_demand_area': 0
        }
        
        # 计算平均服务率
        total_demand = sum(self.demand.values())
        if total_demand > 0:
            stats['avg_service_rate'] = stats['total_served_passengers'] / total_demand
        
        # 找出高峰需求时段和区域
        max_demand = 0
        for (t, i), demand in self.demand.items():
            if demand > max_demand:
                max_demand = demand
                stats['peak_demand_period'] = t
                stats['peak_demand_area'] = i
        
        # 计算利用率统计
        total_service_taxis = np.sum(solution['service_taxis'])
        total_available_periods = self.T * self.m * self.L
        if total_available_periods > 0:
            stats['avg_utilization'] = total_service_taxis / total_available_periods
        
        return stats

if __name__ == "__main__":
    # 测试联合优化模型
    print("测试联合优化模型...")
    
    # 创建测试配置
    config = {
        'm_areas': 3,
        'L_energy_levels': 5,
        'T_periods': 4,
        'beta': -0.1,
        'demand': {(0, 0): 10, (0, 1): 8, (1, 0): 6, (1, 1): 12},
        'reachability': {},  # 默认全部可达
        'distance_weights': {(0, 1): 1.0, (1, 0): 1.0, (0, 2): 2.0, (2, 0): 2.0, (1, 2): 1.5, (2, 1): 1.5},
        'solver': {'time_limit': 60, 'mip_gap': 0.05}
    }
    
    # 创建优化模型
    optimizer = JointOptimizationModel(config)
    
    # 添加换电站
    optimizer.add_bss_station({
        'id': 1,
        'location': 1,
        'capacity': 20,
        'chargers': 5
    })
    
    # 创建测试状态
    np.random.seed(42)
    current_state = {
        'vacant_taxis': np.random.randint(0, 10, (config['T_periods'], config['m_areas'], config['L_energy_levels'])),
        'occupied_taxis': np.random.randint(0, 5, (config['T_periods'], config['m_areas'], config['L_energy_levels']))
    }
    
    # 求解
    solution = optimizer.solve(current_state, method='heuristic')
    
    # 验证解
    is_valid = optimizer.validate_solution(solution, current_state)
    print(f"解的可行性: {is_valid}")
    
    # 获取统计信息
    stats = optimizer.get_optimization_statistics(solution)
    print(f"优化统计: {stats}")
    
    print("联合优化模型测试完成!")