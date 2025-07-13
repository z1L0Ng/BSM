# models/optimization_model.py
"""
使用 Gurobi 或启发式算法求解联合优化模型。
"""
import logging
import numpy as np
import copy

try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    gp = None
    GRB = None
    GUROBI_AVAILABLE = False
    print("警告: Gurobi不可用，将使用启发式算法")

class JointOptimizer:
    def __init__(self, config, demand_model, travel_model):
        self.config = config
        self.demand_model = demand_model
        self.travel_model = travel_model
        self.logger = logging.getLogger(__name__)
        self.m_areas = config.m_areas
        self.L_energy_levels = config.L_energy_levels
        self.T_periods = config.T_periods
        self.alpha = config.alpha
        self.beta = config.beta
        self.gamma = config.gamma
        self.delta_t = config.delta_t

    def optimize_single_period(self, t: int, current_state: dict, use_gurobi: bool = False) -> dict:
        self.logger.info(f"开始为时间段 {t} 进行优化...")
        if use_gurobi and GUROBI_AVAILABLE:
            try:
                solution = self.solve_gurobi(current_state, t)
            except Exception as e:
                self.logger.error(f"Gurobi 求解失败: {e}。将自动切换到启发式算法。")
                solution = self.solve_heuristic(current_state, t)
        else:
            if use_gurobi and not GUROBI_AVAILABLE:
                self.logger.warning("请求使用Gurobi，但Gurobi不可用。将使用启发式算法代替。")
            solution = self.solve_heuristic(current_state, t)
            
        if not self.validate_solution(solution, current_state):
            self.logger.error("生成的解决方案无效！将返回一个空的（安全）决策以避免模拟错误。")
            return self._create_empty_solution()

        self.logger.info(f"时间段 {t} 的优化完成。")
        return solution

    def solve_gurobi(self, current_state: dict, t: int) -> dict:
        self.logger.info("使用 Gurobi 求解器构建优化模型...")
        vacant_taxis_initial = current_state['vacant_taxis'][0]
        bss_inventories_initial = current_state['bss_inventories']
        model = gp.Model("etaxi_bss_joint_opt")
        
        X = model.addVars(self.m_areas, self.m_areas, self.L_energy_levels, vtype=GRB.INTEGER, name="X")
        Y = model.addVars(self.m_areas, self.m_areas, self.L_energy_levels, vtype=GRB.INTEGER, name="Y")
        
        J_service = self.alpha * gp.quicksum(X)
        J_idle = self.beta * gp.quicksum(self.travel_model.get_distance(i, j) * (X[i, j, l] + Y[i, j, l]) for i in range(self.m_areas) for j in range(self.m_areas) for l in range(self.L_energy_levels))
        
        model.setObjective(J_service - J_idle, GRB.MAXIMIZE)

        for i in range(self.m_areas):
            for l in range(self.L_energy_levels):
                model.addConstr(gp.quicksum(X[i, j, l] for j in range(self.m_areas)) + gp.quicksum(Y[i, j, l] for j in range(self.m_areas)) <= vacant_taxis_initial[i, l], name=f"taxi_conservation_{i}_{l}")

        demand_t = self.demand_model.get_demand(t)
        for i in range(self.m_areas):
            for j in range(self.m_areas):
                model.addConstr(gp.quicksum(X[i, j, l] for l in range(self.L_energy_levels)) <= demand_t[i, j], name=f"demand_satisfaction_{i}_{j}")

        station_locations = {s['id']: s['location'] for s in self.config.stations}
        for station_id, station_loc in station_locations.items():
            if station_id in bss_inventories_initial:
                available_batteries = bss_inventories_initial[station_id]['charged']
                model.addConstr(gp.quicksum(Y[i, station_loc, l] for i in range(self.m_areas) for l in range(self.L_energy_levels)) <= available_batteries, name=f"bss_capacity_{station_id}")
        
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            solution = self._create_empty_solution()
            for i, j, l in X.keys():
                if X[i, j, l].X > 0.5:
                    solution['passenger_dispatch'][0, i, j, l] = int(X[i, j, l].X)
            for i, j, l in Y.keys():
                if Y[i, j, l].X > 0.5:
                    solution['swap_dispatch'][0, i, j, l] = int(Y[i, j, l].X)
            
            solution['service_quality'] = J_service.getValue()
            solution['idle_distance'] = J_idle.getValue() / self.beta if self.beta != 0 else 0
            solution['objective_value'] = model.objVal
            return solution
        else:
            self.logger.warning("Gurobi 未找到最优解，将使用启发式算法。")
            return self.solve_heuristic(current_state, t)

    def solve_heuristic(self, current_state: dict, t: int) -> dict:
        self.logger.info("正在使用启发式算法求解...")
        state_copy = copy.deepcopy(current_state)
        solution = self._create_empty_solution()
        self._heuristic_swap_dispatch(t, state_copy, solution)
        self._heuristic_passenger_dispatch(t, state_copy, solution)
        self._heuristic_charging_schedule(t, state_copy, solution)
        solution['objective_value'] = (self.alpha * solution['service_quality'] - self.beta * solution['idle_distance'] - self.gamma * solution['charging_cost'])
        return solution
    
    def _heuristic_swap_dispatch(self, t: int, state: dict, solution: dict):
        vacant_taxis = state['vacant_taxis'][0]
        bss_inventories = state['bss_inventories']
        low_energy_threshold = self.L_energy_levels // 2
        station_locations = {s['id']: s['location'] for s in self.config.stations}

        for l in range(low_energy_threshold):
            for i in range(self.m_areas):
                num_taxis_to_swap = int(vacant_taxis[i, l])
                if num_taxis_to_swap == 0:
                    continue
                
                # 【逻辑修正】不再硬性过滤掉没有电池的站，而是给它们一个评分
                # 评分 = w1 * 库存 - w2 * 距离
                # 简化：优先选择库存最多的最近站点
                best_station_id = None
                max_score = -float('inf')
                
                for sid, sloc in station_locations.items():
                    distance = self.travel_model.get_distance(i, sloc)
                    # 避免除以零
                    distance_penalty = distance * 0.1 
                    inventory_bonus = bss_inventories[sid].get('charged', 0)
                    score = inventory_bonus - distance_penalty
                    
                    if score > max_score:
                        max_score = score
                        best_station_id = sid

                if not best_station_id:
                    continue
                
                station_area_idx = station_locations[best_station_id]
                
                # 【逻辑修正】即使库存为0也分配，让出租车去等待
                dispatch_count = num_taxis_to_swap
                
                solution['swap_dispatch'][0, i, station_area_idx, l] += dispatch_count
                distance = self.travel_model.get_distance(i, station_area_idx)
                solution['idle_distance'] += dispatch_count * distance
                
                vacant_taxis[i, l] -= dispatch_count
                # 注意：这里只更新决策，不应假设电池已被取走，模拟器会处理
                # bss_inventories[best_station_id]['charged'] -= dispatch_count
    
    def _heuristic_passenger_dispatch(self, t: int, state: dict, solution: dict):
        vacant_taxis = state['vacant_taxis'][0]
        demand = self.demand_model.get_demand(t).copy()

        for l in range(self.L_energy_levels - 1, -1, -1):
            for i in range(self.m_areas):
                available_taxis = int(vacant_taxis[i, l])
                if available_taxis == 0:
                    continue
                for j in range(self.m_areas):
                    if demand[i, j] > 0:
                        dispatch_count = min(available_taxis, int(demand[i, j]))
                        if dispatch_count == 0:
                            continue
                        travel_energy_cost = self.travel_model.get_energy_consumption(i, j)
                        if l >= travel_energy_cost:
                            solution['passenger_dispatch'][0, i, j, l] += dispatch_count
                            solution['service_quality'] += dispatch_count
                            vacant_taxis[i, l] -= dispatch_count
                            demand[i, j] -= dispatch_count
                            available_taxis -= dispatch_count
                            if available_taxis == 0:
                                break
    
    def _heuristic_charging_schedule(self, t: int, state: dict, solution: dict):
        for station_id, inventory in state['bss_inventories'].items():
            if inventory['empty'] > self.config.bss_capacity // 2:
                solution['charging_schedule'][0] = 1
                solution['charging_cost'] += 50 
                break

    def validate_solution(self, solution: dict, current_state: dict) -> bool:
        passenger_dispatched = np.sum(solution['passenger_dispatch'][0], axis=(1, 2))
        swap_dispatched = np.sum(solution['swap_dispatch'][0], axis=(1, 2))
        total_dispatched = passenger_dispatched + swap_dispatched
        available_taxis = np.sum(current_state['vacant_taxis'][0], axis=1) # 应该按区域检查
        # 修正: 验证应该在 (区域, 电量) 级别上进行
        passenger_dispatched_il = np.sum(solution['passenger_dispatch'][0], axis=1) # (i,l)
        swap_dispatched_il = np.sum(solution['swap_dispatch'][0], axis=1) # (i,l)
        total_dispatched_il = passenger_dispatched_il + swap_dispatched_il
        available_taxis_il = current_state['vacant_taxis'][0]

        if np.any(total_dispatched_il > available_taxis_il + 1e-6):
            self.logger.error("验证失败: 总调度车辆数超过了可用空闲车辆数。")
            return False
        return True

    def _create_empty_solution(self) -> dict:
        t, m, l_ = 1, self.m_areas, self.L_energy_levels
        return {
            'passenger_dispatch': np.zeros((t, m, m, l_), dtype=int),
            'swap_dispatch': np.zeros((t, m, m, l_), dtype=int),
            'charging_schedule': np.zeros(t, dtype=int),
            'service_quality': 0.0,
            'idle_distance': 0.0,
            'charging_cost': 0.0,
            'objective_value': 0.0
        }