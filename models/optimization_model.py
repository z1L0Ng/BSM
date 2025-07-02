# models/optimization_model.py
"""
使用 Gurobi 或启发式算法求解联合优化模型。
"""
import logging
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import copy  # 导入copy模块，用于深拷贝

class JointOptimizer:
    """
    联合优化器，负责为电动出租车和换电站系统计算最优调度和充电策略。
    """
    def __init__(self, config, demand_model, travel_model):
        """
        初始化优化器。

        Args:
            config (SimulationConfig): 包含所有模拟参数的配置对象。
            demand_model: 需求模型实例，用于预测乘客需求。
            travel_model: 出行模型实例，用于计算距离和时间。
        """
        self.config = config
        self.demand_model = demand_model
        self.travel_model = travel_model
        self.logger = logging.getLogger(__name__)
        
        # 从配置中提取常用参数以方便访问
        self.m_areas = config.m_areas
        self.L_energy_levels = config.L_energy_levels
        self.T_periods = config.T_periods
        self.alpha = config.alpha
        self.beta = config.beta
        self.gamma = config.gamma
        self.delta_t = config.delta_t

    def optimize_single_period(self, t: int, current_state: dict, use_gurobi: bool = False) -> dict:
        """
        为单个时间段 t 运行优化。

        这个函数作为主入口，根据 `use_gurobi` 标志选择精确算法或启发式算法。

        Args:
            t (int): 当前模拟的时间步。
            current_state (dict): 包含系统当前状态的字典 (如车辆分布、站点库存)。
            use_gurobi (bool): 如果为True，则尝试使用Gurobi求解器；否则使用启发式算法。

        Returns:
            dict: 包含调度决策和目标函数值的解决方案字典。
        """
        self.logger.info(f"开始为时间段 {t} 进行优化...")
        
        if use_gurobi:
            try:
                # 尝试使用Gurobi进行精确求解
                solution = self.solve_gurobi(current_state)
            except gp.GurobiError as e:
                self.logger.error(f"Gurobi 求解失败，错误代码: {e.errno}。将自动切换到启发式算法。")
                solution = self.solve_heuristic(current_state)
        else:
            # 直接使用启发式算法
            solution = self.solve_heuristic(current_state)
            
        # 在返回决策前，验证其有效性
        if not self.validate_solution(solution, current_state):
            self.logger.error("生成的解决方案无效！将返回一个空的（安全）决策以避免模拟错误。")
            return self._create_empty_solution()

        self.logger.info(f"时间段 {t} 的优化完成。")
        return solution

    def solve_gurobi(self, current_state: dict) -> dict:
        """
        使用 Gurobi 求解完整的混合整数规划（MIP）问题。

        注意: 这是一个框架，实际的MIP模型实现非常复杂，需要精确定义所有
        论文中的变量、约束和目标函数。目前，它会直接调用启发式算法。
        """
        self.logger.info("尝试使用 Gurobi 求解器...")
        self.logger.warning("Gurobi 精确求解器尚未完全实现。将使用启发式算法作为替代。")
        
        # model = gp.Model("etaxi_bss_joint_opt")
        # --- 变量定义 (示例) ---
        # X_tij_l = model.addVars(...)  # 乘客调度
        # Y_tij_l = model.addVars(...)  # 换电调度
        # ...

        # --- 约束定义 ---
        # model.addConstrs(...) # 车辆流动平衡
        # model.addConstrs(...) # 换电站库存平衡
        # ...

        # --- 目标函数 ---
        # model.setObjective(...)

        # model.optimize()

        # 由于未实现，直接返回启发式算法的结果
        return self.solve_heuristic(current_state)

    def solve_heuristic(self, current_state: dict) -> dict:
        """
        使用启发式算法快速求解优化问题。

        启发式算法遵循一组简单的规则来做出决策，速度快但不保证全局最优。
        """
        self.logger.info("正在使用启发式算法求解...")

        # --- 关键修复：创建状态的深拷贝，以避免在算法中修改原始状态数据 ---
        state_copy = copy.deepcopy(current_state)
        
        # 启发式算法通常只关注当前时间步，我们定义为 t=0
        t = 0
        
        # 初始化一个空的解决方案容器
        solution = self._create_empty_solution()

        # 依次执行各个启发式决策模块
        # 1. 换电调度：优先处理低电量车辆
        self._heuristic_swap_dispatch(t, state_copy, solution)
        
        # 2. 乘客分配：优先满足高需求区域
        self._heuristic_passenger_dispatch(t, state_copy, solution)
        
        # 3. 充电计划：根据站点库存决定是否启动充电
        self._heuristic_charging_schedule(t, state_copy, solution)

        # 计算最终的目标函数值
        solution['objective_value'] = (self.alpha * solution['service_quality'] -
                                     self.beta * solution['idle_distance'] -
                                     self.gamma * solution['charging_cost'])
        return solution

    def _heuristic_swap_dispatch(self, t: int, state: dict, solution: dict):
        """启发式换电调度：让电量最低的车辆去最近的、有可用电池的站点。"""
        vacant_taxis = state['vacant_taxis']
        bss_inventories = state['bss_inventories']
        
        # 定义需要换电的电量阈值 (例如，低于总电量的一半)
        low_energy_threshold = self.L_energy_levels // 2
        
        for l in range(low_energy_threshold):
            for i in range(self.m_areas):
                # 获取在区域i，电量为l的空闲车辆数
                num_taxis_to_swap = int(vacant_taxis[t, i, l])
                if num_taxis_to_swap == 0:
                    continue

                # 为这些车辆寻找最近的、有可用电池的换电站
                # (简化逻辑: 按站点ID顺序遍历作为距离代理)
                for station_id in sorted(bss_inventories.keys()):
                    station_area_idx = int(station_id.split('_')[1])
                    
                    # 如果该站有充满的电池
                    if bss_inventories[station_id]['charged'] > 0:
                        # 计算实际可以调度去换电的数量
                        dispatch_count = min(num_taxis_to_swap, bss_inventories[station_id]['charged'])
                        if dispatch_count == 0:
                            continue
                        
                        # 更新解决方案
                        solution['swap_dispatch'][t, i, station_area_idx, l] += dispatch_count
                        distance = self.travel_model.get_distance(i, station_area_idx)
                        solution['idle_distance'] += dispatch_count * distance
                        
                        # 更新状态拷贝，为后续决策提供依据
                        vacant_taxis[t, i, l] -= dispatch_count
                        bss_inventories[station_id]['charged'] -= dispatch_count
                        num_taxis_to_swap -= dispatch_count
                        
                        if num_taxis_to_swap == 0:
                            break  # 此区域此电量的车辆已全部调度完毕
    
    def _heuristic_passenger_dispatch(self, t: int, state: dict, solution: dict):
        """启发式乘客调度：将空闲车辆调度到需求最迫切的区域。"""
        vacant_taxis = state['vacant_taxis']
        demand = self.demand_model.get_demand(t) # 获取 t 时刻的需求矩阵 D_ij

        for i in range(self.m_areas):
            # 从高电量到低电量遍历，优先使用电量充足的车辆服务乘客
            for l in range(self.L_energy_levels - 1, -1, -1):
                available_taxis = int(vacant_taxis[t, i, l])
                if available_taxis == 0:
                    continue

                # 将区域 i 的可用车辆分配给有需求的目的地 j
                # (简化逻辑: 按区域ID顺序分配)
                for j in range(self.m_areas):
                    if demand[i, j] > 0:
                        dispatch_count = min(available_taxis, demand[i, j])
                        if dispatch_count == 0:
                            continue

                        # 更新解决方案
                        solution['passenger_dispatch'][t, i, j, l] += dispatch_count
                        solution['service_quality'] += dispatch_count
                        
                        # 更新状态拷贝
                        vacant_taxis[t, i, l] -= dispatch_count
                        demand[i, j] -= dispatch_count
                        available_taxis -= dispatch_count

                        if available_taxis == 0:
                            break # 此区域此电量的车辆已用完
    
    def _heuristic_charging_schedule(self, t: int, state: dict, solution: dict):
        """启发式充电计划：当任何一个站点的空电池数量超过阈值时，启动充电。"""
        for station_id, inventory in state['bss_inventories'].items():
            # 如果空电池数量超过容量的一半，就认为需要充电
            if inventory['empty'] > self.config.bss_capacity // 2:
                solution['charging_schedule'][t] = 1  # 决策为“充电”
                solution['charging_cost'] += 50 # 假设一个固定的充电成本
                # 只要有一个站需要，就启动全局充电决策，然后跳出
                break

    def validate_solution(self, solution: dict, current_state: dict) -> bool:
        """
        验证生成的解决方案是否可行（例如，调度的车辆数是否超过可用数）。
        这是非常重要的一步，可以防止无效决策传递给模拟器。
        """
        # --- 验证乘客调度 ---
        # dispatched_passengers[i, l] = sum over j (passenger_dispatch[t, i, j, l])
        passenger_dispatched_from_area_l = np.sum(solution['passenger_dispatch'][0], axis=1)
        
        # --- 验证换电调度 ---
        # dispatched_swaps[i, l] = sum over j (swap_dispatch[t, i, j, l])
        swap_dispatched_from_area_l = np.sum(solution['swap_dispatch'][0], axis=1)

        # 总调度数量不能超过该区域该电量的可用空闲车数量
        total_dispatched = passenger_dispatched_from_area_l + swap_dispatched_from_area_l
        available_taxis = current_state['vacant_taxis'][0] # t=0
        
        # 使用 np.any 检查是否有任何一个元素的调度数超过可用数
        if np.any(total_dispatched > available_taxis + 1e-6): # 加一个小的容差避免浮点误差
            self.logger.error("验证失败: 总调度车辆数（乘客+换电）超过了可用空闲车辆数。")
            # 打印出有问题的具体条目以供调试
            problem_indices = np.where(total_dispatched > available_taxis + 1e-6)
            for i, l in zip(*problem_indices):
                self.logger.debug(f"区域 {i}, 电量 {l}: "
                                  f"调度数 {total_dispatched[i, l]} > "
                                  f"可用数 {available_taxis[i, l]}")
            return False
            
        # TODO: 可以添加更多验证，例如换电站库存约束等。
        
        return True

    def _create_empty_solution(self) -> dict:
        """
        创建一个空的（“什么都不做”）解决方案字典。
        在验证失败或不需要任何操作时返回。
        """
        # 解决方案的维度只针对当前时间步，所以时间维度为1
        t, m, l = 1, self.m_areas, self.L_energy_levels
        return {
            'passenger_dispatch': np.zeros((t, m, m, l)),
            'swap_dispatch': np.zeros((t, m, m, l)), # 注意这里的第三个维度是换电站索引
            'charging_schedule': np.zeros(t),
            'service_quality': 0.0,
            'idle_distance': 0.0,
            'charging_cost': 0.0,
            'objective_value': 0.0
        }