"""实现论文中的电动出租车状态转移模型。"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

class ETaxiStateModel:
    """
    电动出租车车队状态转移模型，实现论文中的公式(1)-(7)。
    """
    
    def __init__(self, m_areas: int, L_energy_levels: int, T_periods: int):
        """
        初始化状态模型。
        
        参数:
            m_areas (int): 城市区域数量
            L_energy_levels (int): 电池能量等级数量
            T_periods (int): 时间段数量
        """
        self.m = m_areas  # 城市区域数量
        self.L = L_energy_levels  # 能量等级数量
        self.T = T_periods  # 时间段数量
        
        # 初始化状态变量
        self.reset_state()
        
        # 日志配置
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def reset_state(self):
        """重置所有状态变量。"""
        # 当前时间段
        self.current_period = 0
        
        # 车辆分布状态 V^t_{i,l}, O^t_{i,l}
        self.vacant_taxis = np.zeros((self.T, self.m, self.L))  # 空闲出租车
        self.occupied_taxis = np.zeros((self.T, self.m, self.L))  # 载客出租车
        
        # 决策变量 X^{t,l}_{i,i'}, Y^{t,l}_{i,i'}
        self.passenger_dispatch = np.zeros((self.T, self.m, self.m, self.L))  # 乘客服务调度
        self.swap_dispatch = np.zeros((self.T, self.m, self.m, self.L))  # 换电调度
        
        # 中间状态变量 B^t_{i,l}, S^t_{i,l}
        self.swapping_taxis = np.zeros((self.T, self.m, self.L))  # 换电中的出租车
        self.serving_taxis = np.zeros((self.T, self.m, self.L))  # 服务中的出租车
        
        # 能量消耗矩阵 E^t_{i,i'}
        self.energy_consumption = np.ones((self.m, self.m)) * 1.0  # 默认每次移动消耗1单位能量
        
        # 移动概率矩阵
        self.P_occupied = np.zeros((self.T, self.m, self.m))  # 载客移动概率
        self.Q_vacant = np.zeros((self.T, self.m, self.m))  # 空车移动概率
        self.P_tilde_serving = np.zeros((self.T, self.m, self.m))  # 服务中变为载客的概率
        self.Q_tilde_serving = np.zeros((self.T, self.m, self.m))  # 服务中变为空车的概率
        
        # 换电站相关状态
        self.swap_success_taxis = np.zeros((self.T, self.m, self.L))  # 成功换电的出租车 H^t_{i,l}
    
    def set_energy_consumption_matrix(self, energy_matrix: np.ndarray):
        """
        设置区域间能量消耗矩阵。
        
        参数:
            energy_matrix (ndarray): m×m的能量消耗矩阵
        """
        if energy_matrix.shape != (self.m, self.m):
            raise ValueError(f"能量消耗矩阵形状应为 ({self.m}, {self.m})")
        
        self.energy_consumption = energy_matrix.copy()
        self.logger.info(f"设置能量消耗矩阵，平均消耗: {np.mean(energy_matrix):.2f}")
    
    def set_movement_probabilities(self, t: int, P_occ: np.ndarray, Q_vac: np.ndarray, 
                                 P_tilde: np.ndarray, Q_tilde: np.ndarray):
        """
        设置时间段t的移动概率矩阵。
        
        参数:
            t (int): 时间段
            P_occ (ndarray): 载客移动概率矩阵 P^t_{i',i}
            Q_vac (ndarray): 空车移动概率矩阵 Q^t_{i',i}
            P_tilde (ndarray): 服务中变载客概率矩阵 P̃^t_{i',i}
            Q_tilde (ndarray): 服务中变空车概率矩阵 Q̃^t_{i',i}
        """
        if t >= self.T:
            raise ValueError(f"时间段 {t} 超出范围 [0, {self.T-1}]")
        
        self.P_occupied[t] = P_occ.copy()
        self.Q_vacant[t] = Q_vac.copy()
        self.P_tilde_serving[t] = P_tilde.copy()
        self.Q_tilde_serving[t] = Q_tilde.copy()
    
    def set_initial_state(self, vacant_dist: np.ndarray, occupied_dist: np.ndarray):
        """
        设置初始车辆分布状态。
        
        参数:
            vacant_dist (ndarray): 初始空闲车辆分布 V^0_{i,l}
            occupied_dist (ndarray): 初始载客车辆分布 O^0_{i,l}
        """
        if vacant_dist.shape != (self.m, self.L):
            raise ValueError(f"空闲车辆分布形状应为 ({self.m}, {self.L})")
        if occupied_dist.shape != (self.m, self.L):
            raise ValueError(f"载客车辆分布形状应为 ({self.m}, {self.L})")
        
        self.vacant_taxis[0] = vacant_dist.copy()
        self.occupied_taxis[0] = occupied_dist.copy()
        
        total_taxis = np.sum(vacant_dist) + np.sum(occupied_dist)
        self.logger.info(f"设置初始状态: {total_taxis} 辆出租车")
    
    def apply_dispatch_decisions(self, t: int, X: np.ndarray, Y: np.ndarray):
        """
        应用调度决策，实现论文公式(1)。
        
        参数:
            t (int): 当前时间段
            X (ndarray): 乘客服务调度决策 X^{t,l}_{i,i'}
            Y (ndarray): 换电调度决策 Y^{t,l}_{i,i'}
        """
        if t >= self.T:
            raise ValueError(f"时间段 {t} 超出范围")
        
        # 存储调度决策
        self.passenger_dispatch[t] = X.copy()
        self.swap_dispatch[t] = Y.copy()
        
        # 计算中间状态变量 B^t_{i,l}, S^t_{i,l} (公式1)
        for i in range(self.m):
            for l in range(self.L):
                # 换电中的出租车：从其他区域来到区域i换电的车辆
                self.swapping_taxis[t, i, l] = np.sum(Y[:, i, l])
                
                # 服务中的出租车：从其他区域来到区域i服务乘客的车辆
                self.serving_taxis[t, i, l] = np.sum(X[:, i, l])
        
        self.logger.info(f"时间段 {t}: 应用调度决策")
    
    def compute_energy_adjusted_indices(self, t: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算考虑能量消耗后的索引映射。
        
        参数:
            t (int): 当前时间段
        
        返回:
            tuple: (能量调整后的索引矩阵，有效性掩码)
        """
        # 为每个(区域,能量等级)对计算移动后的能量等级
        energy_adjusted = np.zeros((self.m, self.m, self.L), dtype=int)
        valid_moves = np.zeros((self.m, self.m, self.L), dtype=bool)
        
        for i in range(self.m):
            for i_prime in range(self.m):
                for l in range(self.L):
                    # 计算移动后的能量等级
                    energy_cost = self.energy_consumption[i, i_prime]
                    new_energy_level = l - int(energy_cost)
                    
                    if new_energy_level >= 0:
                        energy_adjusted[i, i_prime, l] = new_energy_level
                        valid_moves[i, i_prime, l] = True
                    else:
                        # 能量不足，无法完成移动
                        energy_adjusted[i, i_prime, l] = 0
                        valid_moves[i, i_prime, l] = False
        
        return energy_adjusted, valid_moves
    
    def update_taxi_distribution(self, t: int, swap_results: Optional[np.ndarray] = None):
        """
        更新下一时间段的出租车分布，实现论文公式(2)-(3)。
        
        参数:
            t (int): 当前时间段
            swap_results (ndarray, optional): 换电结果 H^t_{i,l}。
                                              这是一个在时间段t, 各区域i成功换电的车辆分布。
                                              形状应为 (m, L)。
        """
        if t + 1 >= self.T:
            return

        # 获取能量调整后的索引
        energy_adjusted, valid_moves = self.compute_energy_adjusted_indices(t)
        
        # 初始化下一时间段的状态
        next_occupied = np.zeros((self.m, self.L))
        next_vacant = np.zeros((self.m, self.L))

        # --- 更新载客出租车分布 O^{t+1} (公式2) ---
        for i in range(self.m):  # 目标区域
            for l in range(self.L):  # 目标能量
                # 1. 来自"服务中"的出租车 (S^t -> O^{t+1})
                # 从各区域i'出发，服务后变为载客到达区域i
                for i_prime in range(self.m):
                    for l_prime in range(self.L):
                        if valid_moves[i_prime, i, l_prime] and energy_adjusted[i_prime, i, l_prime] == l:
                            next_occupied[i, l] += self.P_tilde_serving[t, i_prime, i] * self.serving_taxis[t, i_prime, l_prime]

                # 2. 来自"已载客"的出租车 (O^t -> O^{t+1})
                # 从各区域i'出发，送客后继续载客到达区域i
                for i_prime in range(self.m):
                    for l_prime in range(self.L):
                        if valid_moves[i_prime, i, l_prime] and energy_adjusted[i_prime, i, l_prime] == l:
                            next_occupied[i, l] += self.P_occupied[t, i_prime, i] * self.occupied_taxis[t, i_prime, l_prime]
        
        # --- 更新空闲出租车分布 V^{t+1} (公式3) ---
        for i in range(self.m):  # 目标区域
            for l in range(self.L):  # 目标能量
                # 1. 来自"服务中"的出租车 (S^t -> V^{t+1})
                # 从各区域i'出发，服务后变为空闲到达区域i
                for i_prime in range(self.m):
                    for l_prime in range(self.L):
                        if valid_moves[i_prime, i, l_prime] and energy_adjusted[i_prime, i, l_prime] == l:
                            next_vacant[i, l] += self.Q_tilde_serving[t, i_prime, i] * self.serving_taxis[t, i_prime, l_prime]
                
                # 2. 来自"已载客"的出租车 (O^t -> V^{t+1})
                # 从各区域i'出发，送客后变为空闲到达区域i
                for i_prime in range(self.m):
                    for l_prime in range(self.L):
                        if valid_moves[i_prime, i, l_prime] and energy_adjusted[i_prime, i, l_prime] == l:
                            next_vacant[i, l] += self.Q_vacant[t, i_prime, i] * self.occupied_taxis[t, i_prime, l_prime]
        
        # 3. 来自成功换电的出租车 (B^t -> H^t -> V^{t+1})
        if swap_results is not None:
            if swap_results.shape != (self.m, self.L):
                raise ValueError(f"swap_results 形状应为 ({self.m}, {self.L})，但得到 {swap_results.shape}")
            # H^t_{i,l} 表示在t时间段，i区域，成功换电后能量为l的车辆数
            next_vacant += swap_results

        # 赋值到下一时间段
        self.occupied_taxis[t+1] = next_occupied
        self.vacant_taxis[t+1] = next_vacant
        
        self.logger.info(f"更新时间段 {t+1} 的车辆分布")
    
    def get_total_taxis_by_area(self, t: int) -> np.ndarray:
        """
        获取每个区域的总出租车数量。
        
        参数:
            t (int): 时间段
        
        返回:
            ndarray: 每个区域的总出租车数量
        """
        if t >= self.T:
            raise ValueError(f"时间段 {t} 超出范围")
        
        vacant_by_area = np.sum(self.vacant_taxis[t], axis=1)
        occupied_by_area = np.sum(self.occupied_taxis[t], axis=1)
        
        return vacant_by_area + occupied_by_area
    
    def get_total_taxis_by_energy(self, t: int) -> np.ndarray:
        """
        获取每个能量等级的总出租车数量。
        
        参数:
            t (int): 时间段
        
        返回:
            ndarray: 每个能量等级的总出租车数量
        """
        if t >= self.T:
            raise ValueError(f"时间段 {t} 超出范围")
        
        vacant_by_energy = np.sum(self.vacant_taxis[t], axis=0)
        occupied_by_energy = np.sum(self.occupied_taxis[t], axis=0)
        
        return vacant_by_energy + occupied_by_energy
    
    def get_state_summary(self, t: int) -> Dict:
        """
        获取时间段t的状态摘要。
        
        参数:
            t (int): 时间段
        
        返回:
            dict: 状态摘要信息
        """
        if t >= self.T:
            raise ValueError(f"时间段 {t} 超出范围")
        
        total_vacant = np.sum(self.vacant_taxis[t])
        total_occupied = np.sum(self.occupied_taxis[t])
        total_swapping = np.sum(self.swapping_taxis[t])
        total_serving = np.sum(self.serving_taxis[t])
        
        summary = {
            'time_period': t,
            'total_taxis': total_vacant + total_occupied,
            'vacant_taxis': total_vacant,
            'occupied_taxis': total_occupied,
            'swapping_taxis': total_swapping,
            'serving_taxis': total_serving,
            'utilization_rate': total_occupied / (total_vacant + total_occupied) if (total_vacant + total_occupied) > 0 else 0,
            'taxis_by_area': self.get_total_taxis_by_area(t).tolist(),
            'taxis_by_energy': self.get_total_taxis_by_energy(t).tolist()
        }
        
        return summary
    
    def validate_state_conservation(self, t: int) -> bool:
        """
        验证状态转移是否保持出租车总数守恒。
        
        参数:
            t (int): 时间段
        
        返回:
            bool: 是否守恒
        """
        if t == 0 or t >= self.T:
            return True
        
        total_prev = np.sum(self.vacant_taxis[t-1]) + np.sum(self.occupied_taxis[t-1])
        total_curr = np.sum(self.vacant_taxis[t]) + np.sum(self.occupied_taxis[t])
        
        is_conserved = abs(total_prev - total_curr) < 1e-6
        
        if not is_conserved:
            self.logger.warning(f"时间段 {t}: 出租车总数不守恒 ({total_prev} -> {total_curr})")
        
        return is_conserved
    
    def simulate_period(self, t: int, X: np.ndarray, Y: np.ndarray, 
                       swap_results: Optional[np.ndarray] = None):
        """
        模拟一个完整的时间段，包括调度决策和状态更新。
        
        参数:
            t (int): 当前时间段
            X (ndarray): 乘客服务调度决策
            Y (ndarray): 换电调度决策
            swap_results (ndarray, optional): 换电结果
        """
        # 1. 应用调度决策
        self.apply_dispatch_decisions(t, X, Y)
        
        # 2. 更新车辆分布
        self.update_taxi_distribution(t, swap_results)
        
        # 3. 验证状态守恒
        self.validate_state_conservation(t + 1)
        
        # 4. 更新当前时间段
        self.current_period = t + 1
    
    def export_state_history(self) -> Dict:
        """
        导出完整的状态历史记录。
        
        返回:
            dict: 包含所有状态变量的历史记录
        """
        return {
            'vacant_taxis': self.vacant_taxis.copy(),
            'occupied_taxis': self.occupied_taxis.copy(),
            'swapping_taxis': self.swapping_taxis.copy(),
            'serving_taxis': self.serving_taxis.copy(),
            'passenger_dispatch': self.passenger_dispatch.copy(),
            'swap_dispatch': self.swap_dispatch.copy(),
            'energy_consumption': self.energy_consumption.copy(),
            'parameters': {
                'm_areas': self.m,
                'L_energy_levels': self.L,
                'T_periods': self.T
            }
        }

class BSSStateModel:
    """
    电池交换站状态模型，实现论文中的公式(4)-(7)。
    """
    
    def __init__(self, station_id: int, location: int, capacity: int, 
                 chargers: int, T_periods: int, L_energy_levels: int):
        """
        初始化电池交换站状态模型。
        
        参数:
            station_id (int): 站点ID
            location (int): 站点位置(区域ID)
            capacity (int): 电池总容量
            chargers (int): 充电器数量
            T_periods (int): 时间段数量
            L_energy_levels (int): 能量等级数量
        """
        self.id = station_id
        self.location = location
        self.capacity = capacity
        self.chargers = chargers
        self.T = T_periods
        self.L = L_energy_levels
        
        # 状态变量
        self.reset_state()
        
        # 日志配置
        self.logger = logging.getLogger(f"BSS_{station_id}")
    
    def reset_state(self):
        """重置站点状态。"""
        # 电池库存 M^t_{i,l}
        self.battery_inventory = np.zeros((self.T, self.L))
        
        # 换电操作矩阵 μ^t_{i,l,l'}
        self.swap_operations = np.zeros((self.T, self.L, self.L))
        
        # 充电决策 y^t_{i,l}
        self.charging_decisions = np.zeros((self.T, self.L))
        
        # 换电后的电池库存 M̂^t_{i,l}
        self.post_swap_inventory = np.zeros((self.T, self.L))
        
        # 参数
        self.service_capacity = 5  # 每时间段最大换电数量 p_i
        self.charge_increment = 1  # 每次充电增加的能量等级 l̂
    
    def set_initial_inventory(self, initial_batteries: np.ndarray):
        """
        设置初始电池库存。
        
        参数:
            initial_batteries (ndarray): 各能量等级的初始电池数量
        """
        if len(initial_batteries) != self.L:
            raise ValueError(f"初始库存长度应为 {self.L}")
        
        self.battery_inventory[0] = initial_batteries.copy()
        self.logger.info(f"设置初始库存: {np.sum(initial_batteries)} 块电池")
    
    def process_swap_requests(self, t: int, arriving_taxis: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        处理换电请求，实现论文公式(4)-(5)。
        
        参数:
            t (int): 当前时间段
            arriving_taxis (ndarray): 到达的出租车数量 B^t_{i,l}
        
        返回:
            tuple: (换电操作矩阵, 换电后的出租车分布)
        """
        if t >= self.T:
            raise ValueError(f"时间段 {t} 超出范围")
        
        # 初始化换电操作矩阵
        mu = np.zeros((self.L, self.L))
        
        # 当前库存
        current_inventory = self.battery_inventory[t].copy()
        
        # 总换电数量限制
        total_swaps = 0
        max_swaps = min(self.service_capacity, np.sum(arriving_taxis))
        
        # 贪心分配策略：优先为低电量车辆换高电量电池
        for l in range(self.L):  # 到达车辆的能量等级
            if arriving_taxis[l] > 0 and total_swaps < max_swaps:
                # 为能量等级l的车辆分配电池
                remaining_taxis = min(arriving_taxis[l], max_swaps - total_swaps)
                
                # 优先分配高能量等级的电池
                for l_prime in range(self.L-1, -1, -1):
                    if current_inventory[l_prime] > 0 and remaining_taxis > 0:
                        # 实际换电数量
                        swaps = min(remaining_taxis, current_inventory[l_prime])
                        
                        # 记录换电操作
                        mu[l, l_prime] = swaps
                        
                        # 更新库存和剩余出租车
                        current_inventory[l_prime] -= swaps
                        remaining_taxis -= swaps
                        total_swaps += swaps
        
        # 存储换电操作
        self.swap_operations[t] = mu.copy()
        
        # 计算换电后的出租车分布 H^t_{i,l} (公式5)
        H = np.zeros(self.L)
        for l in range(self.L):
            # 原有能量等级l的出租车数量
            original_count = arriving_taxis[l]
            # 离开的出租车数量(换成其他能量等级)
            departed = np.sum(mu[l, :])
            # 到达的出租车数量(从其他能量等级换到l)
            arrived = np.sum(mu[:, l])
            # 最终数量
            H[l] = original_count - departed + arrived
        
        # 计算换电后的库存 M̂^t_{i,l}
        post_swap_inv = self.battery_inventory[t].copy()
        for l in range(self.L):
            for l_prime in range(self.L):
                if l != l_prime:
                    # 取走的电池
                    post_swap_inv[l_prime] -= mu[l, l_prime]
                    # 放回的电池
                    post_swap_inv[l] += mu[l_prime, l]
        
        self.post_swap_inventory[t] = post_swap_inv
        
        self.logger.info(f"时间段 {t}: 处理 {total_swaps} 次换电")
        
        return mu, H
    
    def make_charging_decisions(self, t: int, strategy: str = 'greedy') -> np.ndarray:
        """
        制定充电决策，实现论文公式(6)。
        
        参数:
            t (int): 当前时间段
            strategy (str): 充电策略 ('greedy', 'balanced')
        
        返回:
            ndarray: 充电决策向量
        """
        if t >= self.T:
            raise ValueError(f"时间段 {t} 超出范围")
        
        # 当前电池库存
        current_inventory = self.post_swap_inventory[t].copy()
        
        # 初始化充电决策
        y = np.zeros(self.L)
        
        # 可用充电器数量
        available_chargers = self.chargers
        
        if strategy == 'greedy':
            # 贪心策略：优先充电低能量等级的电池
            for l in range(self.L):
                if l < self.L - self.charge_increment:  # 确保充电后不超过最大能量等级
                    # 可充电的电池数量
                    chargeable = min(current_inventory[l], available_chargers)
                    y[l] = chargeable
                    available_chargers -= chargeable
                    
                    if available_chargers <= 0:
                        break
        
        elif strategy == 'balanced':
            # 平衡策略：尝试保持各能量等级的平衡
            target_per_level = self.capacity // self.L
            
            for l in range(self.L):
                if l < self.L - self.charge_increment:
                    # 计算需要充电的数量以达到目标
                    target_high_level = l + self.charge_increment
                    current_high = current_inventory[target_high_level]
                    shortage = max(0, target_per_level - current_high)
                    
                    # 实际充电数量
                    chargeable = min(current_inventory[l], shortage, available_chargers)
                    y[l] = chargeable
                    available_chargers -= chargeable
                    
                    if available_chargers <= 0:
                        break
        
        # 存储充电决策
        self.charging_decisions[t] = y.copy()
        
        self.logger.info(f"时间段 {t}: 充电决策 {np.sum(y)} 块电池")
        
        return y
    
    def update_inventory(self, t: int):
        """
        更新下一时间段的电池库存，实现论文公式(7)。
        
        参数:
            t (int): 当前时间段
        """
        if t + 1 >= self.T:
            return
        
        # 获取当前的库存和充电决策
        current_inventory = self.post_swap_inventory[t].copy()
        charging = self.charging_decisions[t].copy()
        
        # 计算下一时间段的库存 (公式7)
        next_inventory = current_inventory.copy()
        
        for l in range(self.L):
            # 减去被充电的电池
            next_inventory[l] -= charging[l]
            
            # 添加充电完成的电池
            if l >= self.charge_increment:
                source_level = l - self.charge_increment
                next_inventory[l] += charging[source_level]
        
        # 更新库存
        self.battery_inventory[t+1] = next_inventory
        
        self.logger.info(f"更新时间段 {t+1} 的电池库存")
    
    def get_station_status(self, t: int) -> Dict:
        """
        获取站点状态摘要。
        
        参数:
            t (int): 时间段
        
        返回:
            dict: 状态摘要
        """
        if t >= self.T:
            raise ValueError(f"时间段 {t} 超出范围")
        
        total_batteries = np.sum(self.battery_inventory[t])
        total_swaps = np.sum(self.swap_operations[t])
        total_charging = np.sum(self.charging_decisions[t])
        
        status = {
            'station_id': self.id,
            'location': self.location,
            'time_period': t,
            'total_batteries': total_batteries,
            'inventory_by_level': self.battery_inventory[t].tolist(),
            'total_swaps': total_swaps,
            'total_charging': total_charging,
            'utilization': total_batteries / self.capacity if self.capacity > 0 else 0,
            'charger_utilization': total_charging / self.chargers if self.chargers > 0 else 0
        }
        
        return status
    
    def simulate_period(self, t: int, arriving_taxis: np.ndarray, charging_strategy: str = 'greedy'):
        """
        模拟一个完整的时间段操作。
        
        参数:
            t (int): 当前时间段
            arriving_taxis (ndarray): 到达的出租车分布
            charging_strategy (str): 充电策略
        
        返回:
            tuple: (换电操作矩阵, 换电后车辆分布)
        """
        # 1. 处理换电请求
        swap_ops, taxi_dist = self.process_swap_requests(t, arriving_taxis)
        
        # 2. 制定充电决策
        self.make_charging_decisions(t, charging_strategy)
        
        # 3. 更新库存
        self.update_inventory(t)
        
        return swap_ops, taxi_dist

if __name__ == "__main__":
    # 测试状态转移模型
    print("测试电动出租车状态转移模型...")
    
    # 创建状态模型
    m_areas = 5
    L_levels = 10
    T_periods = 24
    
    taxi_model = ETaxiStateModel(m_areas, L_levels, T_periods)
    
    # 设置初始状态
    np.random.seed(42)
    initial_vacant = np.random.randint(0, 20, (m_areas, L_levels))
    initial_occupied = np.random.randint(0, 10, (m_areas, L_levels))
    
    taxi_model.set_initial_state(initial_vacant, initial_occupied)
    
    # 测试一个时间段的模拟
    X = np.random.randint(0, 5, (m_areas, m_areas, L_levels))
    Y = np.random.randint(0, 3, (m_areas, m_areas, L_levels))
    
    taxi_model.simulate_period(0, X, Y)
    
    # 获取状态摘要
    summary = taxi_model.get_state_summary(1)
    print(f"时间段1状态摘要: {summary}")
    
    # 测试换电站模型
    print("\n测试电池交换站模型...")
    
    bss_model = BSSStateModel(
        station_id=1,
        location=2,
        capacity=50,
        chargers=10,
        T_periods=T_periods,
        L_energy_levels=L_levels
    )
    
    # 设置初始库存
    initial_inventory = np.array([2, 3, 4, 5, 6, 5, 4, 3, 2, 1])  # 各能量等级的电池数量
    bss_model.set_initial_inventory(initial_inventory)
    
    # 模拟换电请求
    arriving_taxis = np.array([3, 2, 1, 0, 0, 0, 0, 0, 0, 0])  # 低电量出租车较多
    swap_ops, taxi_dist = bss_model.simulate_period(0, arriving_taxis)
    
    # 获取站点状态
    bss_status = bss_model.get_station_status(1)
    print(f"换电站状态: {bss_status}")
    
    print("\n状态转移模型测试完成!")