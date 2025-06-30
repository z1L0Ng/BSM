"""联合优化器，整合出租车调度和换电站充电优化。"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime

from models.optimization_model import JointOptimizationModel
from models.state_transition import ETaxiStateModel, BSSStateModel

@dataclass
class OptimizationConfig:
    """优化配置类。"""
    m_areas: int
    L_energy_levels: int
    T_periods: int
    period_length_minutes: int = 15
    beta: float = -0.1
    solver_method: str = 'auto'
    time_limit: int = 300
    mip_gap: float = 0.05
    heuristic_params: Dict = None

class JointOptimizer:
    """
    联合优化器，实现论文中的整体优化策略。
    协调出租车调度和换电站充电管理。
    """
    
    def __init__(self, config: OptimizationConfig, network_data: Dict):
        """
        初始化联合优化器。
        
        参数:
            config (OptimizationConfig): 优化配置
            network_data (dict): 网络数据（距离、可达性等）
        """
        self.config = config
        self.network_data = network_data
        
        model_config = {
            'm_areas': config.m_areas,
            'L_energy_levels': config.L_energy_levels,
            'T_periods': config.T_periods,
            'beta': config.beta,
            'demand': {},
            'reachability': network_data.get('reachability_matrix', {}),
            'distance_weights': network_data.get('distance_matrix', {}),
            'solver': {
                'time_limit': config.time_limit,
                'mip_gap': config.mip_gap
            }
        }
        
        self.optimization_model = JointOptimizationModel(model_config)
        
        self.state_history = []
        self.solution_history = []
        
        self.performance_stats = {
            'total_passengers_served': 0,
            'total_swaps_completed': 0,
            'total_idle_distance': 0,
            'avg_response_time': [],
            'utilization_rates': []
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def add_battery_swap_station(self, station_config: Dict):
        """添加电池交换站到优化模型。"""
        self.optimization_model.add_bss_station(station_config)
        self.logger.info(f"添加换电站 {station_config['id']} 到优化器")
    
    def set_demand_forecast(self, demand_data: Dict):
        """设置需求预测数据。"""
        self.optimization_model.demand = demand_data
        self.logger.info(f"设置需求预测，总需求: {sum(demand_data.values())}")
    
    def optimize_single_period(self, current_time: int, system_state: Dict) -> Dict:
        """优化单个时间段的决策。"""
        start_time = datetime.now()
        
        horizon_state = self._create_horizon_state(current_time, system_state)
        
        solution = self.optimization_model.solve(
            horizon_state, 
            method=self.config.solver_method
        )
        
        # 验证时使用原始的 horizon_state
        if not self.optimization_model.validate_solution(solution, self._create_horizon_state(current_time, system_state)):
            self.logger.warning("优化解不可行，使用备用策略")
            solution = self._fallback_strategy(current_time, system_state)
        
        current_decisions = self._extract_current_decisions(solution, current_time)
        
        solve_time = (datetime.now() - start_time).total_seconds()
        self._update_performance_stats(solution, solve_time)
        
        self.state_history.append({
            'time': current_time,
            'state': system_state.copy(),
            'solve_time': solve_time
        })
        self.solution_history.append({
            'time': current_time,
            'solution': solution,
            'decisions': current_decisions
        })
        
        return current_decisions
    
    def _create_horizon_state(self, current_time: int, system_state: Dict) -> Dict:
        """创建滚动时域优化的状态。"""
        horizon_length = min(6, self.config.T_periods - current_time)
        
        horizon_state = {
            'vacant_taxis': np.zeros((horizon_length, self.config.m_areas, self.config.L_energy_levels)),
            'occupied_taxis': np.zeros((horizon_length, self.config.m_areas, self.config.L_energy_levels)),
            'bss_inventories': {}
        }
        
        horizon_state['vacant_taxis'][0] = system_state.get('vacant_taxis', np.zeros((self.config.m_areas, self.config.L_energy_levels)))
        horizon_state['occupied_taxis'][0] = system_state.get('occupied_taxis', np.zeros((self.config.m_areas, self.config.L_energy_levels)))
        
        for station_id, inventory in system_state.get('bss_inventories', {}).items():
            horizon_state['bss_inventories'][station_id] = inventory
        
        for t in range(1, horizon_length):
            horizon_state['vacant_taxis'][t] = horizon_state['vacant_taxis'][0] * 0.9
            horizon_state['occupied_taxis'][t] = horizon_state['occupied_taxis'][0] * 0.9
        
        return horizon_state
    
    def _extract_current_decisions(self, solution: Dict, current_time: int) -> Dict:
        """从完整解中提取当前时间段的决策。"""
        t_idx = 0
        
        decisions = {
            'passenger_dispatch': solution['passenger_dispatch'][t_idx],
            'swap_dispatch': solution['swap_dispatch'][t_idx],
            'service_assignments': solution['service_taxis'][t_idx],
            'served_passengers': solution['served_passengers'][t_idx],
            'time_period': current_time,
            'objective_value': solution['objective_value'],
            'performance': {
                'service_quality': solution['service_quality'],
                'idle_distance': solution['idle_distance']
            }
        }
        
        return decisions
    
    def _fallback_strategy(self, current_time: int, system_state: Dict) -> Dict:
        """当优化求解失败时使用的备用策略。"""
        self.logger.info("使用备用策略")
        
        fallback_solution = {
            'passenger_dispatch': np.zeros((1, self.config.m_areas, self.config.m_areas, self.config.L_energy_levels)),
            'swap_dispatch': np.zeros((1, self.config.m_areas, self.config.m_areas, self.config.L_energy_levels)),
            'service_taxis': np.zeros((1, self.config.m_areas, self.config.L_energy_levels)),
            'served_passengers': np.zeros((1, self.config.m_areas)),
            'objective_value': 0, 'service_quality': 0, 'idle_distance': 0
        }
        
        vacant_taxis = system_state.get('vacant_taxis', np.zeros((self.config.m_areas, self.config.L_energy_levels)))
        
        for i in range(self.config.m_areas):
            demand = self.optimization_model.demand.get((current_time, i), 0)
            for l in range(self.config.L_energy_levels-1, -1, -1):
                available = vacant_taxis[i, l]
                if available > 0 and demand > 0:
                    service_count = min(available, demand)
                    fallback_solution['passenger_dispatch'][0, i, i, l] = service_count
                    fallback_solution['service_taxis'][0, i, l] = service_count
                    fallback_solution['served_passengers'][0, i] += service_count
                    demand -= service_count
            
            for l in range(2):
                low_battery_taxis = vacant_taxis[i, l]
                if low_battery_taxis > 0:
                    best_station = self._find_nearest_station(i)
                    if best_station is not None:
                        swap_count = min(low_battery_taxis, 2)
                        fallback_solution['swap_dispatch'][0, i, best_station, l] = swap_count
        
        return fallback_solution
    
    def _find_nearest_station(self, area: int) -> Optional[int]:
        """寻找最近的换电站。"""
        min_distance = float('inf')
        nearest_station = None
        
        for station_id, bss_model in self.optimization_model.bss_models.items():
            station_area = bss_model.location
            distance = self.network_data.get('distance_matrix', {}).get((area, station_area), float('inf'))
            if distance < min_distance:
                min_distance = distance
                nearest_station = station_area
        
        return nearest_station
    
    def _update_performance_stats(self, solution: Dict, solve_time: float):
        """更新性能统计。"""
        self.performance_stats['total_passengers_served'] += solution['service_quality']
        self.performance_stats['total_swaps_completed'] += np.sum(solution['swap_dispatch'])
        self.performance_stats['total_idle_distance'] += solution['idle_distance']
        self.performance_stats['avg_response_time'].append(solve_time)
        
        total_service_taxis = np.sum(solution['service_taxis'])
        total_available = self.config.m_areas * self.config.L_energy_levels
        utilization = total_service_taxis / total_available if total_available > 0 else 0
        self.performance_stats['utilization_rates'].append(utilization)
    
    def optimize_charging_schedule(self, current_time: int, station_states: Dict, 
                                 electricity_prices: Dict = None) -> Dict:
        """优化充电调度策略。"""
        charging_decisions = {}
        
        for station_id, inventory in station_states.items():
            if station_id in self.optimization_model.bss_models:
                bss_model = self.optimization_model.bss_models[station_id]
                
                # --- 关键修改：直接使用传入的 inventory 数组 ---
                current_inventory = inventory
                
                future_demand = self._predict_station_demand(station_id, current_time)
                current_price = electricity_prices.get(current_time, 1.0) if electricity_prices else 1.0
                
                charging_strategy = self._determine_charging_strategy(
                    current_inventory, future_demand, current_price, bss_model
                )
                
                charging_decisions[station_id] = {
                    'charging_schedule': charging_strategy,
                    'expected_cost': self._calculate_charging_cost(charging_strategy, current_price),
                    'inventory_target': self._calculate_inventory_target(future_demand)
                }
        
        return charging_decisions

    def _predict_station_demand(self, station_id: int, current_time: int, horizon: int = 6) -> np.ndarray:
        """预测换电站未来需求。"""
        station_location = self.optimization_model.bss_models[station_id].location
        predicted_demand = np.zeros((horizon, self.config.L_energy_levels))
        
        for t in range(horizon):
            future_time = current_time + t
            if future_time < self.config.T_periods:
                area_demand = self.optimization_model.demand.get((future_time, station_location), 0)
                swap_demand = area_demand * 0.3
                for l in range(min(3, self.config.L_energy_levels)):
                    predicted_demand[t, l] = swap_demand / 3
        
        return predicted_demand
    
    def _determine_charging_strategy(self, current_inventory: np.ndarray, 
                                   future_demand: np.ndarray, 
                                   electricity_price: float,
                                   bss_model) -> np.ndarray:
        """确定充电策略。"""
        charging_decision = np.zeros(self.config.L_energy_levels)
        total_future_demand = np.sum(future_demand, axis=0)
        available_chargers = bss_model.chargers
        
        for l in range(self.config.L_energy_levels - 1):
            target_level = l + 1
            current_stock = current_inventory[target_level]
            expected_demand = total_future_demand[target_level]
            shortage = max(0, expected_demand - current_stock)
            
            available_to_charge = current_inventory[l]
            
            if shortage > 0 and available_to_charge > 0 and available_chargers > 0:
                charge_amount = min(shortage, available_to_charge, available_chargers)
                if electricity_price > 1.5: charge_amount = int(charge_amount * 0.7)
                elif electricity_price < 0.8: charge_amount = min(available_to_charge, available_chargers)
                
                charging_decision[l] = charge_amount
                available_chargers -= charge_amount
        
        return charging_decision
    
    def _calculate_charging_cost(self, charging_strategy: np.ndarray, electricity_price: float) -> float:
        """计算充电成本。"""
        return np.sum(charging_strategy) * 20.0 * electricity_price
    
    def _calculate_inventory_target(self, future_demand: np.ndarray) -> np.ndarray:
        """计算库存目标。"""
        return (np.sum(future_demand, axis=0) * 1.2).astype(int)

    
    def _calculate_charging_cost(self, charging_strategy: np.ndarray, 
                               electricity_price: float) -> float:
        """计算充电成本。"""
        # 假设每次充电消耗20kWh
        energy_per_charge = 20.0  # kWh
        total_charges = np.sum(charging_strategy)
        return total_charges * energy_per_charge * electricity_price
    
    def _calculate_inventory_target(self, future_demand: np.ndarray) -> np.ndarray:
        """计算库存目标。"""
        # 简单的库存目标：满足未来需求 + 安全库存
        safety_stock_ratio = 1.2  # 20%安全库存
        target = np.sum(future_demand, axis=0) * safety_stock_ratio
        return target.astype(int)
    
    def generate_recommendations(self, current_time: int, 
                               system_state: Dict) -> Dict:
        """
        生成运营建议。
        
        参数:
            current_time (int): 当前时间
            system_state (dict): 系统状态
        
        返回:
            dict: 运营建议
        """
        recommendations = {
            'taxi_dispatch': [],
            'charging_schedule': [],
            'capacity_adjustments': [],
            'performance_alerts': []
        }
        
        # 分析当前性能
        current_utilization = self._calculate_current_utilization(system_state)
        
        # 出租车调度建议
        if current_utilization < 0.6:
            recommendations['taxi_dispatch'].append({
                'type': 'increase_service',
                'message': '当前利用率较低，建议增加服务车辆',
                'priority': 'medium'
            })
        
        # 换电站容量建议
        for station_id, bss_model in self.optimization_model.bss_models.items():
            station_inventory = system_state.get('bss_inventories', {}).get(station_id, np.zeros(self.config.L_energy_levels))
            total_inventory = np.sum(station_inventory)
            capacity_ratio = total_inventory / bss_model.capacity
            
            if capacity_ratio < 0.3:
                recommendations['capacity_adjustments'].append({
                    'station_id': station_id,
                    'type': 'low_inventory',
                    'message': f'站点 {station_id} 库存过低 ({capacity_ratio:.1%})',
                    'priority': 'high'
                })
            elif capacity_ratio > 0.9:
                recommendations['capacity_adjustments'].append({
                    'station_id': station_id,
                    'type': 'high_inventory',
                    'message': f'站点 {station_id} 库存过高 ({capacity_ratio:.1%})',
                    'priority': 'low'
                })
        
        # 性能警报
        avg_response_time = np.mean(self.performance_stats['avg_response_time'][-10:]) if self.performance_stats['avg_response_time'] else 0
        if avg_response_time > 30:  # 30秒阈值
            recommendations['performance_alerts'].append({
                'type': 'slow_optimization',
                'message': f'优化求解时间过长 ({avg_response_time:.1f}s)',
                'priority': 'medium'
            })
        
        return recommendations
    
    def _calculate_current_utilization(self, system_state: Dict) -> float:
        """计算当前系统利用率。"""
        occupied_taxis = system_state.get('occupied_taxis', np.zeros((self.config.m_areas, self.config.L_energy_levels)))
        vacant_taxis = system_state.get('vacant_taxis', np.zeros((self.config.m_areas, self.config.L_energy_levels)))
        
        total_occupied = np.sum(occupied_taxis)
        total_taxis = np.sum(occupied_taxis) + np.sum(vacant_taxis)
        
        return total_occupied / total_taxis if total_taxis > 0 else 0
    
    def get_performance_summary(self) -> Dict:
        """
        获取性能摘要报告。
        
        返回:
            dict: 性能摘要
        """
        stats = self.performance_stats
        
        summary = {
            'total_periods_optimized': len(self.solution_history),
            'total_passengers_served': stats['total_passengers_served'],
            'total_swaps_completed': stats['total_swaps_completed'],
            'total_idle_distance': stats['total_idle_distance'],
            'avg_response_time': np.mean(stats['avg_response_time']) if stats['avg_response_time'] else 0,
            'avg_utilization': np.mean(stats['utilization_rates']) if stats['utilization_rates'] else 0,
            'service_efficiency': 0,
            'cost_efficiency': 0
        }
        
        # 计算效率指标
        if stats['total_idle_distance'] > 0:
            summary['service_efficiency'] = stats['total_passengers_served'] / stats['total_idle_distance']
        
        # 时间序列性能
        if len(stats['utilization_rates']) > 1:
            summary['utilization_trend'] = np.polyfit(range(len(stats['utilization_rates'])), stats['utilization_rates'], 1)[0]
        
        return summary
    
    def export_optimization_history(self, filepath: str):
        """
        导出优化历史记录。
        
        参数:
            filepath (str): 导出文件路径
        """
        history_data = {
            'state_history': self.state_history,
            'solution_history': self.solution_history,
            'performance_stats': self.performance_stats,
            'config': {
                'areas': self.config.m_areas,
                'energy_levels': self.config.L_energy_levels,
                'periods': self.config.T_periods,
                'beta': self.config.beta
            }
        }
        
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(history_data, f)
        
        self.logger.info(f"优化历史已导出到: {filepath}")
    
    def reset_performance_stats(self):
        """重置性能统计。"""
        self.performance_stats = {
            'total_passengers_served': 0,
            'total_swaps_completed': 0,
            'total_idle_distance': 0,
            'avg_response_time': [],
            'utilization_rates': []
        }
        self.state_history.clear()
        self.solution_history.clear()
        
        self.logger.info("性能统计已重置")

class AdaptiveOptimizer(JointOptimizer):
    """
    自适应优化器，可以根据系统性能动态调整优化策略。
    """
    
    def __init__(self, config: OptimizationConfig, network_data: Dict):
        super().__init__(config, network_data)
        
        # 自适应参数
        self.adaptation_history = []
        self.performance_threshold = 0.7  # 性能阈值
        self.adaptation_interval = 10  # 每10个周期检查一次
        
    def adapt_strategy(self, current_time: int):
        """
        根据历史性能自适应调整优化策略。
        
        参数:
            current_time (int): 当前时间
        """
        if current_time % self.adaptation_interval != 0:
            return
        
        # 分析最近性能
        recent_utilization = self.performance_stats['utilization_rates'][-self.adaptation_interval:]
        recent_response_time = self.performance_stats['avg_response_time'][-self.adaptation_interval:]
        
        if len(recent_utilization) < self.adaptation_interval:
            return
        
        avg_utilization = np.mean(recent_utilization)
        avg_response_time = np.mean(recent_response_time)
        
        adaptations = []
        
        # 调整beta参数
        if avg_utilization < self.performance_threshold:
            # 利用率低，减少对空驶距离的惩罚
            new_beta = max(-0.05, self.config.beta * 0.8)
            if new_beta != self.config.beta:
                self.config.beta = new_beta
                self.optimization_model.beta = new_beta
                adaptations.append(f"调整beta参数: {new_beta:.3f}")
        
        # 调整求解时间限制
        if avg_response_time > 20:  # 20秒阈值
            # 响应时间过长，减少求解时间
            new_time_limit = max(60, int(self.config.time_limit * 0.8))
            if new_time_limit != self.config.time_limit:
                self.config.time_limit = new_time_limit
                self.optimization_model.solver_config['time_limit'] = new_time_limit
                adaptations.append(f"调整时间限制: {new_time_limit}s")
        
        # 记录自适应历史
        if adaptations:
            self.adaptation_history.append({
                'time': current_time,
                'adaptations': adaptations,
                'performance_trigger': {
                    'utilization': avg_utilization,
                    'response_time': avg_response_time
                }
            })
            
            self.logger.info(f"时间段 {current_time}: 执行自适应调整 - {', '.join(adaptations)}")

if __name__ == "__main__":
    # 测试联合优化器
    print("测试联合优化器...")
    
    # 创建配置
    config = OptimizationConfig(
        m_areas=5,
        L_energy_levels=10,
        T_periods=24,
        period_length_minutes=15,
        beta=-0.1,
        solver_method='heuristic',
        time_limit=60
    )
    
    # 创建网络数据
    network_data = {
        'distance_matrix': {(i, j): abs(i-j) for i in range(5) for j in range(5)},
        'reachability_matrix': {(t, i, j): True for t in range(24) for i in range(5) for j in range(5)}
    }
    
    # 创建优化器
    optimizer = JointOptimizer(config, network_data)
    
    # 添加换电站
    optimizer.add_battery_swap_station({
        'id': 1,
        'location': 2,
        'capacity': 30,
        'chargers': 8
    })
    
    # 设置需求预测
    demand_data = {(t, i): 10 + 5*np.sin(t/24*2*np.pi) for t in range(24) for i in range(5)}
    optimizer.set_demand_forecast(demand_data)
    
    # 模拟几个时间段
    for t in range(3):
        # 创建模拟系统状态
        system_state = {
            'vacant_taxis': np.random.randint(5, 15, (5, 10)),
            'occupied_taxis': np.random.randint(2, 8, (5, 10)),
            'bss_inventories': {1: np.random.randint(2, 8, 10)}
        }
        
        # 优化当前时间段
        decisions = optimizer.optimize_single_period(t, system_state)
        print(f"时间段 {t}: 服务乘客 {np.sum(decisions['served_passengers'])}")
        
        # 优化充电调度
        charging_decisions = optimizer.optimize_charging_schedule(t, {'1': system_state['bss_inventories'][1]})
        print(f"时间段 {t}: 充电决策 {charging_decisions}")
    
    # 获取性能摘要
    summary = optimizer.get_performance_summary()
    print(f"性能摘要: {summary}")
    
    print("联合优化器测试完成!")