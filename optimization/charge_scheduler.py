"""充电任务调度器，实现论文第III-D节的充电任务生成。"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class ChargingTask:
    """充电任务类。"""
    task_id: str
    station_id: int
    battery_level: int
    target_level: int
    priority: int
    estimated_time: float
    cost: float
    created_at: datetime

@dataclass
class ElectricityPricing:
    """电价信息类。"""
    time_period: int
    base_price: float
    peak_multiplier: float
    off_peak_discount: float
    renewable_bonus: float

class ChargeTaskGenerator:
    """
    充电任务生成器，根据论文第III-D节实现充电任务的智能生成。
    """
    
    def __init__(self, config: Dict):
        """
        初始化充电任务生成器。
        
        参数:
            config (dict): 配置参数
        """
        self.config = config
        self.L_energy_levels = config['L_energy_levels']
        self.charge_increment = config.get('charge_increment', 1)
        self.base_charge_time = config.get('base_charge_time', 30)  # 基础充电时间(分钟)
        
        # 电价模型
        self.pricing_model = self._create_pricing_model(config.get('pricing', {}))
        
        # 需求预测模型
        self.demand_predictor = DemandPredictor(config.get('demand_prediction', {}))
        
        # 任务历史
        self.task_history = []
        self.completed_tasks = []
        
        # 日志配置
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _create_pricing_model(self, pricing_config: Dict) -> Dict:
        """创建电价模型。"""
        default_pricing = {
            'base_price': 0.12,  # 基础电价 $/kWh
            'peak_hours': [(7, 10), (17, 20)],  # 高峰时段
            'peak_multiplier': 1.8,  # 高峰时段倍数
            'off_peak_discount': 0.7,  # 非高峰折扣
            'renewable_hours': [(11, 15)],  # 可再生能源丰富时段
            'renewable_bonus': 0.8  # 可再生能源折扣
        }
        
        return {**default_pricing, **pricing_config}
    
    def get_electricity_price(self, time_period: int) -> ElectricityPricing:
        """
        获取指定时间段的电价信息。
        
        参数:
            time_period (int): 时间段
        
        返回:
            ElectricityPricing: 电价信息
        """
        hour = (time_period * self.config.get('period_length_minutes', 15) // 60) % 24
        
        base_price = self.pricing_model['base_price']
        peak_multiplier = 1.0
        off_peak_discount = 1.0
        renewable_bonus = 1.0
        
        # 检查是否为高峰时段
        for start, end in self.pricing_model['peak_hours']:
            if start <= hour < end:
                peak_multiplier = self.pricing_model['peak_multiplier']
                break
        
        # 检查是否为非高峰时段
        if peak_multiplier == 1.0:  # 非高峰时段
            off_peak_discount = self.pricing_model['off_peak_discount']
        
        # 检查是否为可再生能源丰富时段
        for start, end in self.pricing_model['renewable_hours']:
            if start <= hour < end:
                renewable_bonus = self.pricing_model['renewable_bonus']
                break
        
        return ElectricityPricing(
            time_period=time_period,
            base_price=base_price,
            peak_multiplier=peak_multiplier,
            off_peak_discount=off_peak_discount,
            renewable_bonus=renewable_bonus
        )
    
    def generate_charging_tasks(self, station_id: int, current_time: int, 
                              battery_inventory: np.ndarray, 
                              demand_forecast: np.ndarray,
                              available_chargers: int) -> List[ChargingTask]:
        """
        生成充电任务列表。
        
        参数:
            station_id (int): 换电站ID
            current_time (int): 当前时间段
            battery_inventory (ndarray): 当前电池库存
            demand_forecast (ndarray): 需求预测
            available_chargers (int): 可用充电器数量
        
        返回:
            List[ChargingTask]: 充电任务列表
        """
        tasks = []
        
        # 计算未来需求和当前库存的缺口
        shortage_analysis = self._analyze_shortage(battery_inventory, demand_forecast)
        
        # 获取当前电价
        current_pricing = self.get_electricity_price(current_time)
        
        # 生成充电任务
        task_count = 0
        for level in range(self.L_energy_levels - self.charge_increment):
            target_level = level + self.charge_increment
            
            # 检查是否需要充电
            current_stock = battery_inventory[level]
            target_shortage = shortage_analysis.get(target_level, 0)
            
            if current_stock > 0 and target_shortage > 0 and task_count < available_chargers:
                # 计算充电数量
                charge_quantity = min(current_stock, target_shortage, available_chargers - task_count)
                
                for i in range(int(charge_quantity)):
                    task = self._create_charging_task(
                        station_id, level, target_level, current_time, current_pricing
                    )
                    tasks.append(task)
                    task_count += 1
        
        # 按优先级排序任务
        tasks.sort(key=lambda x: (-x.priority, x.cost))
        
        self.logger.info(f"站点 {station_id}: 生成 {len(tasks)} 个充电任务")
        
        return tasks
    
    def _analyze_shortage(self, current_inventory: np.ndarray, 
                         demand_forecast: np.ndarray) -> Dict[int, float]:
        """
        分析库存缺口。
        
        参数:
            current_inventory (ndarray): 当前库存
            demand_forecast (ndarray): 需求预测
        
        返回:
            dict: 各能量等级的缺口
        """
        shortage = {}
        
        # 累计未来需求
        total_demand = np.sum(demand_forecast, axis=0) if demand_forecast.ndim > 1 else demand_forecast
        
        for level in range(self.L_energy_levels):
            current_stock = current_inventory[level]
            expected_demand = total_demand[level] if level < len(total_demand) else 0
            
            # 计算缺口，包括安全库存
            safety_stock = max(2, expected_demand * 0.2)  # 20%安全库存
            total_needed = expected_demand + safety_stock
            
            if total_needed > current_stock:
                shortage[level] = total_needed - current_stock
        
        return shortage
    
    def _create_charging_task(self, station_id: int, from_level: int, to_level: int,
                            current_time: int, pricing: ElectricityPricing) -> ChargingTask:
        """
        创建单个充电任务。
        
        参数:
            station_id (int): 站点ID
            from_level (int): 起始能量等级
            to_level (int): 目标能量等级
            current_time (int): 当前时间
            pricing (ElectricityPricing): 电价信息
        
        返回:
            ChargingTask: 充电任务
        """
        # 生成任务ID
        task_id = f"CHG_{station_id}_{current_time}_{from_level}_{to_level}_{len(self.task_history)}"
        
        # 计算充电时间
        level_diff = to_level - from_level
        estimated_time = self.base_charge_time * level_diff
        
        # 计算充电成本
        energy_per_level = 20.0  # 每个能量等级对应20kWh
        energy_cost = energy_per_level * level_diff
        
        # 应用电价
        final_price = (pricing.base_price * 
                      pricing.peak_multiplier * 
                      pricing.off_peak_discount * 
                      pricing.renewable_bonus)
        
        total_cost = energy_cost * final_price
        
        # 计算优先级
        priority = self._calculate_task_priority(from_level, to_level, current_time, pricing)
        
        task = ChargingTask(
            task_id=task_id,
            station_id=station_id,
            battery_level=from_level,
            target_level=to_level,
            priority=priority,
            estimated_time=estimated_time,
            cost=total_cost,
            created_at=datetime.now()
        )
        
        return task
    
    def _calculate_task_priority(self, from_level: int, to_level: int, 
                               current_time: int, pricing: ElectricityPricing) -> int:
        """
        计算充电任务优先级。
        
        参数:
            from_level (int): 起始能量等级
            to_level (int): 目标能量等级
            current_time (int): 当前时间
            pricing (ElectricityPricing): 电价信息
        
        返回:
            int: 优先级分数 (越高越优先)
        """
        priority = 0
        
        # 基础优先级：低能量等级优先
        priority += (self.L_energy_levels - from_level) * 10
        
        # 目标等级优先级：高目标等级优先
        priority += to_level * 5
        
        # 电价优先级：低电价时段优先
        price_factor = (pricing.off_peak_discount * pricing.renewable_bonus) / pricing.peak_multiplier
        priority += int(price_factor * 20)
        
        # 时间优先级：避免高峰时段
        hour = (current_time * self.config.get('period_length_minutes', 15) // 60) % 24
        if 7 <= hour < 10 or 17 <= hour < 20:  # 高峰时段
            priority -= 15
        elif 23 <= hour or hour < 6:  # 深夜时段
            priority += 10
        
        return max(0, priority)
    
    def optimize_charging_schedule(self, tasks: List[ChargingTask], 
                                 available_chargers: int,
                                 time_horizon: int = 6) -> List[ChargingTask]:
        """
        优化充电调度。
        
        参数:
            tasks (List[ChargingTask]): 候选任务列表
            available_chargers (int): 可用充电器数量
            time_horizon (int): 时间窗口(小时)
        
        返回:
            List[ChargingTask]: 优化后的任务调度
        """
        if not tasks:
            return []
        
        # 按优先级和成本排序
        sorted_tasks = sorted(tasks, key=lambda x: (-x.priority, x.cost))
        
        # 贪心选择任务
        selected_tasks = []
        used_chargers = 0
        total_time = 0
        
        for task in sorted_tasks:
            if used_chargers < available_chargers:
                # 检查时间窗口约束
                if total_time + task.estimated_time <= time_horizon * 60:  # 转换为分钟
                    selected_tasks.append(task)
                    used_chargers += 1
                    total_time += task.estimated_time
        
        # 记录任务历史
        self.task_history.extend(selected_tasks)
        
        self.logger.info(f"优化调度: 选择 {len(selected_tasks)}/{len(tasks)} 个任务")
        
        return selected_tasks
    
    def execute_charging_tasks(self, tasks: List[ChargingTask], 
                             current_inventory: np.ndarray) -> Tuple[np.ndarray, List[ChargingTask]]:
        """
        执行充电任务。
        
        参数:
            tasks (List[ChargingTask]): 要执行的任务
            current_inventory (ndarray): 当前库存
        
        返回:
            tuple: (更新后的库存, 完成的任务)
        """
        updated_inventory = current_inventory.copy()
        completed_tasks = []
        
        for task in tasks:
            from_level = task.battery_level
            to_level = task.target_level
            
            # 检查是否有足够的电池可以充电
            if updated_inventory[from_level] > 0:
                # 执行充电
                updated_inventory[from_level] -= 1
                updated_inventory[to_level] += 1
                
                # 标记任务完成
                task.completed_at = datetime.now()
                completed_tasks.append(task)
                
                self.logger.debug(f"完成充电任务: {task.task_id}")
        
        self.completed_tasks.extend(completed_tasks)
        
        return updated_inventory, completed_tasks
    
    def get_charging_recommendations(self, station_id: int, current_time: int,
                                   battery_inventory: np.ndarray,
                                   historical_demand: List[np.ndarray]) -> Dict:
        """
        获取充电建议。
        
        参数:
            station_id (int): 站点ID
            current_time (int): 当前时间
            battery_inventory (ndarray): 当前库存
            historical_demand (List[ndarray]): 历史需求数据
        
        返回:
            dict: 充电建议
        """
        # 预测未来需求
        demand_forecast = self.demand_predictor.predict(historical_demand, horizon=6)
        
        # 分析当前状态
        current_pricing = self.get_electricity_price(current_time)
        shortage_analysis = self._analyze_shortage(battery_inventory, demand_forecast)
        
        # 生成建议
        recommendations = {
            'immediate_actions': [],
            'scheduled_actions': [],
            'cost_optimization': [],
            'risk_warnings': []
        }
        
        # 即时行动建议
        critical_shortage = sum(1 for shortage in shortage_analysis.values() if shortage > 5)
        if critical_shortage > 0:
            recommendations['immediate_actions'].append({
                'type': 'urgent_charging',
                'message': f'{critical_shortage} 个能量等级严重缺货，建议立即充电',
                'priority': 'high'
            })
        
        # 成本优化建议
        if current_pricing.off_peak_discount < 0.8:  # 低电价时段
            recommendations['cost_optimization'].append({
                'type': 'cost_saving',
                'message': f'当前电价较低 ({current_pricing.off_peak_discount:.2f}折扣)，建议增加充电',
                'priority': 'medium'
            })
        
        # 风险警告
        low_inventory_levels = [i for i, count in enumerate(battery_inventory) if count < 2]
        if len(low_inventory_levels) > 3:
            recommendations['risk_warnings'].append({
                'type': 'low_inventory',
                'message': f'多个能量等级库存过低: {low_inventory_levels}',
                'priority': 'high'
            })
        
        return recommendations

class DemandPredictor:
    """需求预测器。"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.prediction_method = config.get('method', 'moving_average')
        self.window_size = config.get('window_size', 5)
    
    def predict(self, historical_demand: List[np.ndarray], horizon: int = 6) -> np.ndarray:
        """
        预测未来需求。
        
        参数:
            historical_demand (List[ndarray]): 历史需求数据
            horizon (int): 预测时域
        
        返回:
            ndarray: 预测需求
        """
        if not historical_demand:
            return np.zeros((horizon, len(historical_demand[0]) if historical_demand else 10))
        
        if self.prediction_method == 'moving_average':
            return self._moving_average_predict(historical_demand, horizon)
        elif self.prediction_method == 'exponential_smoothing':
            return self._exponential_smoothing_predict(historical_demand, horizon)
        else:
            return self._simple_repeat_predict(historical_demand, horizon)
    
    def _moving_average_predict(self, historical_demand: List[np.ndarray], horizon: int) -> np.ndarray:
        """移动平均预测。"""
        recent_demand = historical_demand[-self.window_size:]
        avg_demand = np.mean(recent_demand, axis=0)
        
        # 简单重复平均值
        return np.tile(avg_demand, (horizon, 1))
    
    def _exponential_smoothing_predict(self, historical_demand: List[np.ndarray], horizon: int) -> np.ndarray:
        """指数平滑预测。"""
        alpha = 0.3  # 平滑参数
        
        if len(historical_demand) == 1:
            return np.tile(historical_demand[0], (horizon, 1))
        
        # 计算指数平滑值
        smoothed = historical_demand[0].copy()
        for demand in historical_demand[1:]:
            smoothed = alpha * demand + (1 - alpha) * smoothed
        
        return np.tile(smoothed, (horizon, 1))
    
    def _simple_repeat_predict(self, historical_demand: List[np.ndarray], horizon: int) -> np.ndarray:
        """简单重复预测。"""
        return np.tile(historical_demand[-1], (horizon, 1))

class ChargingCostOptimizer:
    """充电成本优化器。"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.cost_threshold = config.get('cost_threshold', 100.0)
        self.optimization_method = config.get('method', 'greedy')
    
    def optimize_cost(self, tasks: List[ChargingTask], 
                     future_pricing: List[ElectricityPricing]) -> List[ChargingTask]:
        """
        优化充电成本。
        
        参数:
            tasks (List[ChargingTask]): 原始任务列表
            future_pricing (List[ElectricityPricing]): 未来电价
        
        返回:
            List[ChargingTask]: 成本优化后的任务
        """
        if self.optimization_method == 'greedy':
            return self._greedy_cost_optimization(tasks, future_pricing)
        elif self.optimization_method == 'dynamic_programming':
            return self._dp_cost_optimization(tasks, future_pricing)
        else:
            return tasks
    
    def _greedy_cost_optimization(self, tasks: List[ChargingTask], 
                                future_pricing: List[ElectricityPricing]) -> List[ChargingTask]:
        """贪心成本优化。"""
        # 根据成本效益比排序
        cost_efficiency = []
        for task in tasks:
            efficiency = task.priority / max(task.cost, 0.01)  # 避免除零
            cost_efficiency.append((task, efficiency))
        
        # 按效益排序
        cost_efficiency.sort(key=lambda x: x[1], reverse=True)
        
        # 选择成本效益最高的任务
        optimized_tasks = []
        total_cost = 0
        
        for task, efficiency in cost_efficiency:
            if total_cost + task.cost <= self.cost_threshold:
                optimized_tasks.append(task)
                total_cost += task.cost
        
        return optimized_tasks
    
    def _dp_cost_optimization(self, tasks: List[ChargingTask], 
                            future_pricing: List[ElectricityPricing]) -> List[ChargingTask]:
        """动态规划成本优化。"""
        # 简化的动态规划实现
        n = len(tasks)
        if n == 0:
            return []
        
        # DP表：dp[i][cost] = 最大优先级
        max_cost = int(self.cost_threshold)
        dp = [[0] * (max_cost + 1) for _ in range(n + 1)]
        
        # 填充DP表
        for i in range(1, n + 1):
            task = tasks[i - 1]
            task_cost = int(min(task.cost, max_cost))
            
            for cost in range(max_cost + 1):
                # 不选择当前任务
                dp[i][cost] = dp[i - 1][cost]
                
                # 选择当前任务
                if cost >= task_cost:
                    dp[i][cost] = max(dp[i][cost], 
                                    dp[i - 1][cost - task_cost] + task.priority)
        
        # 回溯找到最优解
        selected_tasks = []
        cost = max_cost
        for i in range(n, 0, -1):
            if dp[i][cost] != dp[i - 1][cost]:
                selected_tasks.append(tasks[i - 1])
                cost -= int(min(tasks[i - 1].cost, max_cost))
        
        return selected_tasks

if __name__ == "__main__":
    # 测试充电任务调度器
    print("测试充电任务调度器...")
    
    # 创建配置
    config = {
        'L_energy_levels': 10,
        'charge_increment': 1,
        'base_charge_time': 30,
        'period_length_minutes': 15,
        'pricing': {
            'base_price': 0.12,
            'peak_hours': [(7, 10), (17, 20)],
            'peak_multiplier': 1.8,
            'off_peak_discount': 0.7
        },
        'demand_prediction': {
            'method': 'moving_average',
            'window_size': 5
        }
    }
    
    # 创建充电任务生成器
    task_generator = ChargeTaskGenerator(config)
    
    # 模拟当前状态
    current_time = 10  # 时间段10
    station_id = 1
    battery_inventory = np.array([5, 3, 2, 4, 6, 5, 4, 3, 2, 1])  # 各能量等级库存
    demand_forecast = np.array([3, 2, 4, 5, 3, 2, 1, 2, 3, 4])  # 需求预测
    available_chargers = 5
    
    # 生成充电任务
    tasks = task_generator.generate_charging_tasks(
        station_id, current_time, battery_inventory, demand_forecast, available_chargers
    )
    
    print(f"生成了 {len(tasks)} 个充电任务")
    for task in tasks[:3]:  # 显示前3个任务
        print(f"任务 {task.task_id}: {task.battery_level}->{task.target_level}, "
              f"优先级: {task.priority}, 成本: ${task.cost:.2f}")
    
    # 优化充电调度
    optimized_tasks = task_generator.optimize_charging_schedule(tasks, available_chargers)
    print(f"优化后选择了 {len(optimized_tasks)} 个任务")
    
    # 执行充电任务
    updated_inventory, completed_tasks = task_generator.execute_charging_tasks(
        optimized_tasks, battery_inventory
    )
    
    print(f"完成了 {len(completed_tasks)} 个充电任务")
    print(f"库存变化: {battery_inventory} -> {updated_inventory}")
    
    # 获取充电建议
    historical_demand = [demand_forecast * 0.9, demand_forecast, demand_forecast * 1.1]
    recommendations = task_generator.get_charging_recommendations(
        station_id, current_time, updated_inventory, historical_demand
    )
    
    print("充电建议:")
    for category, items in recommendations.items():
        if items:
            print(f"  {category}: {len(items)} 项建议")
    
    print("充电任务调度器测试完成!")