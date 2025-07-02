# simulation/simulation.py
"""
基于 SimPy 的事件驱动模拟框架。
"""
import simpy
import random
import logging
import numpy as np
from collections import defaultdict

class ResultsCollector:
    """收集和存储模拟过程中的各项指标。"""
    def __init__(self, env):
        self.env = env
        self.wait_times = []
        self.service_times = []
        self.swap_events = defaultdict(int) # station_id -> count
        self.passengers_served = 0
        self.total_idle_dist = 0
        
        # 用于记录每个时间点的状态
        self.history = []

    def record_wait_time(self, wait):
        self.wait_times.append(wait)

    def record_service(self, dist):
        self.service_times.append(dist)
        self.passengers_served += 1
        
    def record_swap(self, station_id):
        self.swap_events[station_id] += 1
        
    def record_idle_dist(self, dist):
        self.total_idle_dist += dist

    def record_system_state(self, taxis, stations):
        state = {
            'time': self.env.now,
            'taxis': [{'id': t.id, 'state': t.state, 'energy': t.energy, 'location': t.location} for t in taxis],
            'stations': [{'id': s.id, 'charged_batteries': s.charged_batteries.level} for s in stations]
        }
        self.history.append(state)

    def get_final_results(self):
        avg_wait = np.mean(self.wait_times) if self.wait_times else 0
        return {
            'average_wait_time': avg_wait,
            'total_passengers_served': self.passengers_served,
            'total_idle_distance': self.total_idle_dist,
            'station_swap_counts': dict(self.swap_events),
            'simulation_history': self.history
        }

class BatterySwapStation:
    """代表一个换电站，管理电池库存。"""
    def __init__(self, env, id, capacity, initial_charged):
        self.env = env
        self.id = id
        self.capacity = capacity
        # 使用 SimPy Container 来管理满电电池库存
        self.charged_batteries = simpy.Container(env, capacity=capacity, init=initial_charged)
        self.location = int(id.split('_')[1]) # 从ID中提取区域索引

class TaxiAgent:
    """代表一个出租车智能体。"""
    def __init__(self, env, id, initial_location, initial_energy, config, stations, results_collector):
        self.env = env
        self.id = id
        self.location = initial_location
        self.energy = initial_energy
        self.config = config
        self.stations = stations
        self.results = results_collector
        
        self.state = 'idle' # idle, serving_passenger, going_to_swap, swapping
        self.action = env.process(self.run())
        self.target_destination = None
        self.task = None

    def run(self):
        """出租车的主要行为循环。"""
        while True:
            try:
                if self.state == 'idle':
                    # 等待调度中心的指令
                    yield self.env.timeout(1)
                
                elif self.state == 'serving_passenger':
                    # 执行载客任务
                    origin = self.task['origin']
                    dest = self.task['destination']
                    
                    # 1. 前往乘客位置 (空驶)
                    travel_time = self.config.delta_t # 简化: 假设区域内移动耗时1个时间单位
                    yield self.env.timeout(travel_time)
                    self.location = origin
                    self.results.record_idle_dist(1) # 简化: 距离为1

                    # 2. 等待乘客 (可以忽略或设为很小的值)
                    
                    # 3. 前往目的地
                    travel_time = self.config.delta_t * 2 # 简化: 跨区域耗时2个单位
                    yield self.env.timeout(travel_time)
                    self.energy -= 1 # 消耗能量
                    self.location = dest
                    self.results.record_service(1)
                    
                    self.state = 'idle' # 完成任务后变为空闲
                    
                elif self.state == 'going_to_swap':
                    # 前往指定的换电站
                    station_loc = self.target_destination
                    travel_time = self.config.delta_t
                    yield self.env.timeout(travel_time)
                    self.results.record_idle_dist(1)
                    self.location = station_loc
                    
                    # 到达后开始换电
                    self.state = 'swapping'
                    self.env.process(self.swap_battery())
                    
                elif self.state == 'swapping':
                    # 等待换电过程完成
                    yield self.env.timeout(1)
                else:
                    yield self.env.timeout(1)

            except simpy.Interrupt:
                # 当调度器分配新任务时，会中断当前等待
                pass

    def swap_battery(self):
        """换电过程。"""
        station = self.stations[self.location]
        self.results.record_swap(station.id)
        
        # 请求一个满电电池
        start_wait = self.env.now
        yield station.charged_batteries.get(1)
        self.results.record_wait_time(self.env.now - start_wait)
        
        # 换电耗时
        yield self.env.timeout(self.config.swap_duration)
        
        # 换电完成，归还空电池 (增加满电池库存)
        # 简化模型: 假设充电是瞬时的或由另一个过程处理
        yield station.charged_batteries.put(1)
        self.energy = self.config.L_energy_levels - 1 # 充满
        self.state = 'idle'

    def assign_task(self, task):
        self.task = task
        self.state = task['type'] # 'serving_passenger' or 'going_to_swap'
        self.target_destination = task.get('destination')
        self.action.interrupt() # 中断等待，立即开始新任务

class Scheduler:
    """调度器进程，周期性运行优化并分派任务。"""
    def __init__(self, env, config, optimizer, taxis, stations, results_collector):
        self.env = env
        self.config = config
        self.optimizer = optimizer
        self.taxis = taxis
        self.stations = stations
        self.results = results_collector
        self.action = env.process(self.run())

    def run(self):
        while True:
            # 每隔一段时间运行一次优化
            yield self.env.timeout(self.config.reoptimization_interval)
            
            print(f"\n--- 调度器运行于时间 {self.env.now} ---")
            
            # 1. 获取当前系统状态
            current_state = self._get_current_system_state()
            self.results.record_system_state(self.taxis, self.stations)

            # 2. 调用优化器获取决策
            # 使用启发式，因为它更快且我们修复了它的bug
            decisions = self.optimizer.optimize_single_period(
                t=int(self.env.now // self.config.delta_t),
                current_state=current_state,
                use_gurobi=False
            )
            
            # 3. 分派任务给出租车
            self._dispatch_tasks(decisions)
    
    def _get_current_system_state(self):
        """从模拟环境中聚合当前状态。"""
        vacant_taxis = np.zeros((1, self.config.m_areas, self.config.L_energy_levels))
        for taxi in self.taxis:
            if taxi.state == 'idle':
                loc = taxi.location
                eng = min(taxi.energy, self.config.L_energy_levels - 1)
                vacant_taxis[0, loc, eng] += 1
                
        bss_inventories = {}
        for station in self.stations:
            bss_inventories[station.id] = {
                'charged': station.charged_batteries.level,
                'empty': station.capacity - station.charged_batteries.level
            }
        
        return {
            'vacant_taxis': vacant_taxis,
            'occupied_taxis': np.zeros_like(vacant_taxis), # 简化: 优化器当前不使用此项
            'bss_inventories': bss_inventories,
        }

    def _dispatch_tasks(self, decisions):
        """根据优化决策为出租车分配任务。"""
        # 筛选出所有空闲的出租车
        idle_taxis = [t for t in self.taxis if t.state == 'idle']
        random.shuffle(idle_taxis) # 随机化以避免偏差
        
        # 分配换电任务
        swap_dispatch = decisions['swap_dispatch'][0] # t=0
        for i in range(self.config.m_areas):
            for j in range(self.config.m_areas):
                for l in range(self.config.L_energy_levels):
                    num_to_dispatch = int(swap_dispatch[i, j, l])
                    for _ in range(num_to_dispatch):
                        # 找到一辆在区域i，电量为l的空闲出租车
                        taxi_to_dispatch = next((t for t in idle_taxis if t.location == i and t.energy == l), None)
                        if taxi_to_dispatch:
                            task = {'type': 'going_to_swap', 'destination': j}
                            taxi_to_dispatch.assign_task(task)
                            idle_taxis.remove(taxi_to_dispatch) # 从空闲列表中移除

        # 分配载客任务
        passenger_dispatch = decisions['passenger_dispatch'][0] # t=0
        for i in range(self.config.m_areas):
            for j in range(self.config.m_areas):
                 for l in range(self.config.L_energy_levels):
                    num_to_dispatch = int(passenger_dispatch[i, j, l])
                    for _ in range(num_to_dispatch):
                        taxi_to_dispatch = next((t for t in idle_taxis if t.location == i and t.energy == l), None)
                        if taxi_to_dispatch:
                            task = {'type': 'serving_passenger', 'origin': i, 'destination': j}
                            taxi_to_dispatch.assign_task(task)
                            idle_taxis.remove(taxi_to_dispatch)


def run_simulation(config, optimizer):
    """设置并运行 SimPy 模拟。"""
    
    env = simpy.Environment()
    results = ResultsCollector(env)
    
    # 创建换电站
    stations = []
    for s_conf in config.stations:
        station = BatterySwapStation(env, s_conf['id'], s_conf['capacity'], s_conf['initial_charged'])
        stations.append(station)
        
    # 创建出租车
    taxis = []
    for i in range(config.num_taxis):
        taxi = TaxiAgent(
            env=env,
            id=f'taxi_{i}',
            initial_location=random.randint(0, config.m_areas - 1),
            initial_energy=random.randint(config.L_energy_levels // 2, config.L_energy_levels - 1),
            config=config,
            stations=stations,
            results_collector=results
        )
        taxis.append(taxi)
        
    # 创建调度器
    Scheduler(env, config, optimizer, taxis, stations, results)
    
    # 运行模拟
    print("--- 模拟开始 ---")
    env.run(until=config.simulation_duration)
    print("--- 模拟结束 ---")
    
    # 返回最终结果
    final_results = results.get_final_results()
    # 为可视化添加静态信息
    final_results['stations_config'] = [{'id': s.id, 'location': s.location} for s in stations]
    final_results['taxis_config'] = [{'id': t.id} for t in taxis]
    
    # 从历史记录中提取最终的电池状态
    final_state = results.history[-1] if results.history else {}
    final_station_states = {s['id']: s['charged_batteries'] for s in final_state.get('stations', [])}
    final_results['final_station_inventories'] = final_station_states

    return final_results