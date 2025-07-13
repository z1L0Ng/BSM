# simulation/simulation.py
"""
基于 SimPy 的事件驱动模拟框架。
(已根据Bug分析进行全面修复和重构)
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
        self.swap_events = defaultdict(int)
        self.passengers_served = 0
        self.total_idle_dist = 0
        self.history = []

    def record_wait_time(self, wait):
        if wait > 0:
            self.wait_times.append(wait)

    def record_service(self, dist):
        self.service_times.append(dist)
        self.passengers_served += 1
        
    def record_idle_dist(self, dist):
        self.total_idle_dist += dist

    # 【逻辑修正】添加缺失的 record_swap 方法
    def record_swap(self, station_id):
        """记录一次换电事件。"""
        self.swap_events[station_id] += 1

    def record_system_state(self, taxis, stations):
        state = {
            'time': self.env.now,
            'taxis': [{'id': t.id, 'state': t.state, 'energy': t.energy, 'location': t.location} for t in taxis],
            'stations': [{'id': s.id, 'charged': s.charged_batteries.level, 'empty': s.empty_batteries.level} for s in stations]
        }
        self.history.append(state)

    def get_final_results(self):
        avg_wait = np.mean(self.wait_times) if self.wait_times else 0
        final_inventories = {}
        if self.history:
            final_state = self.history[-1]
            final_inventories = {s['id']: s['charged'] for s in final_state.get('stations', [])}
        return {
            'average_wait_time': avg_wait,
            'total_passengers_served': self.passengers_served,
            'total_idle_distance': self.total_idle_dist,
            'station_swap_counts': dict(self.swap_events),
            'final_station_inventories': final_inventories,
            'simulation_history': self.history
        }

class BatterySwapStation:
    def __init__(self, env, id, location, capacity, initial_charged, num_chargers=2, charge_time=30):
        self.env = env
        self.id = id
        self.location = location
        self.capacity = capacity
        self.charged_batteries = simpy.Container(env, capacity=capacity, init=initial_charged)
        self.empty_batteries = simpy.Container(env, capacity=capacity, init=capacity - initial_charged)
        self.chargers = simpy.Resource(env, capacity=num_chargers)
        self.charge_time = charge_time
        self.env.process(self._charge_process())

    def _charge_process(self):
        while True:
            yield self.empty_batteries.get(1)
            with self.chargers.request() as req:
                yield req
                yield self.env.timeout(self.charge_time)
                yield self.charged_batteries.put(1)

class TaxiAgent:
    def __init__(self, env, id, initial_location, initial_energy, config, stations, travel_model, results_collector):
        self.env = env
        self.id = id
        self.location = initial_location
        self.energy = initial_energy
        self.config = config
        self.stations = stations
        self.travel_model = travel_model
        self.results = results_collector
        self.state = 'idle'
        self.action = env.process(self.run())
        self.target_station = None
        self.task = None

    def run(self):
        while True:
            try:
                if self.state == 'idle':
                    yield self.env.timeout(1)
                
                elif self.state == 'serving_passenger':
                    origin = self.task['origin']
                    dest = self.task['destination']
                    
                    dist_to_origin = self.travel_model.get_distance(self.location, origin)
                    time_to_origin = self.travel_model.get_time(self.location, origin)
                    yield self.env.timeout(time_to_origin)
                    self.location = origin
                    self.results.record_idle_dist(dist_to_origin)

                    dist_to_dest = self.travel_model.get_distance(origin, dest)
                    time_to_dest = self.travel_model.get_time(origin, dest)
                    yield self.env.timeout(time_to_dest)
                    
                    energy_cost = self.travel_model.get_energy_consumption(origin, dest)
                    self.energy -= energy_cost
                    self.location = dest
                    self.results.record_service(dist_to_dest)
                    
                    self.state = 'idle'
                    
                elif self.state == 'going_to_swap':
                    station_loc = self.target_station.location
                    
                    dist_to_station = self.travel_model.get_distance(self.location, station_loc)
                    time_to_station = self.travel_model.get_time(self.location, station_loc)
                    yield self.env.timeout(time_to_station)
                    self.results.record_idle_dist(dist_to_station)
                    self.location = station_loc
                    
                    self.state = 'swapping'
                    self.env.process(self.swap_battery(self.target_station))
                    
                elif self.state == 'swapping':
                    yield self.env.timeout(1)
                else:
                    yield self.env.timeout(1)

            except simpy.Interrupt as interrupt:
                # 任务被中断时，直接进入下一次循环处理新状态
                pass
            except Exception as e:
                logging.error(f"出租车 {self.id} 运行时发生错误: {e}", exc_info=True)
                self.state = 'idle' # 发生错误时，重置为空闲状态以避免卡死


    def swap_battery(self, station):
        if not station:
            self.state = 'idle'
            return

        # 调用现在已存在的方法
        self.results.record_swap(station.id)
        start_wait = self.env.now
        
        # 请求一个满电电池，这是一个 SimPy 进程
        yield station.charged_batteries.get(1)
        self.results.record_wait_time(self.env.now - start_wait)
        
        # 换电操作耗时
        yield self.env.timeout(self.config.swap_duration)
        
        # 将用完的空电池放回“待充电”库存
        yield station.empty_batteries.put(1)
        
        self.energy = self.config.L_energy_levels - 1 # 充满电
        self.state = 'idle' # 切换回空闲状态
        self.target_station = None

    def assign_task(self, task):
        self.task = task
        self.state = task['type']
        if self.state == 'going_to_swap':
            self.target_station = task['station_obj']
        # 中断当前正在执行的动作（例如，空闲等待）
        if self.action.is_alive:
            self.action.interrupt()

class Scheduler:
    def __init__(self, env, config, optimizer, taxis, stations, results_collector):
        self.env = env
        self.config = config
        self.optimizer = optimizer
        self.taxis = taxis
        self.stations = stations
        self.station_map = {s.id: s for s in stations}
        self.results = results_collector
        self.action = env.process(self.run())

    def run(self):
        while self.env.now < self.config.simulation_duration:
            yield self.env.timeout(self.config.reoptimization_interval)
            
            logging.info(f"--- 调度器运行于时间 {self.env.now:.2f} ---")
            
            current_state = self._get_current_system_state()
            self.results.record_system_state(self.taxis, self.stations)

            # 确保传递的时间步是整数
            current_time_step = int(self.env.now // self.config.delta_t)

            decisions = self.optimizer.optimize_single_period(
                t=current_time_step,
                current_state=current_state,
                use_gurobi=False
            )
            self._dispatch_tasks(decisions)
    
    def _get_current_system_state(self):
        vacant_taxis = np.zeros((1, self.config.m_areas, self.config.L_energy_levels))
        for taxi in self.taxis:
            if taxi.state == 'idle':
                loc = taxi.location
                eng = min(int(taxi.energy), self.config.L_energy_levels - 1)
                if 0 <= loc < self.config.m_areas and 0 <= eng < self.config.L_energy_levels:
                    vacant_taxis[0, loc, eng] += 1
        bss_inventories = {s.id: {'charged': s.charged_batteries.level, 'empty': s.empty_batteries.level} for s in self.stations}
        return {'vacant_taxis': vacant_taxis, 'occupied_taxis': np.zeros_like(vacant_taxis), 'bss_inventories': bss_inventories}

    def _dispatch_tasks(self, decisions):
        idle_taxi_pool = defaultdict(list)
        # 只获取真正空闲的出租车
        true_idle_taxis = [t for t in self.taxis if t.state == 'idle']
        
        for taxi in true_idle_taxis:
            loc = taxi.location
            eng = min(int(taxi.energy), self.config.L_energy_levels - 1)
            idle_taxi_pool[(loc, eng)].append(taxi)

        station_locations = {s.location: s for s in self.stations}
        swap_dispatch = decisions['swap_dispatch'][0]
        # 使用 np.argwhere 提高效率
        for i, j_loc, l in np.argwhere(swap_dispatch > 0):
            num_to_dispatch = int(swap_dispatch[i, j_loc, l])
            if j_loc in station_locations:
                for _ in range(num_to_dispatch):
                    if not idle_taxi_pool.get((i, l)): break
                    taxi_to_dispatch = idle_taxi_pool[(i, l)].pop()
                    station_obj = station_locations[j_loc]
                    task = {'type': 'going_to_swap', 'station_obj': station_obj}
                    taxi_to_dispatch.assign_task(task)

        passenger_dispatch = decisions['passenger_dispatch'][0]
        for i, j, l in np.argwhere(passenger_dispatch > 0):
            num_to_dispatch = int(passenger_dispatch[i, j, l])
            for _ in range(num_to_dispatch):
                if not idle_taxi_pool.get((i, l)): break
                taxi_to_dispatch = idle_taxi_pool[(i, l)].pop()
                task = {'type': 'serving_passenger', 'origin': i, 'destination': j}
                taxi_to_dispatch.assign_task(task)

def run_simulation(config, optimizer):
    env = simpy.Environment()
    results = ResultsCollector(env)
    stations = [BatterySwapStation(env, id=s_conf['id'], location=s_conf['location'], capacity=s_conf['capacity'], initial_charged=s_conf['initial_charged']) for s_conf in config.stations]
    
    taxis = [TaxiAgent(env, id=f'taxi_{i}', initial_location=random.randint(0, config.m_areas - 1), initial_energy=random.randint(*config.initial_energy_range), config=config, stations=stations, travel_model=optimizer.travel_model, results_collector=results) for i in range(config.num_taxis)]
        
    Scheduler(env, config, optimizer, taxis, stations, results)
    
    print("--- 模拟开始 ---")
    try:
        env.run(until=config.simulation_duration)
    except Exception as e:
        logging.error(f"SimPy 环境运行时捕获到异常: {e}", exc_info=True)
    print("--- 模拟结束 ---")
    
    final_results = results.get_final_results()
    final_results['stations_config'] = [{'id': s.id, 'location': s.location} for s in stations]
    final_results['taxis_config'] = [{'id': t.id} for t in taxis]
    return final_results