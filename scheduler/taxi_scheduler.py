"""电动出租车车队的中央调度器。"""
import simpy
import random
import numpy as np
import pandas as pd

class ETaxiScheduler:
    def __init__(self, env, taxis, stations, network, check_interval=5, 
                 use_real_data=False, trip_data=None):
        """
        参数:
        ----------
        env : simpy.Environment
            模拟环境
        taxis : list[TaxiAgent]
            出租车对象列表
        stations : list[BatterySwapStation]
            换电站对象列表
        network : BlockNetwork
            区块网络对象
        check_interval : int
            连续调度决策之间的分钟数
        use_real_data : bool
            是否使用真实出行数据
        trip_data : DataFrame
            真实出行数据(如果use_real_data=True)
        """
        self.env = env
        self.taxis = taxis
        self.stations = stations
        self.network = network
        self.interval = check_interval
        
        # 调度参数
        self.use_real_data = use_real_data
        self.trip_data = trip_data
        
        # 状态跟踪
        self.reset_stats()
        
        # 调度策略配置
        self.proactive_swap = True   # 主动换电(在电池电量较低时)
        self.load_balancing = True   # 站点负载均衡
        self.min_battery_threshold = 25  # 主动换电的电池电量阈值(%)
        
        # 权重配置
        self.distance_weight = 0.4    # 距离权重
        self.battery_weight = 0.2     # 电池库存权重
        self.queue_weight = 0.2       # 队列长度权重
        self.wait_time_weight = 0.2   # 等待时间权重
        
        # 等待时间追踪
        self.station_wait_times = {station.id: [] for station in stations}  # 每个站点的历史等待时间
        self.wait_time_window = 60    # 计算平均等待时间的时间窗口（分钟）
        self.max_acceptable_wait = 180 # 最大可接受等待时间（秒）
        
        # 将出租车ID映射到其对象的字典，用于快速查找
        self.taxi_dict = {taxi.id: taxi for taxi in taxis}
        # 将站点ID映射到其对象的字典
        self.station_dict = {station.id: station for station in stations}
        
        # 启动调度进程
        self.schedule_process = env.process(self._run())
        
        # 添加状态记录进程
        self.stats_process = env.process(self._record_stats())
    
    def _record_stats(self, interval=60):
        """
        定期记录系统状态统计信息。
        
        参数:
            interval (int): 记录间隔(分钟)
        """
        stats_history = []
        
        while True:
            # 等待指定的间隔时间
            yield self.env.timeout(interval)
            
            # 收集当前状态
            current_time = self.env.now
            current_hour = int(current_time / 60) % 24
            
            # 出租车状态
            active_taxis = len(self.taxis)
            idle_taxis = sum(1 for taxi in self.taxis if taxi.state == "待命")
            serving_taxis = sum(1 for taxi in self.taxis if taxi.state == "服务中")
            charging_taxis = sum(1 for taxi in self.taxis if taxi.state in ["前往换电", "换电中"])
            
            # 站点状态
            total_charged = sum(station.charged_batteries.level for station in self.stations)
            total_empty = sum(station.empty_batteries.level for station in self.stations)
            total_capacity = sum(station.capacity for station in self.stations)
            
            # 计算性能指标
            taxi_utilization = serving_taxis / active_taxis if active_taxis > 0 else 0
            battery_availability = total_charged / total_capacity if total_capacity > 0 else 0
            
            # 记录统计数据
            stats = {
                "time": current_time,
                "hour": current_hour,
                "active_taxis": active_taxis,
                "idle_taxis": idle_taxis,
                "serving_taxis": serving_taxis,
                "charging_taxis": charging_taxis,
                "taxi_utilization": taxi_utilization,
                "total_charged_batteries": total_charged,
                "total_empty_batteries": total_empty,
                "battery_availability": battery_availability,
                "total_assignments": self.total_assignments,
                "swap_assignments": self.swap_assignments,
                "trip_assignments": self.trip_assignments
            }
            
            stats_history.append(stats)
            
            # 打印一些关键统计信息
            print(f"\n--- 系统状态 @ {current_time:.1f}分钟 (小时: {current_hour}) ---")
            print(f"出租车: {idle_taxis}空闲/{serving_taxis}服务/{charging_taxis}充电 (利用率: {taxi_utilization*100:.1f}%)")
            print(f"电池: {total_charged}已充电/{total_empty}待充电/{total_capacity}总容量 (可用率: {battery_availability*100:.1f}%)")
            print(f"调度: {self.total_assignments}总计, {self.swap_assignments}换电, {self.trip_assignments}行程")
    
    # ---------------- 调度器循环 ---------------- #
    def _run(self):
        """调度器主循环进程"""
        while True:
            # 等待指定的间隔时间
            yield self.env.timeout(self.interval)
            
            # 更新网络中的当前时间(用于考虑交通拥堵)
            current_hour = int(self.env.now / 60) % 24
            self.network.update_time(current_hour)
            
            # 执行调度决策
            self.dispatch_taxis()
    
    # ------------- 核心决策逻辑 ------------- #
    def dispatch_taxis(self):
        """
        执行出租车调度，包括：
        1. 为需要换电的出租车分配换电站
        2. 为空闲的出租车分配行程(如果使用真实数据)
        """
        # 第一步：处理需要换电的出租车
        self.dispatch_swaps()
        
        # 第二步：如果使用真实数据，为空闲出租车分配行程
        if self.use_real_data and self.trip_data is not None:
            self.dispatch_trips()
    
    def dispatch_swaps(self):
        """
        换电调度策略:
        1. 找出需要换电的出租车(电池电量 <= 阈值 & 未收到指令)
        2. 为每辆车挑选最合适的换电站，考虑:
           - 站点距离
           - 已充电电池库存
           - 站点当前排队情况
        3. 发送换电指令
        """
        for taxi in self.taxis:
            # 检查出租车是否需要换电或应该主动换电
            needs_swap = taxi.needs_swap()  # 电池电量低于最低阈值
            should_proactive_swap = (self.proactive_swap and 
                                    taxi.state == "待命" and
                                    taxi.battery_level / taxi.battery_capacity * 100 <= self.min_battery_threshold)
            
            if (needs_swap or should_proactive_swap) and taxi.swap_order is None:
                # 过滤出有已充电电池的站点
                candidate_stations = [
                    s for s in self.stations
                    if s.charged_batteries.level > 0
                ]
                
                if not candidate_stations:
                    continue  # 暂时没有可用电池
                
                # 如果启用了负载均衡，使用综合评分选择站点
                if self.load_balancing:
                    best_station = self._select_best_station(taxi, candidate_stations)
                else:
                    # 简单地选择最近且有库存的站点
                    best_station = min(
                        candidate_stations,
                        key=lambda s: self.network.distance(taxi.location, s.location)
                    )
                
                # 分配换电站
                taxi.assign_swap(best_station)
                self.swap_assignments += 1
                self.total_assignments += 1
                
                # 记录日志
                battery_percent = (taxi.battery_level / taxi.battery_capacity) * 100
                print(f"时间 {self.env.now:.1f}: 为出租车 {taxi.id} (电量: {battery_percent:.1f}%) "
                      f"分配换电站 {best_station.id}, 距离: {self.network.distance(taxi.location, best_station.location):.1f}")
    
    def _select_best_station(self, taxi, candidate_stations):
        """
        为出租车选择最佳换电站，综合考虑：
        1. 距离（越近越好）
        2. 电池库存（越多越好）
        3. 当前队列长度（越短越好）
        4. 历史等待时间（越短越好）
        """
        scores = []
        
        # 获取所有站点的平均等待时间
        avg_wait_times = self._get_station_wait_times()
        max_wait_time = max(avg_wait_times.values()) if avg_wait_times else self.max_acceptable_wait
        
        for station in candidate_stations:
            # 1. 距离分数 (越近越好)
            distance = self.network.distance(taxi.location, station.location)
            max_distance = 100  # 假设的最大距离
            distance_score = 1 - min(distance / max_distance, 1.0)
            
            # 2. 电池库存分数 (库存越多越好)
            battery_inventory = station.charged_batteries.level
            max_inventory = station.capacity
            battery_score = battery_inventory / max_inventory
            
            # 3. 队列长度分数 (队列越短越好)
            queue_length = len(station.swap_bay.queue)
            max_queue = 10  # 假设的最大队列长度
            queue_score = 1 - min(queue_length / max_queue, 1.0)
            
            # 4. 等待时间分数 (等待时间越短越好)
            avg_wait = avg_wait_times.get(station.id, 0)
            wait_time_score = 1 - min(avg_wait / max_wait_time, 1.0)
            
            # 动态调整等待时间权重 - 当等待时间超过阈值时增加其权重
            dynamic_wait_weight = self.wait_time_weight
            if avg_wait > self.max_acceptable_wait / 2:  # 如果等待时间超过阈值的一半
                dynamic_wait_weight = min(0.4, self.wait_time_weight * 1.5)  # 增加权重但不超过0.4
                
            # 重新平衡其他权重
            remaining_weight = 1.0 - dynamic_wait_weight
            adjusted_distance_weight = self.distance_weight * (remaining_weight / (1 - self.wait_time_weight))
            adjusted_battery_weight = self.battery_weight * (remaining_weight / (1 - self.wait_time_weight))
            adjusted_queue_weight = self.queue_weight * (remaining_weight / (1 - self.wait_time_weight))
            
            # 计算综合评分
            composite_score = (
                adjusted_distance_weight * distance_score +
                adjusted_battery_weight * battery_score +
                adjusted_queue_weight * queue_score +
                dynamic_wait_weight * wait_time_score
            )
            
            scores.append((station, composite_score))
        
        # 选择评分最高的站点
        if not scores:
            return None
        
        best_station = max(scores, key=lambda x: x[1])[0]
        return best_station

    def _get_station_wait_times(self):
        """
        计算每个站点的平均等待时间
        返回: dict[station_id: float] - 每个站点的平均等待时间(秒)
        """
        current_time = self.env.now
        avg_wait_times = {}
        
        for station_id, wait_times in self.station_wait_times.items():
            # 清理旧数据 - 只保留时间窗口内的数据
            recent_times = [(t, w) for t, w in wait_times 
                          if current_time - t <= self.wait_time_window]
            self.station_wait_times[station_id] = recent_times
            
            if recent_times:
                avg_wait = sum(w for _, w in recent_times) / len(recent_times)
                avg_wait_times[station_id] = avg_wait
            else:
                avg_wait_times[station_id] = 0
        
        return avg_wait_times
    
    def dispatch_trips(self):
        """
        如果使用真实数据，为空闲出租车分配行程。
        根据当前小时筛选相关行程，并按照就近原则分配。
        """
        # 获取当前小时
        current_hour = int(self.env.now / 60) % 24
        
        # 找到所有空闲且电池电量充足的出租车
        available_taxis = [
            taxi for taxi in self.taxis
            if taxi.state == "待命" and 
            taxi.battery_level / taxi.battery_capacity > 0.3  # 至少30%电量
        ]
        
        if not available_taxis or self.trip_data is None:
            return
        
        # 筛选当前小时的行程数据
        if 'hour' in self.trip_data.columns:
            current_trips = self.trip_data[self.trip_data['hour'] == current_hour].copy()
        else:
            current_trips = self.trip_data.copy()
        
        if current_trips.empty:
            return
        
        # 为每辆空闲出租车分配一个近距离行程
        for taxi in available_taxis:
            # 计算出租车到每个上车点的距离
            current_trips['distance_to_taxi'] = current_trips['block_id'].apply(
                lambda block: self.network.distance(taxi.location, block)
            )
            
            # 找到最近的行程
            nearest_trip = current_trips.nsmallest(1, 'distance_to_taxi')
            
            if not nearest_trip.empty:
                # 从数据中移除已分配的行程(避免重复分配)
                trip_idx = nearest_trip.index[0]
                trip = current_trips.loc[trip_idx]
                current_trips = current_trips.drop(trip_idx)
                
                # 获取行程信息
                pickup_block = int(trip['block_id'])
                
                # 为出租车分配行程(在实际应用中，这里需要与taxi.process()进程协调)
                # 这里只是模拟分配过程
                print(f"时间 {self.env.now:.1f}: 为出租车 {taxi.id} 分配行程，"
                      f"从区块 {taxi.location} 到 {pickup_block}，距离: {trip['distance_to_taxi']:.1f}")
                
                self.trip_assignments += 1
                self.total_assignments += 1
                
                # 注意：这里不直接修改出租车状态，因为这需要与taxi.process()进行协调
                # 在实际实现中，可能需要使用事件或消息机制
    
    def reset_stats(self):
        """重置所有统计数据"""
        self.total_assignments = 0  # 总调度次数
        self.swap_assignments = 0   # 换电调度次数
        self.trip_assignments = 0   # 行程调度次数
        self.rejected_assignments = 0  # 拒绝的调度请求
        self.avg_response_time = []  # 平均响应时间
        self.station_loads = {}  # 各站点的负载情况
        
    def get_status(self):
        """返回调度器状态的快照"""
        return {
            'total_assignments': self.total_assignments,
            'swap_assignments': self.swap_assignments,
            'trip_assignments': self.trip_assignments,
            'rejected_assignments': self.rejected_assignments,
            'avg_response_time': sum(self.avg_response_time)/len(self.avg_response_time) if self.avg_response_time else 0,
            'station_loads': self.station_loads.copy() if hasattr(self, 'station_loads') else {}
        }

    def stop(self):
        """停止调度器并清理资源"""
        pass  # 如果需要清理资源，在这里添加代码
    
    def record_swap_wait_time(self, station_id, wait_time):
        """
        记录一次换电等待时间
        
        参数:
            station_id: 换电站ID
            wait_time: 等待时间（秒）
        """
        current_time = self.env.now
        if station_id in self.station_wait_times:
            self.station_wait_times[station_id].append((current_time, wait_time))
            
            # 如果站点等待时间持续超高，动态调整该站点的接受新车的阈值
            avg_times = self._get_station_wait_times()
            if avg_times.get(station_id, 0) > self.max_acceptable_wait:
                station = self.station_dict.get(station_id)
                if station:
                    # 临时降低站点接受新车的概率
                    station.acceptance_rate = 0.5  # 50%的概率拒绝新车
            else:
                # 恢复正常接受率
                station = self.station_dict.get(station_id)
                if station:
                    station.acceptance_rate = 1.0