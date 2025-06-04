"""电动出租车代理模型。"""
import simpy
import random
import numpy as np

class TaxiAgent:
    def __init__(self, env, taxi_id, initial_block, battery_capacity=100, consumption_rate=1, swap_threshold=20, real_id=None):
        """
        带有简单状态机运营逻辑的出租车代理。
        
        参数:
            env (simpy.Environment): 模拟环境
            taxi_id (int): 出租车标识符
            initial_block (int): 起始区块位置ID
            battery_capacity (float): 最大电池容量(能量单位)
            consumption_rate (float): 每分钟驾驶的电池消耗率
            swap_threshold (float): 触发电池更换的电池电量阈值
            real_id: 真实的出租车ID（如果有的话）
        """
        self.env = env
        self.id = taxi_id
        self.real_id = real_id
        self.location = initial_block
        self.battery_capacity = battery_capacity
        self.battery_level = battery_capacity  # 开始时电池电量满
        self.consumption_rate = consumption_rate
        self.swap_threshold = swap_threshold
        
        # 运营状态
        self.state = "待命"  # 可能的状态: 待命, 服务中, 前往换电, 换电中
        self.target_station = None  # 目标换电站
        self.swap_order = None  # 当前换电指令
        self.current_passenger = None  # 当前乘客
        
        # 初始化统计数据
        self.reset_stats()
        
        # 时间戳
        self.state_change_time = env.now
        
        # 随机数生成器，用于生成行程属性
        self.rng = random.Random(taxi_id)  # 使用taxi_id作为种子以获得可重复性
    
    def reset_stats(self):
        """重置所有统计数据"""
        self.trips_completed = 0
        self.swap_count = 0
        self.total_distance = 0
        self.total_revenue = 0
        self.idle_time = 0
        self.service_time = 0
        self.charging_time = 0
        self.waiting_time = 0  # 等待充电的时间
    
    def drive_to(self, destination_block, network):
        """
        驾驶到目标区块，消耗时间和电池电量。
        
        参数:
            destination_block: 目标区块ID
            network: 区块网络对象
        """
        if self.location == destination_block:
            return
            
        # 计算距离和时间
        distance = network.distance(self.location, destination_block)
        drive_time = network.travel_time(self.location, destination_block)
        
        # 计算电量消耗（每公里消耗1kWh）
        energy_consumption = distance * self.consumption_rate
        
        # 如果剩余电量不足以完成行程，先去换电
        if self.battery_level - energy_consumption < 0:
            print(f"出租车 {self.id} 电量不足以完成行程，需要先去换电")
            return
        
        # 更新统计数据
        self.total_distance += distance
        
        # 扣除电池电量
        self.battery_level = max(0, self.battery_level - energy_consumption)
        
        # 等待行驶时间
        yield self.env.timeout(drive_time)
        
        # 更新位置
        self.location = destination_block
        
        # 打印调试信息
        print(f"出租车 {self.id} 移动到区块 {destination_block}，剩余电量: {self.battery_level:.1f}%")
    
    def perform_trip(self, network, trip_data=None):
        """
        模拟执行一次乘客行程到随机目的地。
        
        参数:
            network (BlockNetwork): 区块网络对象
            trip_data (DataFrame, optional): 包含实际行程数据的DataFrame
        
        产出:
            SimPy事件
        """
        # 更新状态
        old_state = self.state
        self.state = "服务中"
        if old_state != self.state:
            self.state_change_time = self.env.now
        
        # 确定目的地
        if trip_data is not None and not trip_data.empty:
            # 从真实数据中选择目的地
            trip_sample = trip_data.sample(1).iloc[0]
            dest = int(trip_sample['dropoff_block'] if 'dropoff_block' in trip_sample else trip_sample['block_id'])
            fare = float(trip_sample['fare_amount'] if 'fare_amount' in trip_sample else 10.0)
        else:
            # 如果没有数据，创建随机目的地
            dest = self.location
            while dest == self.location:
                dest = self.rng.choice(list(network.block_positions.keys()))
            # 生成随机车费（基于距离）
            distance = network.distance(self.location, dest)
            fare = max(5.0, 2.5 + distance * 0.5)  # 最低$5，然后按距离增加
        
        # 前往目的地
        yield from self.drive_to(dest, network)
        
        # 记录收入和完成的行程
        self.trips_completed += 1
        self.total_revenue += fare
        
        # 更新状态
        self.state = "待命"
        self.state_change_time = self.env.now
        self.service_time += self.env.now - self.state_change_time
    
    def needs_swap(self):
        """
        检查出租车是否需要更换电池。
        
        返回:
            bool: 如果电池电量低于阈值，返回True
        """
        return self.battery_level <= self.swap_threshold
    
    def assign_swap(self, station):
        """
        分配一个换电站给出租车。
        
        参数:
            station (BatterySwapStation): 指定的换电站
        """
        self.target_station = station
        self.swap_order = {"station": station, "assigned_at": self.env.now}
        self.state = "前往换电"
        self.state_change_time = self.env.now
    
    def go_to_swap(self, network):
        """
        前往指定的换电站换电。
        
        参数:
            network (BlockNetwork): 区块网络对象
        
        产出:
            SimPy事件
        """
        if self.target_station is None:
            return
        
        # 前往换电站
        station_block = self.target_station.location
        yield from self.drive_to(station_block, network)
        
        # 更新状态
        self.state = "换电中"
        self.state_change_time = self.env.now
        
        # 执行电池更换
        swap_start = self.env.now
        yield from self.target_station.swap_battery(self)
        swap_duration = self.env.now - swap_start
        
        # 记录换电次数和等待时间
        self.swap_count += 1
        self.waiting_time += swap_duration
        
        # 重置状态
        self.target_station = None
        self.swap_order = None
        self.state = "待命"
        self.state_change_time = self.env.now
    
    def update_state(self, new_state):
        """更新状态并记录时间"""
        if new_state == self.state:
            return
            
        # 计算在当前状态停留的时间
        elapsed = self.env.now - self.state_change_time
        
        # 更新时间统计
        if self.state == "待命":
            self.idle_time += elapsed
        elif self.state == "服务中":
            self.service_time += elapsed
        elif self.state == "换电中":
            self.charging_time += elapsed
        elif self.state == "前往换电":
            self.idle_time += elapsed  # 前往换电站的时间计入空闲时间
            
        # 更新状态和时间戳
        print(f"时间 {self.env.now:.1f}: 出租车 {self.id} 状态从 {self.state} 变为 {new_state}")
        self.state = new_state
        self.state_change_time = self.env.now

    def process(self, network, stations, trip_data=None):
        """出租车的主要运行过程"""
        self.update_state("待命")  # 初始状态
        
        while True:
            try:
                current_time = self.env.now
                
                # 检查电池状态
                if self.battery_level <= self.swap_threshold and self.state not in ["前往换电", "换电中"]:
                    if not self.target_station:
                        self.update_state("待命")  # 等待分配换电站
                        yield self.env.timeout(1)
                        continue
                    else:
                        self.update_state("前往换电")
                        yield from self.go_to_swap(network)
                        continue
                
                # 正常运营
                if self.state == "待命":
                    # 等待1-5分钟
                    wait_time = random.randint(1, 5)
                    yield self.env.timeout(wait_time)
                    
                    # 尝试接新乘客
                    if self.battery_level > self.swap_threshold:
                        self.update_state("服务中")
                        yield from self.serve_trip(network, trip_data)
                        self.update_state("待命")
                
            except Exception as e:
                print(f"出租车 {self.id} 运行出错: {str(e)}")
                yield self.env.timeout(1)
    
    def get_status(self):
        """
        获取出租车当前状态的摘要。
        
        返回:
            dict: 包含出租车当前状态信息的字典
        """
        return {
            "id": self.id,
            "real_id": self.real_id,
            "location": self.location,
            "battery": f"{self.battery_level:.1f}/{self.battery_capacity}",
            "battery_percent": f"{(self.battery_level/self.battery_capacity)*100:.1f}%",
            "state": self.state,
            "trips": self.trips_completed,
            "swaps": self.swap_count,
            "revenue": f"${self.total_revenue:.2f}",
            "distance": f"{self.total_distance:.1f} km"
        }