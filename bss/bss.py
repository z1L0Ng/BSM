"""电池交换站(BSS)模型。"""
import simpy
import random

class BatterySwapStation:
    def __init__(self, env, station_id, location, capacity=10, initial_batteries=None, 
                 swap_time=5, charge_time=60, n_chargers=1):
        """
        电池交换站模型。
        
        参数:
            env (simpy.Environment): 模拟环境
            station_id (int): 站点标识符
            location (int): 站点所在的区块ID
            capacity (int): 站点可容纳的电池总数
            initial_batteries (int): 初始可用的已充电电池数量(默认为满容量)
            swap_time (float): 执行一次电池更换的时间(分钟)
            charge_time (float): 为一块电池充电的时间(分钟)
            n_chargers (int): 充电单元数量(用于并行充电)
        """
        self.env = env
        self.id = station_id
        self.location = location
        self.capacity = capacity
        
        # SimPy存储器，用于可用的已充电电池
        self.charged_batteries = simpy.Container(env, capacity=capacity)
        # SimPy存储器，用于等待充电的空电池
        self.empty_batteries = simpy.Container(env, capacity=capacity)
        
        # 初始化电池库存
        if initial_batteries is None:
            initial_batteries = capacity // 2  # 默认一半电池是充满电的
        
        # 填充已充电电池存储器
        self.charged_batteries.level = initial_batteries
        # 余下的电池都是空的(已耗尽)
        self.empty_batteries.level = capacity - initial_batteries
        
        # 用于换电区的资源(每个换电区同一时间只能服务一辆出租车)
        self.swap_bay = simpy.Resource(env, capacity=1)
        # 用于充电站的资源(限制并行充电操作)
        self.charger_resource = simpy.Resource(env, capacity=n_chargers)
        
        # 统计追踪
        self.swap_count = 0
        self.wait_times = []  # 记录等待已充电电池的时间(如果有的话)
        self.queue_lengths = []  # 记录队列长度
        self.utilization_samples = []  # 充电器利用率样本
        
        # 电池更换的时间
        self.swap_time = swap_time
        # 充电时间，可以是固定值或分布
        self.charge_time = charge_time
        
        # 启动后台充电进程
        self.charge_process = env.process(self._charging_process())
        
        # 添加统计采样进程
        env.process(self._sample_statistics())
    
    def _sample_statistics(self, interval=60):
        """
        定期采样站点统计数据。
        
        参数:
            interval (int): 采样间隔(分钟)
        """
        while True:
            # 等待指定的间隔时间
            yield self.env.timeout(interval)
            
            # 记录当前队列长度
            queue_length = len(self.swap_bay.queue)
            self.queue_lengths.append(queue_length)
            
            # 记录充电器利用率
            utilization = self.charger_resource.count / self.charger_resource.capacity
            self.utilization_samples.append(utilization)
    
    def _charging_process(self):
        """
        持续为耗尽的电池充电的后台进程。
        """
        while True:
            # 检查是否有空电池需要充电
            if self.empty_batteries.level > 0:
                # 获取充电器资源请求
                with self.charger_resource.request() as req:
                    yield req
                    # 从空电池容器中获取一块电池
                    yield self.empty_batteries.get(1)
                    
                    # 电池充电时间(可以是确定性的或随机的)
                    if isinstance(self.charge_time, tuple):
                        # 使用均匀分布的随机充电时间
                        min_time, max_time = self.charge_time
                        actual_charge_time = random.uniform(min_time, max_time)
                    else:
                        actual_charge_time = self.charge_time
                    
                    # 充电过程(耗时charge_time)
                    yield self.env.timeout(actual_charge_time)
                    
                    # 充电完成后，将电池移到已充电电池存储器
                    yield self.charged_batteries.put(1)
            else:
                # 如果没有空电池，等待一小
                yield self.env.timeout(1)  # 等待1分钟后再检查
            # 充电器利用率统计
            if self.charger_resource.count > 0:
                utilization = self.charger_resource.count / self.charger_resource.capacity
                self.utilization_samples.append(utilization)
            else:
                self.utilization_samples.append(0)
        # 充电进程会持续运行，直到模拟结束

    def swap_battery(self, taxi):
        """
        为到达的出租车更换电池的过程。
        
        参数:
            taxi (TaxiAgent): 需要更换电池的出租车对象
        
        产出:
            SimPy事件
        """
        arrival_time = self.env.now
        
        # 请求访问换电区
        with self.swap_bay.request() as req:
            yield req
            
            # 记录当前队列长度（已经被允许进入换电区）
            queue_length = len(self.swap_bay.queue)
            self.queue_lengths.append(queue_length)
            
            # 检查是否有已充电电池可用
            if self.charged_batteries.level == 0:
                # 没有已充电电池，需要等待
                print(f"时间 {self.env.now:.1f}: 站点 {self.id} 没有可用电池，出租车 {taxi.id} 需要等待")
                
                # 等待直到有已充电电池可用（有充电完成的电池）
                yield self.charged_batteries.get(1)
            else:
                # 有已充电电池可用，直接获取
                yield self.charged_batteries.get(1)
            
            # 计算等待时间
            wait_duration = self.env.now - arrival_time
            self.wait_times.append(wait_duration)
            
            # 将出租车的耗尽电池放入空电池队列等待充电
            yield self.empty_batteries.put(1)
            
            # 模拟电池更换时间
            yield self.env.timeout(self.swap_time)
            
            # 电池更换完成，出租车电池现在充满
            taxi.battery_level = taxi.battery_capacity
            
            # 更新统计数据
            self.swap_count += 1
            
            # 打印日志
            print(f"时间 {self.env.now:.1f}: 站点 {self.id} 完成了出租车 {taxi.id} 的电池更换")
    
    def get_status(self):
        """
        获取电池交换站的当前状态摘要。
            
        返回:
            dict: 包含站点当前状态信息的字典
        """
        avg_wait = sum(self.wait_times) / len(self.wait_times) if self.wait_times else 0
        avg_queue = sum(self.queue_lengths) / len(self.queue_lengths) if self.queue_lengths else 0
        avg_util = sum(self.utilization_samples) / len(self.utilization_samples) if self.utilization_samples else 0
            
        return {
            "id": self.id,
            "location": self.location,
            "charged_batteries": self.charged_batteries.level,
            "empty_batteries": self.empty_batteries.level,
            "total_capacity": self.capacity,
            "swaps_completed": self.swap_count,
            "avg_wait_time": f"{avg_wait:.2f}分钟",
            "avg_queue_length": f"{avg_queue:.2f}辆车",
            "charger_utilization": f"{avg_util*100:.1f}%",
            "chargers": f"{self.charger_resource.count}/{self.charger_resource.capacity} 使用中"
        }