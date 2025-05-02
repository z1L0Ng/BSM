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
            initial_batteries (int): 初始可用的已充电电池数量(默认为半容量)
            swap_time (float): 执行一次电池更换的时间(分钟)
            charge_time (float): 为一块电池充电的时间(分钟)
            n_chargers (int): 并行充电单元数量
        """
        self.env = env
        self.id = station_id
        self.location = location
        self.capacity = capacity

        # 默认初始充电电池数量为容量的一半
        if initial_batteries is None:
            initial_batteries = capacity // 2
        charged_init = initial_batteries
        empty_init = capacity - initial_batteries

        # 使用 SimPy 容器初始化电池库存
        self.charged_batteries = simpy.Container(env, init=charged_init, capacity=capacity)
        self.empty_batteries = simpy.Container(env, init=empty_init, capacity=capacity)

        # 换电区资源
        self.swap_bay = simpy.Resource(env, capacity=1)
        # 充电器资源
        self.charger_resource = simpy.Resource(env, capacity=n_chargers)

        # 统计字段
        self.swap_count = 0
        self.wait_times = []
        self.queue_lengths = []
        self.utilization_samples = []

        # 参数
        self.swap_time = swap_time
        self.charge_time = charge_time

        # 启动后台进程
        self.charge_process = env.process(self._charging_process())
        env.process(self._sample_statistics())

    def _sample_statistics(self, interval=60):
        """
        定期采样队列长度和充电器利用率。
        """
        while True:
            yield self.env.timeout(interval)
            self.queue_lengths.append(len(self.swap_bay.queue))
            util = self.charger_resource.count / self.charger_resource.capacity
            self.utilization_samples.append(util)

    def _charging_process(self):
        """
        持续为空电池充电。
        """
        while True:
            if self.empty_batteries.level > 0:
                with self.charger_resource.request() as req:
                    yield req
                    yield self.empty_batteries.get(1)
                    if isinstance(self.charge_time, tuple):
                        actual = random.uniform(*self.charge_time)
                    else:
                        actual = self.charge_time
                    yield self.env.timeout(actual)
                    yield self.charged_batteries.put(1)
            else:
                yield self.env.timeout(1)
            # 记录利用率
            util = self.charger_resource.count / self.charger_resource.capacity if self.charger_resource.capacity else 0
            self.utilization_samples.append(util)

    def swap_battery(self, taxi):
        """
        执行一次电池交换。
        """
        arrival = self.env.now
        with self.swap_bay.request() as req:
            yield req
            self.queue_lengths.append(len(self.swap_bay.queue))
            if self.charged_batteries.level == 0:
                print(f"时间 {self.env.now:.1f}: 站点 {self.id} 无可用电池，出租车 {taxi.id} 等待")
                yield self.charged_batteries.get(1)
            else:
                yield self.charged_batteries.get(1)
            wait = self.env.now - arrival
            self.wait_times.append(wait)
            yield self.empty_batteries.put(1)
            yield self.env.timeout(self.swap_time)
            taxi.battery_level = taxi.battery_capacity
            self.swap_count += 1
            print(f"时间 {self.env.now:.1f}: 站点 {self.id} 完成出租车 {taxi.id} 电池更换")

    def get_status(self):
        """
        返回站点状态摘要。
        """
        avg_wait = sum(self.wait_times)/len(self.wait_times) if self.wait_times else 0
        avg_queue = sum(self.queue_lengths)/len(self.queue_lengths) if self.queue_lengths else 0
        avg_util  = sum(self.utilization_samples)/len(self.utilization_samples) if self.utilization_samples else 0
        return {
            'id': self.id,
            'location': self.location,
            'charged_batteries': self.charged_batteries.level,
            'empty_batteries': self.empty_batteries.level,
            'total_capacity': self.capacity,
            'swaps_completed': self.swap_count,
            'avg_wait_time': f"{avg_wait:.2f} 分钟",
            'avg_queue_length': f"{avg_queue:.2f} 辆",
            'charger_utilization': f"{avg_util*100:.1f}%",
            'chargers': f"{self.charger_resource.count}/{self.charger_resource.capacity} 使用中"
        }
