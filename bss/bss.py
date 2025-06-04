"""电池交换站(BSS)模型。"""
import simpy
import random

class BatterySwapStation:
    def __init__(self, env, station_id, location, capacity=10, initial_batteries=None, 
                 swap_time=5, charge_time=60, n_chargers=1, scheduler=None):
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
            scheduler (Scheduler): 调度器引用，用于报告等待时间
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
        self.reset_stats()

        # 参数
        self.swap_time = swap_time
        self.charge_time = charge_time

        # 调度器引用（用于报告等待时间）
        self.scheduler = scheduler
        
        # 等待时间和负载管理
        self.acceptance_rate = 1.0  # 默认接受所有请求
        self.current_waiting_taxis = {}  # 记录每辆出租车的等待开始时间
        
        # 启动后台进程
        self.charge_process = env.process(self._charging_process())
        env.process(self._sample_statistics())
        env.process(self._monitor_wait_times())

    def reset_stats(self):
        """重置所有统计数据"""
        self.swap_count = 0
        self.wait_times = []
        self.queue_lengths = []
        self.utilization_samples = []
        self.total_charging_time = 0
        self.total_swap_time = 0
        self.peak_queue_length = 0

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
        持续为空电池充电的后台进程。
        """
        while True:
            if self.empty_batteries.level > 0:
                with self.charger_resource.request() as req:
                    yield req
                    
                    # 获取一个空电池
                    yield self.empty_batteries.get(1)
                    
                    # 确定充电时间
                    if isinstance(self.charge_time, tuple):
                        actual_charge_time = random.uniform(*self.charge_time)
                    else:
                        actual_charge_time = self.charge_time
                        
                    # 充电过程
                    self.total_charging_time += actual_charge_time
                    yield self.env.timeout(actual_charge_time)
                    
                    # 将充好的电池放入充电电池库存
                    yield self.charged_batteries.put(1)
                    
                    # 更新充电器利用率
                    util = self.charger_resource.count / self.charger_resource.capacity
                    self.utilization_samples.append(util)
            else:
                # 如果没有空电池，等待一分钟再检查
                yield self.env.timeout(1)

    def _monitor_wait_times(self, check_interval=1):
        """
        监控等待时间的后台进程
        """
        while True:
            yield self.env.timeout(check_interval)
            current_time = self.env.now
            
            # 更新当前等待时间
            for taxi_id in list(self.current_waiting_taxis.keys()):
                start_time = self.current_waiting_taxis[taxi_id]
                wait_time = current_time - start_time
                
                # 如果等待时间超过阈值，通知调度器
                if self.scheduler and wait_time > 300:  # 5分钟阈值
                    self.scheduler.record_swap_wait_time(self.id, wait_time)

    def swap_battery(self, taxi):
        """
        为出租车更换电池的过程。
        
        参数:
            taxi: 需要换电的出租车对象
        
        返回:
            swap_success: 是否成功完成换电
            wait_time: 等待时间（分钟）
        """
        # 记录等待开始时间
        start_time = self.env.now
        self.current_waiting_taxis[taxi.id] = start_time
        
        # 根据acceptance_rate决定是否接受请求
        if random.random() > self.acceptance_rate:
            del self.current_waiting_taxis[taxi.id]
            return False, 0
        
        # 请求换电区
        with self.swap_bay.request() as bay_req:
            # 等待换电区可用
            yield bay_req
            
            # 计算等待时间
            wait_time = self.env.now - start_time
            
            # 检查是否还有充电电池可用
            if self.charged_batteries.level > 0:
                # 获取充电电池
                yield self.charged_batteries.get(1)
                
                # 执行换电操作
                yield self.env.timeout(self.swap_time)
                
                # 放入空电池
                yield self.empty_batteries.put(1)
                
                # 更新统计数据
                self.swap_count += 1
                self.total_swap_time += self.swap_time
                self.wait_times.append(wait_time)
                current_queue = len(self.swap_bay.queue)
                self.queue_lengths.append(current_queue)
                self.peak_queue_length = max(self.peak_queue_length, current_queue)
                
                # 报告等待时间给调度器
                if self.scheduler:
                    self.scheduler.record_swap_wait_time(self.id, wait_time)
                
                # 清除等待记录
                del self.current_waiting_taxis[taxi.id]
                
                return True, wait_time
            else:
                # 没有可用电池
                del self.current_waiting_taxis[taxi.id]
                return False, wait_time

    def get_status(self):
        """返回站点当前状态的快照"""
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
            'chargers': f"{self.charger_resource.count}/{self.charger_resource.capacity} 使用中",
            'total_charging_time': self.total_charging_time,
            'total_swap_time': self.total_swap_time,
            'peak_queue_length': self.peak_queue_length
        }
