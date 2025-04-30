"""管理BatterySwapStation内部充电器的调度器。"""
import simpy
import random

class StationScheduler:
    def __init__(self, station, check_interval=10, min_charged=2, max_charged=None,
                 dynamic_charging=True, peak_hours=None, off_peak_discount=0.7):
        """
        参数:
        ----------
        station : BatterySwapStation
            要管理的电池交换站
        check_interval : int
            充电器分配决策之间的间隔分钟数
        min_charged : int
            保持至少这么多已充电电池随时可用
        max_charged : int | None
            已充电电池的上限(如果超过则暂停充电)
        dynamic_charging : bool
            是否启用基于时间的动态充电策略
        peak_hours : list
            高峰时段列表，格式为[(start_hour, end_hour), ...]
        off_peak_discount : float
            非高峰时段的电价折扣系数
        """
        self.station = station
        self.env = station.env
        self.interval = check_interval
        self.min_charged = min_charged
        self.max_charged = max_charged or station.capacity
        
        # 动态充电策略设置
        self.dynamic_charging = dynamic_charging
        self.peak_hours = peak_hours or [(7, 10), (17, 20)]  # 默认早晚高峰
        self.off_peak_discount = off_peak_discount
        
        # 电价相关
        self.base_price_per_kwh = 0.2  # 基础电价，美元/kWh
        self.current_price = self.base_price_per_kwh  # 当前电价
        
        # 充电记录
        self.charging_cost = 0  # 总充电成本
        self.energy_charged = 0  # 充电的总能量(kWh)
        
        # 启动调度进程
        self.env.process(self._run())
        
        # 如果启用动态充电，添加价格更新进程
        if self.dynamic_charging:
            self.env.process(self._update_pricing())
    
    def _run(self):
        """主调度循环，定期检查并调整充电策略"""
        while True:
            yield self.env.timeout(self.interval)
            
            # 根据当前状态调度充电器
            self._schedule_chargers()
            
            # 记录当前状态
            self._log_status()
    
    def _update_pricing(self, update_interval=60):
        """
        根据一天中的时间更新电价。
        高峰时段电价较高，非高峰时段有折扣。
        """
        while True:
            # 等待指定的时间间隔
            yield self.env.timeout(update_interval)
            
            # 获取当前小时(模拟时间假设从0开始，以分钟为单位)
            current_hour = int(self.env.now / 60) % 24
            
            # 检查当前是否为高峰时段
            is_peak = any(start <= current_hour < end for start, end in self.peak_hours)
            
            if is_peak:
                # 高峰时段使用基础价格
                self.current_price = self.base_price_per_kwh
            else:
                # 非高峰时段应用折扣
                self.current_price = self.base_price_per_kwh * self.off_peak_discount
    
    def _schedule_chargers(self):
        """
        调度充电任务，保持库存在[min_charged, max_charged]之间，
        并考虑当前时段的电价因素。
        """
        charged = self.station.charged_batteries.level
        empty = self.station.empty_batteries.level
        idle_chargers = self.station.charger_resource.capacity - self.station.charger_resource.count
        
        # 获取当前小时
        current_hour = int(self.env.now / 60) % 24
        
        # 检查当前是否为高峰时段
        is_peak = any(start <= current_hour < end for start, end in self.peak_hours)
        
        # 根据时段调整最小充电阈值
        adjusted_min = self.min_charged
        if self.dynamic_charging:
            if is_peak:
                # 高峰时段减少充电(除非库存不足)
                adjusted_min = max(1, self.min_charged // 2)
            else:
                # 非高峰时段增加充电
                adjusted_min = min(self.max_charged, self.min_charged * 2)
        
        # 如果已充电电池 < 最小阈值，尽可能多地启动充电
        if charged < adjusted_min and empty > 0 and idle_chargers > 0:
            n_start = min(adjusted_min - charged, empty, idle_chargers)
            
            # 启动n_start个充电进程
            for _ in range(n_start):
                # 计算充电成本(假设每块电池是20kWh)
                battery_capacity = 20  # kWh
                charge_cost = battery_capacity * self.current_price
                
                # 累计成本和能量
                self.charging_cost += charge_cost
                self.energy_charged += battery_capacity
                
                # 记录充电决策
                print(f"时间 {self.env.now:.1f}: 站点 {self.station.id} 启动充电, 当前电价: ${self.current_price:.3f}/kWh")
        
        # 如果已充电电池 > 最大阈值，允许当前充电完成但不启动新的充电
        
    def _log_status(self):
        """记录当前站点状态"""
        charged = self.station.charged_batteries.level
        empty = self.station.empty_batteries.level
        current_hour = int(self.env.now / 60) % 24
        is_peak = any(start <= current_hour < end for start, end in self.peak_hours)
        
        print(f"时间 {self.env.now:.1f}: 站点 {self.station.id} 状态 - "
              f"已充电: {charged}/{self.station.capacity}, "
              f"待充电: {empty}/{self.station.capacity}, "
              f"时段: {'高峰' if is_peak else '非高峰'}, "
              f"电价: ${self.current_price:.3f}/kWh")
    
    def get_status(self):
        """
        获取调度器状态摘要。
        
        返回:
            dict: 包含调度器状态的字典
        """
        current_hour = int(self.env.now / 60) % 24
        is_peak = any(start <= current_hour < end for start, end in self.peak_hours)
        
        avg_price = self.charging_cost / self.energy_charged if self.energy_charged > 0 else 0
        
        return {
            "station_id": self.station.id,
            "charged_batteries": self.station.charged_batteries.level,
            "empty_batteries": self.station.empty_batteries.level,
            "min_charged_target": self.min_charged,
            "current_hour": current_hour,
            "is_peak_hour": is_peak,
            "current_price": f"${self.current_price:.3f}/kWh",
            "total_charging_cost": f"${self.charging_cost:.2f}",
            "average_price": f"${avg_price:.3f}/kWh",
            "energy_charged": f"{self.energy_charged:.1f} kWh"
        }