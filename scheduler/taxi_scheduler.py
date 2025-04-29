"""Centralised scheduler for the e-taxi fleet."""
import simpy
import random

class ETaxiScheduler:
    def __init__(self, env, taxis, stations, check_interval=5):
        """
        Parameters
        ----------
        env : simpy.Environment
        taxis : list[TaxiAgent]
        stations : list[BatterySwapStation]
        check_interval : int
            Minutes between successive scheduling decisions.
        """
        self.env = env
        self.taxis = taxis
        self.stations = stations
        self.interval = check_interval

        # launch process
        env.process(self._run())

    # ---------------- scheduler loop ---------------- #
    def _run(self):
        while True:
            yield self.env.timeout(self.interval)
            self.dispatch_swaps()

    # ------------- core decision heuristic ----------- #
    def dispatch_swaps(self):
        """Simple heuristic:
        1. 找出需要换电的出租车 (battery <= threshold & 未收到指令)
        2. 为每辆车挑选“可用充电电池最多且距离最近”的站
        3. 下达 swap_order
        """
        for taxi in self.taxis:
            if taxi.needs_swap() and taxi.swap_order is None:
                # 过滤掉没有库存的站
                candidate = [
                    s for s in self.stations
                    if s.charged_batteries.level > 0   # SimPy Store 的 level
                ]
                if not candidate:
                    continue  # 暂时无可用电池
                # 选择最近且库存最多的站
                best = min(
                    candidate,
                    key=lambda s: (s.distance_to(taxi.location), -s.charged_batteries.level)
                )
                taxi.assign_swap(best)