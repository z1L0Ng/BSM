"""Scheduler that manages internal chargers of a BatterySwapStation."""
class StationScheduler:
    def __init__(self, station, check_interval=10,
                 min_charged=2, max_charged=None):
        """
        Parameters
        ----------
        station : BatterySwapStation
        check_interval : int
            Minutes between charger-allocation decisions.
        min_charged : int
            Keep at least this many charged batteries ready.
        max_charged : int | None
            Upper bound of charged batteries (pause charging if exceeded).
        """
        self.station = station
        self.env = station.env
        self.interval = check_interval
        self.min_charged = min_charged
        self.max_charged = max_charged or station.capacity

        self.env.process(self._run())

    def _run(self):
        while True:
            yield self.env.timeout(self.interval)
            self._schedule_chargers()

    def _schedule_chargers(self):
        """Start/stop charging tasks to keep stock between [min_charged, max_charged]."""
        charged = self.station.charged_batteries.level
        empty   = self.station.empty_batteries.level
        idle_chargers = self.station.charger_resource.capacity - \
                        self.station.charger_resource.count

        # If charged stock < min_charged, start charging as many as possible
        if charged < self.min_charged and empty > 0 and idle_chargers > 0:
            n_start = min(self.min_charged - charged, empty, idle_chargers)
            for _ in range(n_start):
                self.env.process(self.station._charge_one_battery())

        # If charged stock > max_charged, allow current charges to finish but don't start new ones