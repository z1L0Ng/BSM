"""Battery Swap Station (BSS) model."""
import simpy

class BatterySwapStation:
    def __init__(self, env, station_id, location, capacity=10, initial_batteries=None, swap_time=5, charge_time=60, n_chargers=1):
        """
        Battery Swap Station model.
        
        Parameters:
            env (simpy.Environment): Simulation environment.
            station_id (int): Identifier for the station.
            location (int): Block ID where the station is located.
            capacity (int): Total number of battery slots the station can hold.
            initial_batteries (int): Initial number of charged batteries available (defaults to full capacity).
            swap_time (float): Time (minutes) to perform a battery swap.
            charge_time (float): Time (minutes) to charge a depleted battery.
            n_chargers (int): Number of charging units (for concurrent charging).
        """
        self.env = env
        self.id = station_id
        self.location = location
        self.capacity = capacity
        # SimPy store for available charged batteries
        self.charged_batteries = simpy.Store(env)
        # SimPy store for empty batteries waiting to be charged
        self.empty_batteries = simpy.Store(env)
        # Initialize inventory of batteries
        if initial_batteries is None:
            initial_batteries = capacity
        # Fill charged batteries store
        for _ in range(initial_batteries):
            self.charged_batteries.put("battery")  # using a placeholder object
        # Remaining batteries start as empty (depleted)
        for _ in range(capacity - initial_batteries):
            self.empty_batteries.put("empty")
        # Resource for a swap bay (only one taxi can swap at a time per bay)
        self.swap_bay = simpy.Resource(env, capacity=1)
        # Resource for charging stations (limit concurrent charging operations)
        self.charger_resource = simpy.Resource(env, capacity=n_chargers)
        # Stats tracking
        self.swap_count = 0
        self.wait_times = []  # record waiting time for a charged battery if any
        # Start background process for charging batteries
        env.process(self._charging_process(charge_time))
        self.swap_time = swap_time
    
    def _charging_process(self, charge_time):
        """
        Background process that continuously charges depleted batteries.
        """
        while True:
            # Wait for an empty battery to charge
            empty_batt = yield self.empty_batteries.get()
            # Acquire a charger resource
            with self.charger_resource.request() as req:
                yield req
                # Charge the battery (takes charge_time)
                yield self.env.timeout(charge_time)
                # Once charged, move it to charged batteries store
                yield self.charged_batteries.put("battery")
            # Loop back to charge next battery
    
    def swap_battery(self, taxi):
        """
        Process of swapping a battery for an arriving taxi.
        """
        arrival_time = self.env.now
        # Request access to a swap bay
        with self.swap_bay.request() as req:
            yield req
            # Wait until a charged battery is available
            yield self.charged_batteries.get()
            wait_duration = self.env.now - arrival_time
            self.wait_times.append(wait_duration)
            # Put the taxi's depleted battery into the empty queue for charging
            yield self.empty_batteries.put("empty")
            # Simulate the battery swap duration
            yield self.env.timeout(self.swap_time)
            # Battery swap complete, taxi's battery is now full
            taxi.battery_level = taxi.battery_capacity
            # Update statistics
            self.swap_count += 1
        # (swap_bay is released automatically here)
        # TODO: Consider multiple swap bays or priority queue if needed.