"""Electric Taxi Agent model."""
import simpy
import random

class TaxiAgent:
    def __init__(self, env, taxi_id, initial_block, battery_capacity=100, consumption_rate=1, swap_threshold=20):
        """
        Taxi agent with a simple state machine for operations.
        
        Parameters:
            env (simpy.Environment): Simulation environment.
            taxi_id (int): Identifier for the taxi.
            initial_block (int): Starting block location ID.
            battery_capacity (float): Maximum battery capacity (energy units).
            consumption_rate (float): Battery consumption rate per minute of driving.
            swap_threshold (float): Battery level threshold to trigger a battery swap.
        """
        self.env = env
        self.id = taxi_id
        self.location = initial_block
        self.battery_capacity = battery_capacity
        self.battery_level = battery_capacity  # start with full battery
        self.consumption_rate = consumption_rate
        self.swap_threshold = swap_threshold
        self.trips_completed = 0
        self.swap_count = 0
    
    def drive_to(self, destination_block, network):
        """
        Drive to a destination block, consuming time and battery.
        """
        if self.location == destination_block:
            return  # no movement needed
        travel_time = network.travel_time(self.location, destination_block)
        if travel_time is None:
            return
        # Consume battery for travel
        consumption = travel_time * self.consumption_rate
        self.battery_level = max(0, self.battery_level - consumption)
        # Simulate travel time passing
        yield self.env.timeout(travel_time)
        # Update location upon arrival
        self.location = destination_block
    
    def perform_trip(self, network):
        """
        Simulate performing a passenger trip to a random destination.
        """
        dest = self.location
        # Choose a different destination than current
        while dest == self.location:
            dest = random.choice(list(network.block_positions.keys()))
        # Travel to the destination
        yield from self.drive_to(dest, network)
        # One trip completed
        self.trips_completed += 1
    
    def process(self, network, stations):
        """
        The main generator process of the taxi agent for SimPy.
        """
        while True:
            if self.battery_level > self.swap_threshold:
                # Continue serving trips if battery is sufficient
                yield from self.perform_trip(network)
            else:
                # Battery low: go to the nearest swap station
                nearest_station = min(stations, key=lambda s: network.distance(self.location, s.location))
                # Travel to nearest station
                yield from self.drive_to(nearest_station.location, network)
                # Perform battery swap (this will recharge the battery to full)
                yield from nearest_station.swap_battery(self)
                # After swapping, battery is full and taxi can resume trips
            # TODO: Integrate real trip requests and dispatch logic instead of random trips.
            # Optionally include idle time here to simulate waiting for next request.