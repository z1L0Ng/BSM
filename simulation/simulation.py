"""Main simulation process using SimPy."""
import simpy
import random
from data import data_loader, distance_matrix
from environment.blocks import BlockNetwork
from agent.taxi import TaxiAgent
from bss.bss import BatterySwapStation

def setup_simulation(env, config):
    """
    Initialize simulation entities (network, stations, taxis).
    """
    # Create block network with distances
    block_positions = config.get('blocks', {})
    network = BlockNetwork(block_positions)
    # Initialize stations
    stations = []
    for i, station_conf in enumerate(config.get('stations', [])):
        station = BatterySwapStation(env, station_id=i,
                                     location=station_conf['block_id'],
                                     capacity=station_conf.get('capacity', 10),
                                     initial_batteries=station_conf.get('initial_batteries', None),
                                     swap_time=station_conf.get('swap_time', 5),
                                     charge_time=station_conf.get('charge_time', 60),
                                     n_chargers=station_conf.get('chargers', 1))
        stations.append(station)
    # Initialize taxis
    taxis = []
    taxi_conf = config.get('taxis', {})
    num_taxis = taxi_conf.get('count', 0)
    for i in range(num_taxis):
        # Determine initial position for taxi
        if 'initial_locations' in taxi_conf:
            init_block = taxi_conf['initial_locations'][i % len(taxi_conf['initial_locations'])]
        else:
            init_block = random.choice(list(block_positions.keys()))
        taxi = TaxiAgent(env, taxi_id=i,
                         initial_block=init_block,
                         battery_capacity=taxi_conf.get('battery_capacity', 100),
                         consumption_rate=taxi_conf.get('consumption_rate', 1),
                         swap_threshold=taxi_conf.get('swap_threshold', 20))
        taxis.append(taxi)
        env.process(taxi.process(network, stations))
    return network, taxis, stations

def run_simulation(config):
    """
    Run the simulation with the given configuration.
    
    Returns:
        dict: Simulation results and statistics.
    """
    env = simpy.Environment()
    network, taxis, stations = setup_simulation(env, config)
    # Run simulation for specified duration (minutes)
    duration = config.get('simulation', {}).get('duration', 0)
    if duration <= 0:
        print("Please specify a positive simulation duration in config.")
        return {}
    env.run(until=duration)
    # Gather results
    results = {
        'total_swaps': sum(s.swap_count for s in stations),
        'station_swap_counts': {s.id: s.swap_count for s in stations},
        'station_avg_wait': {s.id: (sum(s.wait_times)/len(s.wait_times) if s.wait_times else 0) for s in stations},
        'trips_completed': sum(t.trips_completed for t in taxis),
        'trips_per_taxi': {t.id: t.trips_completed for t in taxis}
    }
    return results

# If this module is executed directly, run a small demo simulation
if __name__ == "__main__":
    demo_config = {
        'blocks': {1: (0, 0), 2: (1, 0), 3: (0, 1), 4: (1, 1)},
        'stations': [
            {'block_id': 2, 'capacity': 5, 'initial_batteries': 5, 'swap_time': 5, 'charge_time': 30, 'chargers': 2},
            {'block_id': 4, 'capacity': 3, 'initial_batteries': 3, 'swap_time': 5, 'charge_time': 30, 'chargers': 1}
        ],
        'taxis': {
            'count': 3,
            'battery_capacity': 50,
            'consumption_rate': 1,
            'swap_threshold': 10
            # 'initial_locations': [1, 1, 3]  # optional starting blocks
        },
        'simulation': {
            'duration': 60  # minutes
        }
    }
    results = run_simulation(demo_config)
    print("Simulation completed.\nResults:")
    print(f"- Total trips completed: {results['trips_completed']}")
    print(f"- Total battery swaps: {results['total_swaps']}")
    for sid, count in results['station_swap_counts'].items():
        print(f"- Station {sid}: {count} swaps, avg wait {results['station_avg_wait'][sid]:.2f} min")