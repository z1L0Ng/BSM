"""Block network representation and utilities."""
from dataprocess.distance import compute_distance_matrix, compute_travel_time_matrix, create_traffic_factors

class BlockNetwork:
    def __init__(self, block_positions, consider_traffic=True):
        """
        Initialize the block network with given block coordinates.
        
        Parameters:
            block_positions (dict): Block ID to (x,y) coordinate mapping
            consider_traffic (bool): Whether to consider traffic conditions
        """
        self.block_positions = block_positions
        
        # Precompute distance matrix between blocks
        self.distance_matrix = compute_distance_matrix(block_positions)
        
        # Traffic factors for different hours
        self.consider_traffic = consider_traffic
        if consider_traffic:
            self.traffic_factors = create_traffic_factors()
            # Initialize different travel time matrices for each hour
            self.travel_time_matrices = {}
            for hour in range(24):
                factor = self.traffic_factors.get(hour, 1.0)
                self.travel_time_matrices[hour] = compute_travel_time_matrix(
                    self.distance_matrix, 
                    avg_speed_km_per_min=0.5, 
                    traffic_factor=factor
                )
        else:
            # Single travel time matrix with constant speed
            self.travel_time_matrices = {
                0: compute_travel_time_matrix(self.distance_matrix, avg_speed_km_per_min=0.5)
            }
        
        # Default current hour
        self.current_hour = 0
    
    def update_time(self, hour):
        """
        Update the current hour to use the appropriate travel time matrix.
        
        Parameters:
            hour (int): Current hour (0-23)
        """
        self.current_hour = hour % 24
    
    def distance(self, block_a, block_b):
        """
        Get the distance between two blocks.
        
        Parameters:
            block_a (int): Origin block ID
            block_b (int): Destination block ID
            
        Returns:
            float: Distance between blocks or None if invalid
        """
        return self.distance_matrix.get(block_a, {}).get(block_b, None)
    
    def travel_time(self, block_a, block_b, hour=None):
        """
        Estimate travel time between blocks given current traffic conditions.
        
        Parameters:
            block_a (int): Origin block ID
            block_b (int): Destination block ID
            hour (int, optional): Hour to use for traffic conditions
                                 (defaults to current hour)
        
        Returns:
            float: Travel time in minutes or None if invalid
        """
        if hour is None:
            hour = self.current_hour
            
        if self.consider_traffic:
            # Use the travel time matrix for the current hour
            time_matrix = self.travel_time_matrices.get(hour % 24, self.travel_time_matrices[0])
            return time_matrix.get(block_a, {}).get(block_b, None)
        else:
            # Use the default travel time matrix
            time_matrix = self.travel_time_matrices[0]
            return time_matrix.get(block_a, {}).get(block_b, None)