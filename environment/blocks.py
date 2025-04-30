"""Block network representation and utilities."""
from data.distance import compute_distance_matrix

class BlockNetwork:
    def __init__(self, block_positions):
        """
        Initialize the block network with given block coordinates.
        """
        self.block_positions = block_positions
        # Precompute distance (or travel time) matrix between blocks
        self.distance_matrix = compute_distance_matrix(block_positions)
    
    def distance(self, block_a, block_b):
        """
        Get the distance between two blocks.
        """
        return self.distance_matrix.get(block_a, {}).get(block_b, None)
    
    def travel_time(self, block_a, block_b, speed_km_per_min=1.0):
        """
        Estimate travel time between blocks given an average speed (km/min).
        """
        dist = self.distance(block_a, block_b)
        if dist is None:
            return None
        return dist / speed_km_per_min