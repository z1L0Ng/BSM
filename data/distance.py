"""Distance matrix construction for city blocks."""
import numpy as np

def compute_distance_matrix(block_positions):
    """
    Compute Manhattan distance matrix between all blocks.
    
    Parameters:
        block_positions (dict): Mapping of block ID to (x, y) coordinates.
    Returns:
        dict: Nested dict of distances {block_i: {block_j: distance, ...}, ...}.
    """
    blocks = list(block_positions.keys())
    dist_matrix = {i: {} for i in blocks}
    for i in blocks:
        x1, y1 = block_positions[i]
        for j in blocks:
            x2, y2 = block_positions[j]
            # Manhattan distance as approximation of city travel distance
            dist = abs(x1 - x2) + abs(y1 - y2)
            dist_matrix[i][j] = dist
    # TODO: Use real travel time data or API for more accuracy in distance matrix.
    return dist_matrix