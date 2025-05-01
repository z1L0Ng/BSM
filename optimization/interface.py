"""Optimization module interface (using Gurobi optimizer)."""
import numpy as np
from scipy.spatial.distance import cdist
try:
    import gurobipy as gp
except ImportError:
    gp = None

def optimize_bss_layout(demand_points, candidate_locations, num_stations, max_distance=None):
    """
    Optimize the placement of battery swap stations.
    
    Parameters:
        demand_points (dict): Locations of demand points {point_id: (x, y)}
        candidate_locations (dict): Possible locations for stations {loc_id: (x, y)}
        num_stations (int): Number of stations to choose
        max_distance (float, optional): Maximum service distance (for coverage model)
    
    Returns:
        list: Selected locations for the stations
    """
    if gp is None:
        print("Gurobi not available. Using greedy optimization.")
        return greedy_optimization(demand_points, candidate_locations, num_stations)
    
    try:
        # Convert dictionaries to lists for processing
        demand_ids = list(demand_points.keys())
        candidate_ids = list(candidate_locations.keys())
        
        # Calculate distance matrix
        demand_coords = np.array([demand_points[i] for i in demand_ids])
        candidate_coords = np.array([candidate_locations[i] for i in candidate_ids])
        
        distances = cdist(demand_coords, candidate_coords, 'euclidean')
        
        # Create optimization model
        model = gp.Model("BSS_Layout_Optimization")
        
        # Decision variables: whether to place a station at location j
        x = model.addVars(len(candidate_ids), vtype=gp.GRB.BINARY, name="x")
        
        if max_distance is not None:
            # Maximum coverage model
            # Decision variable: whether demand point i is covered
            y = model.addVars(len(demand_ids), vtype=gp.GRB.BINARY, name="y")
            
            # Coverage constraint: demand point i is covered if it's within max_distance of any station
            for i in range(len(demand_ids)):
                model.addConstr(
                    gp.quicksum(x[j] for j in range(len(candidate_ids)) 
                               if distances[i,j] <= max_distance) >= y[i]
                )
            
            # Objective: maximize number of covered demand points
            model.setObjective(gp.quicksum(y), gp.GRB.MAXIMIZE)
        else:
            # p-median model
            # Decision variable: whether demand point i is assigned to station j
            z = model.addVars(len(demand_ids), len(candidate_ids), vtype=gp.GRB.BINARY, name="z")
            
            # Each demand point must be assigned to exactly one station
            for i in range(len(demand_ids)):
                model.addConstr(gp.quicksum(z[i,j] for j in range(len(candidate_ids))) == 1)
            
            # Demand points can only be assigned to built stations
            for i in range(len(demand_ids)):
                for j in range(len(candidate_ids)):
                    model.addConstr(z[i,j] <= x[j])
            
            # Objective: minimize total distance
            model.setObjective(
                gp.quicksum(distances[i,j] * z[i,j] for i in range(len(demand_ids)) 
                           for j in range(len(candidate_ids))),
                gp.GRB.MINIMIZE
            )
        
        # Constraint: exactly num_stations stations
        model.addConstr(gp.quicksum(x) == num_stations)
        
        # Set Gurobi parameters
        model.setParam('OutputFlag', 1)  # Show output
        model.setParam('TimeLimit', 300)  # 5-minute time limit
        
        # Optimize model
        model.optimize()
        
        # Check optimization status
        if model.status == gp.GRB.Status.OPTIMAL or model.status == gp.GRB.Status.TIME_LIMIT:
            # Get selected stations
            selected_locations = [candidate_ids[j] for j in range(len(candidate_ids)) 
                                 if x[j].x > 0.5]
            
            # Print coverage or total distance
            if max_distance is not None:
                coverage = sum(y[i].x for i in range(len(demand_ids)))
                coverage_ratio = coverage / len(demand_ids)
                print(f"Coverage: {coverage}/{len(demand_ids)} ({coverage_ratio*100:.1f}%)")
            else:
                total_dist = sum(distances[i,j] * z[i,j].x 
                                for i in range(len(demand_ids)) 
                                for j in range(len(candidate_ids)))
                avg_dist = total_dist / len(demand_ids)
                print(f"Total distance: {total_dist:.1f}, Average distance: {avg_dist:.1f}")
            
            return selected_locations
        else:
            print(f"Optimization failed with status {model.status}. Using greedy method.")
            return greedy_optimization(demand_points, candidate_locations, num_stations)
    
    except Exception as e:
        print(f"Optimization error: {e}")
        print("Using greedy optimization.")
        return greedy_optimization(demand_points, candidate_locations, num_stations)

def greedy_optimization(demand_points, candidate_locations, num_stations):
    """
    Greedy algorithm for BSS placement when Gurobi is not available.
    
    Parameters:
        demand_points (dict): Locations of demand points {point_id: (x, y)}
        candidate_locations (dict): Possible locations for stations {loc_id: (x, y)}
        num_stations (int): Number of stations to choose
    
    Returns:
        list: Selected locations for the stations
    """
    # Convert dictionaries to lists
    demand_ids = list(demand_points.keys())
    candidate_ids = list(candidate_locations.keys())
    
    # Calculate distance matrix
    demand_coords = np.array([demand_points[i] for i in demand_ids])
    candidate_coords = np.array([candidate_locations[i] for i in candidate_ids])
    
    distances = cdist(demand_coords, candidate_coords, 'euclidean')
    
    # Greedy station selection
    selected_indices = []
    remaining_indices = list(range(len(candidate_ids)))
    
    # Initialize minimum distances to infinity
    min_distances = np.full(len(demand_ids), np.inf)
    
    for _ in range(min(num_stations, len(candidate_ids))):
        max_improvement = -np.inf
        best_idx = -1
        
        for idx in remaining_indices:
            # Calculate new minimum distances if this station is added
            new_min_distances = np.minimum(min_distances, distances[:, idx])
            
            # Calculate improvement in total distance
            improvement = np.sum(min_distances - new_min_distances)
            
            if improvement > max_improvement:
                max_improvement = improvement
                best_idx = idx
        
        if best_idx != -1:
            # Add best station
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
            
            # Update minimum distances
            min_distances = np.minimum(min_distances, distances[:, best_idx])
    
    # Convert back to original IDs
    selected_locations = [candidate_ids[idx] for idx in selected_indices]
    
    # Print statistics
    print(f"Greedy algorithm selected {len(selected_locations)} stations")
    print(f"Average minimum distance: {np.mean(min_distances):.2f}")
    
    return selected_locations