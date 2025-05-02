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
        demand_points (dict): Demand counts per block {block_id: demand_count}
        candidate_locations (dict): Possible station coordinates {block_id: (x, y)}
        num_stations (int): Number of stations to choose
        max_distance (float, optional): Maximum coverage distance
    
    Returns:
        list: Selected block IDs for stations
    """
    if gp is None:
        print("Gurobi not available. Using greedy optimization.")
        return greedy_optimization(demand_points, candidate_locations, num_stations, max_distance)

    try:
        demand_ids = list(demand_points.keys())
        cand_ids = list(candidate_locations.keys())
        # Use candidate_locations to get coordinates for demands
        demand_coords = np.array([candidate_locations[bid] for bid in demand_ids])
        candidate_coords = np.array([candidate_locations[bid] for bid in cand_ids])
        # Calculate distance matrix
        distances = cdist(demand_coords, candidate_coords, 'euclidean')

        model = gp.Model("BSS_Layout_Optimization")
        x = model.addVars(len(cand_ids), vtype=gp.GRB.BINARY, name="x")

        if max_distance is not None:
            y = model.addVars(len(demand_ids), vtype=gp.GRB.BINARY, name="y")
            for i in range(len(demand_ids)):
                model.addConstr(
                    gp.quicksum(x[j] for j in range(len(cand_ids))
                               if distances[i, j] <= max_distance) >= y[i]
                )
            model.setObjective(gp.quicksum(y[i] for i in range(len(demand_ids))), gp.GRB.MAXIMIZE)
        else:
            z = model.addVars(len(demand_ids), len(cand_ids), vtype=gp.GRB.BINARY, name="z")
            for i in range(len(demand_ids)):
                model.addConstr(gp.quicksum(z[i, j] for j in range(len(cand_ids))) == 1)
                for j in range(len(cand_ids)):
                    model.addConstr(z[i, j] <= x[j])
            model.setObjective(
                gp.quicksum(distances[i, j] * z[i, j]
                           for i in range(len(demand_ids))
                           for j in range(len(cand_ids))), gp.GRB.MINIMIZE
            )
        model.addConstr(gp.quicksum(x[j] for j in range(len(cand_ids))) == num_stations)
        model.setParam('OutputFlag', 1)
        model.setParam('TimeLimit', 300)
        model.optimize()

        if model.status in (gp.GRB.Status.OPTIMAL, gp.GRB.Status.TIME_LIMIT):
            selected = [cand_ids[j] for j in range(len(cand_ids)) if x[j].X > 0.5]
            if max_distance is not None:
                covered = sum(int(y[i].X) for i in range(len(demand_ids)))
                print(f"Coverage: {covered}/{len(demand_ids)} ({covered/len(demand_ids)*100:.1f}%)")
            else:
                total_dist = sum(distances[i, j] * z[i, j].X
                                 for i in range(len(demand_ids))
                                 for j in range(len(cand_ids)))
                print(f"Total distance: {total_dist:.1f}")
            return selected
        else:
            print(f"Optimization failed (status {model.status}). Using greedy optimization.")
            return greedy_optimization(demand_points, candidate_locations, num_stations, max_distance)
    except Exception as e:
        print(f"Optimization error: {e}. Using greedy optimization.")
        return greedy_optimization(demand_points, candidate_locations, num_stations, max_distance)


def greedy_optimization(demand_points, candidate_locations, num_stations, max_distance=None):
    """
    Greedy algorithm for BSS placement when Gurobi is unavailable.
    """
    demand_ids = list(demand_points.keys())
    cand_ids = list(candidate_locations.keys())
    demand_coords = np.array([candidate_locations[bid] for bid in demand_ids])
    candidate_coords = np.array([candidate_locations[bid] for bid in cand_ids])
    distances = cdist(demand_coords, candidate_coords, 'euclidean')

    weights = np.array([demand_points[bid] for bid in demand_ids])
    selected = []
    for _ in range(min(num_stations, len(cand_ids))):
        scores = []
        for j in range(len(cand_ids)):
            if cand_ids[j] in selected:
                scores.append(-1)
            else:
                cover = distances[:, j] <= (max_distance if max_distance is not None else np.inf)
                scores.append(np.dot(cover.astype(int), weights))
        best_idx = int(np.argmax(scores))
        selected.append(cand_ids[best_idx])
        if max_distance is not None:
            covered_idx = np.where(distances[:, best_idx] <= max_distance)[0]
            weights[covered_idx] = 0
    print(f"Greedy selected {len(selected)} stations; avg coverage: {np.mean(weights==0):.2f}")
    return selected
