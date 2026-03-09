# Baseline Combinations Summary (FCFS Station)

Source matrix: `results/baseline_combinations/full_matrix.csv`  
Grid size: 108 runs (3 combos x 4 inventories x 3 chargers x 3 swap capacities)

## Average over full grid

| Combo | Avg Served | Avg Swap Success | Avg Waiting Time (slots/request) | Avg Waiting Vehicles | Avg Peak Station Power (kW) | Avg Idle Moves |
|---|---:|---:|---:|---:|---:|---:|
| algorithm_plus_fcfs | 52741.67 | 0.997930 | 0.002070 | 1.667 | 230.0 | 53542.33 |
| heuristic_plus_fcfs | 52741.67 | 0.997930 | 0.002070 | 1.667 | 230.0 | 53542.33 |
| ideal_plus_fcfs | 52899.00 | 1.000000 | 0.000000 | 0.000 | 230.0 | 53700.00 |

## Hard setting (inventory=50, chargers=5, swap_capacity=6)

| Combo | Served | Swap Success | Total Waiting Vehicles | Avg Waiting Time | Peak Station Power (kW) |
|---|---:|---:|---:|---:|---:|
| algorithm_plus_fcfs | 52427 | 0.993789 | 5 | 0.006211 | 170.0 |
| heuristic_plus_fcfs | 52427 | 0.993789 | 5 | 0.006211 | 170.0 |
| ideal_plus_fcfs | 52899 | 1.000000 | 0 | 0.000000 | 170.0 |

## Notes

- In this matrix, `heuristic_plus_fcfs` matches `algorithm_plus_fcfs` on aggregate metrics.
- `ideal_plus_fcfs` removes swap bottlenecks by design and reaches the upper-bound service profile.
