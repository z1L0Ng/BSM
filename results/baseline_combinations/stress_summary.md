# Stress Baseline Combinations Summary (FCFS Station)

Source matrix: `results/baseline_combinations/stress_full_matrix.csv`
Grid size: 81 runs (3 combos x 3 inventories x 3 chargers x 3 swap capacities)

## Average over stress grid

| Combo | Avg Served | Avg Swap Success | Avg Swap Miss | Avg Charging Miss | Avg Waiting Vehicles | Avg Peak Station Power (kW) | Avg Idle Moves |
|---|---:|---:|---:|---:|---:|---:|---:|
| algorithm_plus_fcfs | 52353.78 | 0.700955 | 0.299045 | 0.998880 | 948.000 | 170.0 | 53154.78 |
| heuristic_plus_fcfs | 52443.11 | 0.915696 | 0.084304 | 0.071210 | 99.037 | 170.0 | 53243.74 |
| ideal_plus_fcfs | 52899.00 | 1.000000 | 0.000000 | 0.076987 | 0.000 | 180.0 | 53700.00 |

## Hard setting (inventory=5, chargers=3, swap_capacity=3)

| Combo | Served | Swap Success | Swap Miss | Charging Miss | Waiting Vehicles | Peak Station Power (kW) |
|---|---:|---:|---:|---:|---:|---:|
| algorithm_plus_fcfs | 53740 | 0.216460 | 0.783540 | 0.997396 | 2780 | 110.0 |
| heuristic_plus_fcfs | 53118 | 0.586461 | 0.413539 | 0.166876 | 562 | 110.0 |
| ideal_plus_fcfs | 52899 | 1.000000 | 0.000000 | 0.199750 | 0 | 110.0 |
