# Evaluation Summary (Critical3 x 6 Combos)

- Source: `results/baseline_combinations/full_matrix_6combo_critical3_6_22_preflight.csv`
- Rows: 6 (expected 18)
- Duplicate key count (group_id+combo): 0

| Group | Combo | Served | Swap Success | Swap Miss | Charging Miss | Waiting | Peak Station Power |
|---|---|---:|---:|---:|---:|---:|---:|
| G2 | heuristic_plus_fcfs | 415 | 0.916667 | 0.083333 | 0.000000 | 64 | 23780.0 |
| G2 | ideal_plus_fcfs | 423 | 1.000000 | 0.000000 | 0.000000 | 0 | 23780.0 |
| G2 | heuristic_plus_gurobi | 415 | 0.916667 | 0.083333 | 0.000000 | 64 | 22070.0 |
| G2 | ideal_plus_gurobi | 423 | 1.000000 | 0.000000 | 0.000000 | 0 | 21440.0 |
| G2 | algorithm_plus_fcfs | 423 | 0.910737 | 0.089263 | 0.000000 | 69 | 23780.0 |
| G2 | algorithm_plus_gurobi | 423 | 0.910737 | 0.089263 | 0.000000 | 69 | 22070.0 |
