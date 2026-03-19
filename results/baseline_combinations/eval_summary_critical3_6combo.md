# Evaluation Summary (Critical3 x 6 Combos)

- Source: `results/baseline_combinations/full_matrix_6combo_critical3_6_22.csv`
- Rows: 18 (expected 18)
- Duplicate key count (group_id+combo): 0

| Group | Combo | Served | Swap Success | Swap Miss | Charging Miss | Waiting | Peak Station Power |
|---|---|---:|---:|---:|---:|---:|---:|
| G1 | heuristic_plus_fcfs | 39273 | 1.000000 | 0.000000 | 0.111252 | 0 | 23810.0 |
| G1 | ideal_plus_fcfs | 66871 | 1.000000 | 0.000000 | 0.610538 | 0 | 28880.0 |
| G1 | heuristic_plus_gurobi | 38890 | 1.000000 | 0.000000 | 0.109863 | 0 | 23690.0 |
| G1 | ideal_plus_gurobi | 66871 | 1.000000 | 0.000000 | 0.318577 | 0 | 28850.0 |
| G1 | algorithm_plus_fcfs | 87652 | 0.144326 | 0.855674 | 0.071262 | 10150 | 26150.0 |
| G1 | algorithm_plus_gurobi | 87225 | 0.111310 | 0.888690 | 0.066667 | 12934 | 25400.0 |
| G2 | heuristic_plus_fcfs | 39232 | 0.836325 | 0.163675 | 0.191919 | 155 | 23780.0 |
| G2 | ideal_plus_fcfs | 66871 | 1.000000 | 0.000000 | 0.610538 | 0 | 28880.0 |
| G2 | heuristic_plus_gurobi | 38168 | 0.833682 | 0.166318 | 0.190715 | 159 | 23780.0 |
| G2 | ideal_plus_gurobi | 66871 | 1.000000 | 0.000000 | 0.318577 | 0 | 28850.0 |
| G2 | algorithm_plus_fcfs | 85477 | 0.135741 | 0.864259 | 0.197326 | 13810 | 27890.0 |
| G2 | algorithm_plus_gurobi | 87315 | 0.303548 | 0.696452 | 0.131202 | 4162 | 26450.0 |
| G3 | heuristic_plus_fcfs | 38943 | 0.839153 | 0.160847 | 0.025221 | 152 | 27740.0 |
| G3 | ideal_plus_fcfs | 66871 | 1.000000 | 0.000000 | 0.377784 | 0 | 43490.0 |
| G3 | heuristic_plus_gurobi | 39267 | 0.835774 | 0.164226 | 0.025031 | 157 | 27650.0 |
| G3 | ideal_plus_gurobi | 66871 | 1.000000 | 0.000000 | 0.150733 | 0 | 42140.0 |
| G3 | algorithm_plus_fcfs | 86599 | 0.428772 | 0.571228 | 0.016940 | 2438 | 34100.0 |
| G3 | algorithm_plus_gurobi | 86437 | 0.309971 | 0.690029 | 0.012222 | 4007 | 31400.0 |
