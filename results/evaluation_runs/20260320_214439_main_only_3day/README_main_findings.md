# Main Findings (Main-Only 3-Day Evaluation)

- Total scenarios completed: 27 / 27
- Date coverage: 2025-11-01, 2025-11-05, 2025-11-09
- Group coverage: G1, G2, G3
- algorithm_plus_gurobi: total_served_mean=77613.67, swap_success_mean=0.1531, deadline_miss_mean=0.8469
- heuristic_plus_fcfs: total_served_mean=36284.33, swap_success_mean=0.8969, deadline_miss_mean=0.1031
- ideal_plus_fcfs: total_served_mean=67840.44, swap_success_mean=1.0000, deadline_miss_mean=0.0000
- algorithm_plus_gurobi vs heuristic_plus_fcfs: mean_delta_total_served=41329.33, p_sign=0.0039 (n=9)
- algorithm_plus_gurobi vs ideal_plus_fcfs: mean_delta_total_served=9773.22, p_sign=0.0039 (n=9)
