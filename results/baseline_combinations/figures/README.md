# Stress Figures

Source CSV: `results/baseline_combinations/stress_full_matrix.csv`

## Generated Figures

- `01_combo_avg_kpis.png`
- `02_served_vs_swap_capacity.png`
- `03_swap_miss_vs_swap_capacity.png`
- `04_charging_miss_vs_chargers.png`
- `05_waiting_vs_swap_capacity.png`
- `06_peak_power_vs_chargers.png`

## Hard Setting Slice (inventory=5, chargers=3, swap_capacity=3)

| Combo | Served | Swap Success | Swap Miss | Charging Miss | Waiting Vehicles | Peak Station Power |
|---|---:|---:|---:|---:|---:|---:|
| algorithm_plus_fcfs | 53740 | 0.216460 | 0.783540 | 0.997396 | 2780 | 110.0 |
| heuristic_plus_fcfs | 53118 | 0.586461 | 0.413539 | 0.166876 | 562 | 110.0 |
| ideal_plus_fcfs | 52899 | 1.000000 | 0.000000 | 0.199750 | 0 | 110.0 |
