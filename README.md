# E-taxi Battery Swapping Simulation

This project implements an end-to-end simulation framework for an e-taxi fleet and battery swapping stations (BSS), following the paper `E_taxi_Battery_Swapping_Stations` and the technical specification.

## Key Concepts Implemented
- Spatial-temporal discretization of city regions and time slots.
- Fleet state modeling with vacant/occupied vehicles and discrete battery levels.
- Battery swapping station inventory and service constraints.
- Charging task scheduling with preemption/migration (EDF baseline).
- State transition dynamics using probabilistic matrices.

## Quick Start

```bash
python scripts/run_simulation.py --config configs/ev_yellow_2025_11.yaml
```

This will read NYC yellow taxi trip data and create a demand forecast by zone and time-bin. If you do not have the trip data, change `data.source` to `synthetic` in the config.
To simulate a single day, set `sim.sim_date` and keep `sim.horizon` equal to the number of slots in that day.

## Configuration
See `configs/ev_yellow_2025_11.yaml` for parameters:
- `sim.horizon`, `sim.time_bin_minutes`, `sim.sim_date`: simulation timeline and optional single-day filter.
- `sim.battery_levels`, `sim.charge_rate_levels_per_slot`: battery discretization and charge rate.
- `station.swapping_capacity`, `station.chargers`: station hardware capacity.
- `data.*`: dataset paths.

## Project Structure
- `etaxi_sim/` core simulation package
- `scripts/` runnable scripts
- `configs/` configuration files
- `docs/` source paper and technical specification

## Notes
- The current repositioning and charging policies are baselines meant to be replaced with optimized or learned policies.
- Distance and reachability matrices are synthetic in this version; you can replace them with real travel time/distance matrices.

## Next Steps
- Plug in a real distance/time matrix and compute `nu_{i,i'}` using travel-time constraints.
- Replace the greedy repositioning policy with an optimization or RL policy.
- Estimate transition probabilities from historical data.
