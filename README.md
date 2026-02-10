# BSM Taxi Simulation (Framework)

A minimal, reproducible framework to prototype NYC TLC Yellow taxi demand-driven simulation using November 2025 data. The model is intentionally lightweight so you can iterate on assumptions, objective functions, and constraints as your paper design evolves.

## Quick Start

1) Create the conda environment.

```bash
conda env create -f environment.yml
conda activate bsm
```

2) Download the November 2025 Yellow dataset.

```bash
python scripts/download_data.py --config configs/yellow_2025_11.yaml
```

3) Process and clean the dataset.

```bash
python scripts/process_data.py --config configs/yellow_2025_11.yaml
```

4) Run a baseline simulation.

```bash
python scripts/run_simulation.py --config configs/yellow_2025_11.yaml
```

## EV Fleet + Swap Station Experiment

This pipeline uses Yellow trip records as demand, maps trips to EV energy consumption, and compares a baseline vs. coordinated optimization.

```bash
python scripts/download_data.py --config configs/ev_yellow_2025_11.yaml
python scripts/process_ev_data.py --config configs/ev_yellow_2025_11.yaml
python scripts/run_ev_experiment.py --config configs/ev_yellow_2025_11.yaml
```

Outputs:
- `reports/fig1_demand_supply.png`
- `reports/fig2_power_profile.png`
- `reports/fig3_battery_inventory.png`
- `reports/ev_metrics.csv`

## Project Layout

- `configs/`: experiment and data configuration
- `data/raw/`: downloaded parquet files
- `data/processed/`: cleaned parquet files with derived fields
- `data/interim/`: optional intermediate statistics
- `reports/`: figures and metrics outputs
- `scripts/`: CLI entry points
- `src/bsm/`: core library (processing, simulation, metrics)

## Data Source

We use the NYC TLC Trip Record data, November 2025 Yellow Taxi (parquet). The TLC page also provides the Trip Record User Guide and the data dictionary for field definitions.

- TLC Trip Record Data page: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- November 2025 Yellow parquet: https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-11.parquet
- Trip Record User Guide: https://www.nyc.gov/assets/tlc/downloads/pdf/trip_record_user_guide.pdf
- Yellow Taxi data dictionary: https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf

## Configuration

Key sections in `configs/yellow_2025_11.yaml`:

- `dataset`: download URL, filename, and columns used
- `processing`: filters for duration/speed and output path
- `simulation`: time step, max wait, number of vehicles, and policy
- `metrics`: list of metric outputs

## Output Metrics (Baseline)

By default the simulation outputs:

- average wait time
- 95th percentile wait time
- unserved rate (requests not matched within max wait)
- vehicle utilization
- average trip duration
- throughput (served requests per hour)

These are placeholders and can be replaced/extended to match your paper's objectives.

## Notes

- The simulator currently uses a simple matching policy and uses observed trip durations from the data. This is meant as a scaffold for your formal model.
- When your paper model is ready, implement it under `src/bsm/simulation/policies.py` and update the config.

## Reproducibility

All steps are driven by config files and a single conda environment definition. To add additional months or datasets, copy `configs/yellow_2025_11.yaml` and update the dataset URL and file name.
