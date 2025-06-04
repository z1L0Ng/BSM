import pandas as pd
import numpy as np
import os

# Create sample data
np.random.seed(42)
n_rows = 1000

# Generate random timestamps for one day
base_date = pd.Timestamp('2025-01-01')
pickup_times = [base_date + pd.Timedelta(minutes=int(x)) for x in np.random.uniform(0, 1440, n_rows)]

# Generate sample data
data = {
    'tpep_pickup_datetime': pickup_times,
    'pickup_latitude': np.random.uniform(40.7, 40.9, n_rows),
    'pickup_longitude': np.random.uniform(-74.0, -73.9, n_rows),
    'dropoff_latitude': np.random.uniform(40.7, 40.9, n_rows),
    'dropoff_longitude': np.random.uniform(-74.0, -73.9, n_rows),
    'passenger_count': np.random.randint(1, 5, n_rows),
    'trip_distance': np.random.uniform(1, 10, n_rows),
    'total_amount': np.random.uniform(10, 50, n_rows),
    'taxi_id': np.random.randint(1, 101, n_rows)
}

# Create DataFrame
df = pd.DataFrame(data)

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

# Save as parquet
df.to_parquet('data/yellow_tripdata_2025-01.parquet', index=False)
print('Sample data file created successfully!')
