"""Data loading and cleaning module for NYC TLC trip records."""
import pandas as pd
import os

def load_trip_data(filepath, use_dask=False, chunksize=None):
    """
    Load NYC TLC trip records from a Parquet file.
    
    Parameters:
        filepath (str): Path to the Parquet file containing trip data.
        use_dask (bool): If True, use Dask for loading large data.
        chunksize (int): Number of rows to load at a time. Only used when use_dask=False.
    
    Returns:
        DataFrame: Pandas DataFrame (or Dask DataFrame if use_dask) with trip records.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file does not exist: {filepath}")

    if use_dask:
        try:
            import dask.dataframe as dd
            # Use Dask for processing large datasets
            df = dd.read_parquet(filepath)
            print(f"Successfully loaded data using Dask, number of partitions: {df.npartitions}")
            return df
        except ImportError:
            print("Dask is not installed, will proceed with pandas.")
            use_dask = False
    
    if chunksize:
        # Use chunked reading for large files
        return pd.read_parquet(filepath, engine='pyarrow', chunksize=chunksize)
    else:
        # Read the entire file at once
        df = pd.read_parquet(filepath, engine='pyarrow')
        print(f"Successfully loaded data, total rows: {len(df)}")
        return df

def clean_trip_data(df):
    """
    Perform basic cleaning on trip data DataFrame.
    
    - Filters out trips with missing or invalid data.
    - Prepares fields for simulation (e.g., extracting pickup/dropoff coordinates or zones).
    
    Returns:
        DataFrame: Cleaned DataFrame ready for simulation.
    """
    # Example cleaning steps (to be replaced with actual logic):
    if hasattr(df, 'compute'):  # Dask DataFrame case
        df = df[df.passenger_count > 0].compute()
    else:
        df = df[df.passenger_count > 0]
    # TODO: Add more cleaning steps (e.g., remove outliers, handle missing values, time conversion).
    return df
