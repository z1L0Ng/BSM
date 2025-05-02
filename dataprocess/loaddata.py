"""Data loading and processing module for NYC TLC trip records."""
import pandas as pd
import os
import numpy as np
from datetime import datetime

def load_trip_data(filepath, use_dask=False, chunksize=None):
    """
    Load NYC TLC taxi trip data from Parquet or CSV file.
    
    Parameters:
        filepath (str): Path to the file containing trip data
        use_dask (bool): If True, use Dask for loading large data
        chunksize (int): Number of rows to load at once. Only used if use_dask=False
    
    Returns:
        DataFrame: Pandas DataFrame with trip records (or Dask DataFrame if use_dask=True)
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file does not exist: {filepath}")

    # Determine file type by extension
    file_ext = os.path.splitext(filepath)[1].lower()

    if use_dask:
        try:
            import dask.dataframe as dd
            # Use Dask for large datasets
            if file_ext == '.parquet':
                df = dd.read_parquet(filepath)
            elif file_ext in ['.csv', '.txt']:
                df = dd.read_csv(filepath)
            else:
                raise ValueError(f"Unsupported file extension: {file_ext}")
            
            print(f"Successfully loaded data with Dask, partitions: {df.npartitions}")
            return df
        except ImportError:
            print("Dask not installed, falling back to pandas.")
            use_dask = False
    
    if file_ext == '.parquet':
        if chunksize:
            # Use chunked reading for large files
            return pd.read_parquet(filepath, engine='pyarrow', chunksize=chunksize)
        else:
            # Load entire file at once
            df = pd.read_parquet(filepath, engine='pyarrow')
    elif file_ext in ['.csv', '.txt']:
        if chunksize:
            return pd.read_csv(filepath, chunksize=chunksize)
        else:
            df = pd.read_csv(filepath)
    else:
        raise ValueError(f"Unsupported file extension: {file_ext}")
    
    print(f"Successfully loaded data, total rows: {len(df)}")
    return df

def clean_trip_data(df, year=2025, month=1):
    """
    Perform basic cleaning on trip data DataFrame.
    
    - Filter out trips with missing or invalid data
    - Prepare fields for simulation (extract pickup/dropoff coordinates or areas)
    - Process timestamp and geographic data
    
    Parameters:
        df (DataFrame): Raw trip data DataFrame
        year (int): Year of the data for handling different formats
        month (int): Month of the data for handling different formats
    
    Returns:
        DataFrame: Cleaned and simulation-ready DataFrame
    """
    # Check if Dask DataFrame
    is_dask = hasattr(df, 'compute')
    
    # Identify format based on columns
    if 'tpep_pickup_datetime' in df.columns:
        # Yellow taxi format
        datetime_col = 'tpep_pickup_datetime'
        lat_lon_format = True
    elif 'lpep_pickup_datetime' in df.columns:
        # Green taxi format
        datetime_col = 'lpep_pickup_datetime'
        lat_lon_format = True
    elif 'pickup_datetime' in df.columns:
        # Legacy format
        datetime_col = 'pickup_datetime'
        lat_lon_format = True
    else:
        # 2023+ format - might have different column names
        datetime_col = next((col for col in df.columns if 'pickup' in col.lower() and 'time' in col.lower()), 
                           'pickup_datetime')
        lat_lon_format = 'pickup_latitude' in df.columns
    
    # Basic data cleaning steps
    # Remove invalid passenger count records
    if 'passenger_count' in df.columns:
        df = df[df.passenger_count > 0]
    
    # Remove invalid trip distance and duration
    if 'trip_distance' in df.columns:
        df = df[df.trip_distance > 0]
    if 'trip_duration' in df.columns:
        df = df[df.trip_duration > 0]
    
    # Ensure valid latitude/longitude coordinates if present
    if lat_lon_format:
        if 'pickup_latitude' in df.columns and 'pickup_longitude' in df.columns:
            valid_lat = (df.pickup_latitude >= 40.5) & (df.pickup_latitude <= 41.0)
            valid_lon = (df.pickup_longitude >= -74.1) & (df.pickup_longitude <= -73.7)
            df = df[valid_lat & valid_lon]
    else:
        # For newer formats that might use location IDs instead of coordinates
        print("Using location ID based format")
        # Here we'd need to join with a lookup table to get coordinates
        # This is a placeholder - actual implementation depends on the 2025 data format
        if 'PULocationID' in df.columns:
            # Filter out invalid location IDs if needed
            pass
    
    # Add time features
    if datetime_col in df.columns:
        if is_dask:
            # Dask data processing
            df['hour'] = df[datetime_col].dt.hour
            df['day_of_week'] = df[datetime_col].dt.dayofweek
        else:
            # Pandas data processing
            df['hour'] = pd.to_datetime(df[datetime_col]).dt.hour
            df['day_of_week'] = pd.to_datetime(df[datetime_col]).dt.dayofweek
    
    # Add grid block ID (divide NYC into grid)
    df = add_block_ids(df, lat_lon_format)
    
    if is_dask:
        df = df.compute()  # Convert Dask DataFrame to Pandas DataFrame
    
    print(f"Data cleaning complete, remaining rows: {len(df)}")
    return df

def add_block_ids(df, lat_lon_format=True, location_id_mapping=None):
    """
    Map latitude/longitude coordinates or location IDs to grid block IDs.
    
    Parameters:
        df (DataFrame): DataFrame with location information
        lat_lon_format (bool): If True, use lat/lon columns, otherwise use location IDs
        location_id_mapping (dict, optional): Mapping from location IDs to block IDs
        
    Returns:
        DataFrame: DataFrame with added block_id column
    """
    if lat_lon_format and 'pickup_latitude' in df.columns and 'pickup_longitude' in df.columns:
        # Define NYC geographic boundaries
        lat_min, lat_max = 40.5, 41.0
        lon_min, lon_max = -74.1, -73.7
        
        # Create 10x10 grid
        grid_size = 10
        lat_step = (lat_max - lat_min) / grid_size
        lon_step = (lon_max - lon_min) / grid_size
        
        # Calculate block ID
        df['lat_bin'] = ((df.pickup_latitude - lat_min) / lat_step).astype(int)
        df['lon_bin'] = ((df.pickup_longitude - lon_min) / lon_step).astype(int)
        
        # Clip bin indices to prevent out-of-bounds
        df['lat_bin'] = df['lat_bin'].clip(0, grid_size-1)
        df['lon_bin'] = df['lon_bin'].clip(0, grid_size-1)
        
        # Compute unique block ID (lat_bin * grid_size + lon_bin)
        df['block_id'] = df['lat_bin'] * grid_size + df['lon_bin']
        
        # Clean up temporary columns
        df = df.drop(['lat_bin', 'lon_bin'], axis=1)
    
    elif not lat_lon_format and 'PULocationID' in df.columns:
        # For location ID based format (2023+ NYC TLC data)
        if location_id_mapping is None:
            # TODO: Implement proper mapping for 2025 data
            # For now, we'll use a simple modulo mapping as placeholder
            print("Warning: Using placeholder mapping for location IDs to blocks")
            df['block_id'] = df['PULocationID'] % 100
        else:
            # Use provided mapping
            df['block_id'] = df['PULocationID'].map(location_id_mapping)
    
    else:
        # If neither format is available, create random block IDs for testing
        print("Warning: Could not determine location format, creating random block IDs")
        df['block_id'] = np.random.randint(0, 100, size=len(df))
    
    return df

def load_location_mapping(filepath):
    """
    Load mapping from location IDs to coordinates for newer TLC data formats.
    
    Parameters:
        filepath (str): Path to mapping file
        
    Returns:
        dict: Mapping from location IDs to (lat, lon) coordinates
    """
    # TODO: Implement based on actual 2025 NYC TLC data format
    # This is a placeholder
    try:
        mapping_df = pd.read_csv(filepath)
        # Assume the file has LocationID, Latitude, Longitude columns
        mapping = {
            row['LocationID']: (row['Latitude'], row['Longitude'])
            for _, row in mapping_df.iterrows()
        }
        return mapping
    except Exception as e:
        print(f"Error loading location mapping: {e}")
        return {}

print(">>> 已进入 loaddata 主程序 <<<")
import sys
print(">>> argv:", sys.argv)

if __name__ == "__main__":
    import argparse, os

    parser = argparse.ArgumentParser(
        description="加载并清洗 NYC TLC 出租车原始数据"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="原始数据所在目录（Parquet 或 CSV 文件）"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="清洗后数据要写入的目录"
    )
    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output, exist_ok=True)

    # 假设文件名固定，你也可以改成遍历目录
    raw_fp = os.path.join(args.input, "yellow_tripdata_2025-01.parquet")

    # 1. 载入数据
    df_raw = load_trip_data(raw_fp)

    # 2. 清洗数据
    df_clean = clean_trip_data(df_raw)

    # 3. 保存结果（这里用 pickle，也可以改成 CSV）
    out_fp = os.path.join(args.output, "trips_cleaned.pkl")
    df_clean.to_pickle(out_fp)
    print(f"已保存清洗后数据到：{out_fp}")
