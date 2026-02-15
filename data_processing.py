import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from datetime import timedelta
import logging
from src.config import SCALER_PATH, SEQ_LENGTH, PRED_LENGTH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_process_weather_data(filepath):
    logger.info(f"Loading weather data from {filepath}")
    df = pd.read_csv(filepath)
    
    # Consolidate geo-temporal (mean across locations)
    # Group by time and take mean of numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if 'location_id' in numeric_cols:
        numeric_cols = numeric_cols.drop('location_id')
        
    df_grouped = df.groupby('time')[numeric_cols].mean().reset_index()
    
    # Timezone handling
    df_grouped['time'] = pd.to_datetime(df_grouped['time']).dt.tz_localize('UTC')
    
    return df_grouped

def load_and_process_pvpc_data(filepath):
    logger.info(f"Loading PVPC data from {filepath}")
    df = pd.read_csv(filepath, sep=';')
    
    # Standardize columns
    # We only need datetime and value
    df = df.rename(columns={'datetime': 'time', 'value': 'price'})
    df = df[['time', 'price']]
    
    # Timezone handling
    # PVPC data is usually in ISO format with offsets. Convert to UTC.
    df['time'] = pd.to_datetime(df['time'], utc=True)
    
    # Sort
    df = df.sort_values('time')
    
    return df

def merge_data(weather_df, pvpc_df):
    logger.info("Merging datasets")
    # Merge on time
    merged_df = pd.merge(pvpc_df, weather_df, on='time', how='inner')
    merged_df = merged_df.sort_values('time').reset_index(drop=True)
    return merged_df

def feature_engineering(df):
    logger.info("Generating features")
    
    # Cyclical features
    df['hour'] = df['time'].dt.hour
    df['month'] = df['time'].dt.month
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Lags
    df['price_lag_24h'] = df['price'].shift(24)
    df['price_lag_168h'] = df['price'].shift(168)
    
    # Drop rows with NaNs (initial lag period)
    df = df.dropna().reset_index(drop=True)
    
    # Drop raw time columns used for features
    df = df.drop(columns=['hour', 'month'])
    
    return df

def scale_data(df, is_training=True):
    logger.info("Scaling data")
    
    # Separate time from features
    time_col = df['time']
    features_df = df.drop(columns=['time'])
    
    scaler = None
    if is_training:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(features_df)
        joblib.dump(scaler, SCALER_PATH)
    else:
        scaler = joblib.load(SCALER_PATH)
        scaled_data = scaler.transform(features_df)
        
    scaled_df = pd.DataFrame(scaled_data, columns=features_df.columns)
    scaled_df['time'] = time_col.values
    
    return scaled_df, scaler

def create_sequences(df, seq_length=SEQ_LENGTH, pred_length=PRED_LENGTH, target_col='price'):
    logger.info("Creating sequences")
    # Ensure target column is the first feature or handle it explicitly
    # For this architecture, we usually pass all features as input.
    # The target is likely 'price'.
    
    data_values = df.drop(columns=['time']).values
    target_idx = df.drop(columns=['time']).columns.get_loc(target_col)
    
    X, y = [], []
    
    for i in range(len(data_values) - seq_length - pred_length):
        X.append(data_values[i:(i + seq_length)])
        # Predict the NEXT 24 hours of PRICE
        y.append(data_values[(i + seq_length):(i + seq_length + pred_length), target_idx])
        
    return np.array(X), np.array(y)

if __name__ == "__main__":
    # Test run
    from src.config import WEATHER_DATA_PATH, PVPC_DATA_PATH, PROCESSED_DATA_PATH
    
    try:
        w_df = load_and_process_weather_data(WEATHER_DATA_PATH)
        p_df = load_and_process_pvpc_data(PVPC_DATA_PATH)
        m_df = merge_data(w_df, p_df)
        f_df = feature_engineering(m_df)
        s_df, _ = scale_data(f_df, is_training=True)
        
        # Save processed data for inspection/model training
        s_df.to_csv(PROCESSED_DATA_PATH, index=False)
        logger.info(f"Processed data saved to {PROCESSED_DATA_PATH}")
        
        X, y = create_sequences(s_df)
        print(f"X shape: {X.shape}, y shape: {y.shape}")
    except Exception as e:
        logger.error(f"Error in data pipeline: {e}")
        raise
