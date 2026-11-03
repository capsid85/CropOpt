import os
import pandas as pd
import numpy as np
import requests

def get_data_path(filename):
    """Get absolute path to data file."""
    # Assume data is in a folder named 'data' in the workspace root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, 'data', filename)

def load_raw_data():
    """Load historical, economic, and district soil data."""
    dist_df = pd.read_csv(get_data_path('districts.csv'))
    history_df = pd.read_csv(get_data_path('crop_history.csv'))
    econ_df = pd.read_csv(get_data_path('crop_economics.csv'))
    return dist_df, history_df, econ_df

def preprocess_soil_data(dist_df):
    """
    Applies preprocessing to soil data.
    - Median imputation for missing numeric values.
    - One-hot encoding for categorical soil texture classes.
    - Computes an engineered Soil Quality Index (SQI).
    """
    df = dist_df.copy()
    
    # 1. Median imputation for numeric features
    num_cols = ['pH', 'organic_carbon', 'phosphorus', 'potassium']
    for col in num_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
            
    # 2. One-hot encoding for soil texture classes
    # List of expected textures to make it consistent across selections
    expected_textures = ['Sandy Clay Loam', 'Clay Loam', 'Clay', 'Sandy Loam', 'Loam', 'Sandy Clay']
    for texture in expected_textures:
        col_name = f"texture_{texture.lower().replace(' ', '_')}"
        df[col_name] = (df['soil_texture'] == texture).astype(int)
        
    # 3. Feature engineering: Soil Quality Index (SQI)
    # OC is usually < 1.5% in Indian soils. phosphorus normal range: 10-50 kg/ha. potassium normal range: 100-400 kg/ha.
    # We scale them to 0-1 based on the range in districts and take a weighted sum.
    oc_min, oc_max = df['organic_carbon'].min(), df['organic_carbon'].max()
    p_min, p_max = df['phosphorus'].min(), df['phosphorus'].max()
    k_min, k_max = df['potassium'].min(), df['potassium'].max()
    
    oc_norm = (df['organic_carbon'] - oc_min) / (oc_max - oc_min + 1e-5)
    p_norm = (df['phosphorus'] - p_min) / (p_max - p_min + 1e-5)
    k_norm = (df['potassium'] - k_min) / (k_max - k_min + 1e-5)
    
    # SQI scaled to 0-10
    df['soil_quality_index'] = round((oc_norm * 0.4 + p_norm * 0.3 + k_norm * 0.3) * 10, 2)
    
    return df

def get_integrated_dataset(history_df, dist_df):
    """
    Merges historical yield statistics with preprocessed soil parameters and SQI.
    """
    preprocessed_soil = preprocess_soil_data(dist_df)
    # Join on district
    integrated_df = pd.merge(history_df, preprocessed_soil, on='district', how='left')
    return integrated_df

def detect_price_outliers_iqr(econ_df):
    """
    Detects price outliers in the economics data using the Interquartile Range (IQR) method.
    Returns a list of dictionaries with outlier information, plus lower and upper bounds.
    """
    prices = econ_df['price_rs_qtl']
    q1 = prices.quantile(0.25)
    q3 = prices.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers_df = econ_df[(econ_df['price_rs_qtl'] < lower_bound) | (econ_df['price_rs_qtl'] > upper_bound)]
    
    outliers_list = []
    for _, row in outliers_df.iterrows():
        outliers_list.append({
            'crop': row['crop'],
            'season': row['season'],
            'price': row['price_rs_qtl'],
            'bounds': (round(lower_bound, 2), round(upper_bound, 2))
        })
    return outliers_list, round(lower_bound, 2), round(upper_bound, 2)

def get_zscore_scalers(history_df, econ_df):
    """
    Computes mean and std of historical yield and prices to allow Z-score scaling.
    """
    yield_mean = history_df['yield_q_ha'].mean()
    yield_std = history_df['yield_q_ha'].std()
    
    price_mean = econ_df['price_rs_qtl'].mean()
    price_std = econ_df['price_rs_qtl'].std()
    
    return {
        'yield_mean': yield_mean,
        'yield_std': yield_std,
        'price_mean': price_mean,
        'price_std': price_std
    }

def fetch_live_weather(lat, lon, season):
    """
    Fetches weather data from Open-Meteo API.
    If the API call fails or there is no internet connection, it returns None.
    Computes representative temperature and seasonal rainfall aggregates.
    """
    # Open-Meteo forecast API for past weather (simulated current season)
    # We fetch a 14-day forecast and scale it to seasonal parameters to represent "current conditions"
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,precipitation_sum&timezone=auto"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            daily = data.get('daily', {})
            
            tmax_list = daily.get('temperature_2m_max', [])
            precip_list = daily.get('precipitation_sum', [])
            
            # Clean lists
            tmax_list = [t for t in tmax_list if t is not None]
            precip_list = [p for p in precip_list if p is not None]
            
            if not tmax_list or not precip_list:
                return None
                
            avg_tmax = np.mean(tmax_list)
            # Scale daily precip to seasonal equivalent
            # Kharif (~120 days of monsoon), Rabi (~150 days), Summer (~90 days)
            days_map = {'Kharif': 120, 'Rabi': 150, 'Summer': 90}
            season_days = days_map.get(season, 120)
            
            # Calculate representative seasonal rainfall (avg daily precipitation sum * season length)
            scaled_rainfall = np.mean(precip_list) * season_days
            
            # Bound scaled rainfall based on realistic season parameters
            if season == 'Kharif':
                scaled_rainfall = max(200.0, min(1500.0, scaled_rainfall))
            elif season == 'Rabi':
                scaled_rainfall = max(20.0, min(300.0, scaled_rainfall))
            else: # Summer
                scaled_rainfall = max(10.0, min(250.0, scaled_rainfall))
                
            return {
                'rainfall_mm': round(scaled_rainfall, 2),
                'tmax_c': round(avg_tmax, 2),
                'is_live': True
            }
    except Exception:
        pass
    return None

def get_district_weather_fallback(district, season, history_df):
    """
    Fallback method to fetch historical weather averages for the district and season.
    """
    dist_season_data = history_df[
        (history_df['district'].str.lower() == district.str.lower()) & 
        (history_df['season'].str.lower() == season.str.lower())
    ]
    if dist_season_data.empty:
        # Global averages as ultimate fallback
        return {'rainfall_mm': 500.0, 'tmax_c': 30.0, 'is_live': False}
        
    avg_rf = dist_season_data['rainfall_mm'].mean()
    avg_tmax = dist_season_data['tmax_c'].mean()
    return {
        'rainfall_mm': round(avg_rf, 2),
        'tmax_c': round(avg_tmax, 2),
        'is_live': False
    }
