import pandas as pd
import numpy as np
import requests
import time
import os

def clean_apy_data():
    apy_df = pd.read_csv('APY.csv')
    # Clean column names
    apy_df.columns = apy_df.columns.str.strip()
    # Clean string columns
    apy_df['District'] = apy_df['District'].astype(str).str.strip().str.title()
    apy_df['Season'] = apy_df['Season'].astype(str).str.strip()
    apy_df['Crop'] = apy_df['Crop'].astype(str).str.strip()
    
    return apy_df

def fetch_weather(lat, lon, start_year, end_year):
    # Fetch historical data
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_year}-01-01&end_date={end_year}-12-31&daily=temperature_2m_max,precipitation_sum&timezone=auto"
    try:
        res = requests.get(url)
        if res.status_code == 200:
            return res.json()
    except Exception as e:
        print(f"Failed to fetch weather for {lat},{lon}: {e}")
    return None

def build_dataset():
    # 1. Load districts
    dist_df = pd.read_csv('districts.csv')
    target_districts = dist_df['district'].str.title().tolist()
    
    # 2. Load Econ to get crops
    econ_df = pd.read_csv('crop_economics.csv')
    target_crops = econ_df['crop'].unique().tolist()
    
    # 3. Load and clean APY
    print("Loading APY data...")
    apy = clean_apy_data()
    
    # Special mapping for APY crop names to our crop names if needed
    crop_mapping = {
        'Cotton(lint)': 'Cotton',
        'Soyabean': 'Soybean',
        'Gram': 'Bengal Gram',
        'Moong(Green Gram)': 'Green Gram',
        'Urad': 'Black Gram',
        'Wheat': 'Wheat',
        'Groundnut': 'Groundnut',
        'Ragi': 'Ragi'
    }
    apy['Crop'] = apy['Crop'].replace(crop_mapping)
    
    # Filter dataset
    filtered_apy = apy[apy['District'].isin(target_districts) & apy['Crop'].isin(target_crops)].copy()
    print(f"Filtered APY records: {len(filtered_apy)}")
    
    # Drop NAs in Area and Production
    filtered_apy = filtered_apy.dropna(subset=['Area', 'Production'])
    
    # Find min and max year
    if len(filtered_apy) == 0:
        print("No matching records found. Exiting.")
        return
        
    start_year = int(filtered_apy['Crop_Year'].min())
    end_year = int(filtered_apy['Crop_Year'].max())
    
    print(f"Data spans from {start_year} to {end_year}")
    
    # 4. Fetch Weather for each district and compute seasonal aggregates
    weather_cache = {}
    
    print("Fetching historical weather from Open-Meteo...")
    for _, row in dist_df.iterrows():
        d_name = row['district'].title()
        lat, lon = row['latitude'], row['longitude']
        print(f"  Fetching for {d_name}...")
        w_data = fetch_weather(lat, lon, start_year, end_year)
        if w_data and 'daily' in w_data:
            df_w = pd.DataFrame(w_data['daily'])
            df_w['time'] = pd.to_datetime(df_w['time'])
            df_w['year'] = df_w['time'].dt.year
            df_w['month'] = df_w['time'].dt.month
            
            def get_season(m):
                if m in [6,7,8,9,10]: return 'Kharif'
                elif m in [11,12,1,2,3]: return 'Rabi'
                else: return 'Summer'
                
            df_w['season'] = df_w['month'].apply(get_season)
            
            agg_w = df_w.groupby(['year', 'season']).agg(
                rainfall_mm=('precipitation_sum', 'sum'),
                tmax_c=('temperature_2m_max', 'mean')
            ).reset_index()
            
            weather_cache[d_name] = agg_w
        time.sleep(1) # rate limit safety
        
    # 5. Merge Data
    final_records = []
    
    for _, row in filtered_apy.iterrows():
        d_name = row['District']
        y = row['Crop_Year']
        c_season = row['Season'].title()
        
        season_map = {
            'Kharif': 'Kharif',
            'Autumn': 'Kharif',
            'Winter': 'Rabi',
            'Rabi': 'Rabi',
            'Summer': 'Summer',
            'Whole Year': 'Kharif'
        }
        mapped_season = season_map.get(c_season, 'Kharif')
        
        w_df = weather_cache.get(d_name)
        rf, tmax = 500.0, 30.0 # fallbacks
        if w_df is not None:
            match = w_df[(w_df['year'] == y) & (w_df['season'] == mapped_season)]
            if len(match) > 0:
                rf = match.iloc[0]['rainfall_mm']
                tmax = match.iloc[0]['tmax_c']
                
        # Handle production data which is sometimes a string with commas
        try:
            prod = float(str(row['Production']).replace(',', ''))
            area = float(str(row['Area']).replace(',', ''))
        except:
            continue
            
        if area > 0:
            yield_q_ha = (prod / area) * 10
        else:
            yield_q_ha = 0
            
        if yield_q_ha > 0 and yield_q_ha < 200:
            final_records.append({
                'year': int(y),
                'district': d_name,
                'crop': row['Crop'],
                'season': mapped_season,
                'yield_q_ha': round(yield_q_ha, 2),
                'rainfall_mm': round(rf, 2),
                'tmax_c': round(tmax, 2)
            })
            
    final_df = pd.DataFrame(final_records)
    final_df = final_df[final_df['crop'].isin(target_crops)]
    
    print(f"Final dataset has {len(final_df)} records.")
    
    final_df.to_csv('crop_history.csv', index=False)
    print("Successfully built actual data crop_history.csv!")

if __name__ == '__main__':
    build_dataset()
