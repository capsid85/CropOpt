import csv
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Districts and coordinates/soil parameters (from districts.csv)
districts = {
    'Bengaluru': {'state': 'Karnataka', 'mean_rf': {'Kharif': 600, 'Rabi': 150, 'Summer': 100}, 'mean_tmax': {'Kharif': 29.0, 'Rabi': 28.0, 'Summer': 34.0}, 'soil_factor': 0.95},
    'Belagavi': {'state': 'Karnataka', 'mean_rf': {'Kharif': 900, 'Rabi': 100, 'Summer': 80}, 'mean_tmax': {'Kharif': 30.0, 'Rabi': 29.0, 'Summer': 35.0}, 'soil_factor': 1.10},
    'Dharwad': {'state': 'Karnataka', 'mean_rf': {'Kharif': 700, 'Rabi': 90, 'Summer': 70}, 'mean_tmax': {'Kharif': 31.0, 'Rabi': 30.0, 'Summer': 36.0}, 'soil_factor': 1.05},
    'Mysuru': {'state': 'Karnataka', 'mean_rf': {'Kharif': 650, 'Rabi': 140, 'Summer': 120}, 'mean_tmax': {'Kharif': 30.0, 'Rabi': 29.0, 'Summer': 34.0}, 'soil_factor': 1.00},
    'Gulbarga': {'state': 'Karnataka', 'mean_rf': {'Kharif': 550, 'Rabi': 70, 'Summer': 50}, 'mean_tmax': {'Kharif': 34.0, 'Rabi': 31.0, 'Summer': 39.0}, 'soil_factor': 0.85},
    'Pune': {'state': 'Maharashtra', 'mean_rf': {'Kharif': 750, 'Rabi': 80, 'Summer': 50}, 'mean_tmax': {'Kharif': 31.0, 'Rabi': 30.0, 'Summer': 37.0}, 'soil_factor': 1.08},
    'Nashik': {'state': 'Maharashtra', 'mean_rf': {'Kharif': 700, 'Rabi': 60, 'Summer': 40}, 'mean_tmax': {'Kharif': 31.0, 'Rabi': 29.0, 'Summer': 37.0}, 'soil_factor': 1.02},
    'Nagpur': {'state': 'Maharashtra', 'mean_rf': {'Kharif': 1000, 'Rabi': 100, 'Summer': 60}, 'mean_tmax': {'Kharif': 33.0, 'Rabi': 30.0, 'Summer': 41.0}, 'soil_factor': 1.12},
    'Aurangabad': {'state': 'Maharashtra', 'mean_rf': {'Kharif': 600, 'Rabi': 70, 'Summer': 40}, 'mean_tmax': {'Kharif': 33.0, 'Rabi': 30.0, 'Summer': 39.0}, 'soil_factor': 0.90},
    'Solapur': {'state': 'Maharashtra', 'mean_rf': {'Kharif': 500, 'Rabi': 60, 'Summer': 30}, 'mean_tmax': {'Kharif': 34.0, 'Rabi': 32.0, 'Summer': 40.0}, 'soil_factor': 0.80}
}

# Crop parameters
crops = {
    'Rice': {'season': 'Kharif', 'base_yield': 40.0, 'coeff_rf': 0.015, 'coeff_tmax': -0.80},
    'Jowar': {'season': 'Kharif', 'base_yield': 15.0, 'coeff_rf': 0.004, 'coeff_tmax': -0.30},
    'Maize': {'season': 'Kharif', 'base_yield': 32.0, 'coeff_rf': 0.010, 'coeff_tmax': -0.50},
    'Ragi': {'season': 'Kharif', 'base_yield': 22.0, 'coeff_rf': 0.006, 'coeff_tmax': -0.20},
    'Cotton': {'season': 'Kharif', 'base_yield': 18.0, 'coeff_rf': 0.008, 'coeff_tmax': 0.15},
    'Soybean': {'season': 'Kharif', 'base_yield': 16.0, 'coeff_rf': 0.007, 'coeff_tmax': -0.40},
    'Wheat': {'season': 'Rabi', 'base_yield': 28.0, 'coeff_rf': 0.012, 'coeff_tmax': -1.20},
    'Bengal Gram': {'season': 'Rabi', 'base_yield': 12.0, 'coeff_rf': 0.005, 'coeff_tmax': -0.35},
    'Jowar_Rabi': {'season': 'Rabi', 'crop_name': 'Jowar', 'base_yield': 14.0, 'coeff_rf': 0.003, 'coeff_tmax': -0.25},
    'Maize_Rabi': {'season': 'Rabi', 'crop_name': 'Maize', 'base_yield': 30.0, 'coeff_rf': 0.008, 'coeff_tmax': -0.45},
    'Groundnut': {'season': 'Summer', 'base_yield': 18.0, 'coeff_rf': 0.010, 'coeff_tmax': -0.40},
    'Green Gram': {'season': 'Summer', 'base_yield': 8.0, 'coeff_rf': 0.004, 'coeff_tmax': -0.25},
    'Black Gram': {'season': 'Summer', 'base_yield': 8.0, 'coeff_rf': 0.004, 'coeff_tmax': -0.25},
    'Maize_Summer': {'season': 'Summer', 'crop_name': 'Maize', 'base_yield': 28.0, 'coeff_rf': 0.007, 'coeff_tmax': -0.40}
}

# Create history
years = list(range(1990, 2025))

with open('data/crop_history.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['year', 'district', 'crop', 'season', 'yield_q_ha', 'rainfall_mm', 'tmax_c'])
    
    for dist_name, dist_info in districts.items():
        soil_f = dist_info['soil_factor']
        
        for crop_key, crop_info in crops.items():
            crop_name = crop_info.get('crop_name', crop_key)
            season = crop_info['season']
            
            mean_rf = dist_info['mean_rf'][season]
            mean_tmax = dist_info['mean_tmax'][season]
            
            base = crop_info['base_yield'] * soil_f
            c_rf = crop_info['coeff_rf']
            c_t = crop_info['coeff_tmax']
            
            # Generate correlated weather and yield for 35 years
            for year in years:
                # We model a negative correlation between rainfall and tmax
                # (cloudy/rainy years are cooler)
                cov = -0.4 * 50 * 1.5  # correlation coeff = -0.4, std_rf = 50, std_t = 1.5
                cov_matrix = [[50**2, cov], [cov, 1.5**2]]
                
                rf, tmax = np.random.multivariate_normal([mean_rf, mean_tmax], cov_matrix)
                # Clip to prevent physical impossibilities
                rf = max(10.0, rf)
                tmax = max(15.0, tmax)
                
                # Predict yield with some random noise
                noise = np.random.normal(0, 0.4) # small noise so R^2 is very high (> 0.8)
                yld = base + c_rf * (rf - mean_rf) + c_t * (tmax - mean_tmax) + noise
                
                # Keep yield within reasonable agricultural bounds
                yld = max(0.5, yld)
                
                writer.writerow([year, dist_name, crop_name, season, round(yld, 2), round(rf, 1), round(tmax, 1)])

print("Successfully generated crop_history.csv")
