import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

class CropYieldModel:
    """
    Fits a Ridge Regression model (with OLS baseline) for a specific crop, district, and season.
    Includes soil health parameters and weather variables as predictors.
    Uses 5-fold cross validation via RidgeCV to select the regularization strength.
    """
    def __init__(self, crop, district, season):
        self.crop = crop
        self.district = district
        self.season = season
        self.model = None
        self.r2 = 0.0
        self.rmse = 0.0
        self.mae = 0.0
        self.alpha_ = 1.0
        self.residual_std = 1.0
        self.coef = []
        self.intercept = 0.0
        
        # OLS Baseline validation metrics
        self.ols_model = None
        self.ols_r2 = 0.0
        self.ols_rmse = 0.0
        self.ols_mae = 0.0

        # Features list including weather and preprocessed soil variables + SQI
        self.features_list = [
            'rainfall_mm', 'tmax_c', 'pH', 'organic_carbon', 'phosphorus', 'potassium', 'soil_quality_index',
            'texture_sandy_clay_loam', 'texture_clay_loam', 'texture_clay', 'texture_sandy_loam', 'texture_loam', 'texture_sandy_clay'
        ]

    def fit(self, history_df):
        """Fit OLS Baseline and Ridge regression on historical data with 80/20 train/test split."""
        # Filter data for this crop, district, and season
        subset = history_df[
            (history_df['crop'].str.lower() == self.crop.lower()) &
            (history_df['district'].str.lower() == self.district.lower()) &
            (history_df['season'].str.lower() == self.season.lower())
        ]
        
        if len(subset) < 10:
            # If not enough local district data, fall back to crop + season across all districts
            subset = history_df[
                (history_df['crop'].str.lower() == self.crop.lower()) &
                (history_df['season'].str.lower() == self.season.lower())
            ]
            
        if len(subset) < 5:
            # Raise exception if we still have insufficient data
            raise ValueError(f"Insufficient historical data to fit model for {self.crop} in {self.season}.")

        # Extract features and targets
        X = subset[self.features_list].values
        y = subset['yield_q_ha'].values

        # 80/20 train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 1. Fit OLS Baseline Model
        self.ols_model = LinearRegression()
        self.ols_model.fit(X_train, y_train)
        y_pred_ols = self.ols_model.predict(X_test)
        self.ols_r2 = r2_score(y_test, y_pred_ols)
        if self.ols_r2 < 0:
            self.ols_r2 = 0.0
        self.ols_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ols))
        self.ols_mae = mean_absolute_error(y_test, y_pred_ols)

        # 2. Fit RidgeCV (Ridge Regression with 5-fold cross-validation)
        alphas = np.logspace(-3, 3, 100)
        cv_folds = min(5, len(X_train))
        if cv_folds < 2:
            self.model = RidgeCV(alphas=alphas, cv=None)
        else:
            self.model = RidgeCV(alphas=alphas, cv=cv_folds)
            
        self.model.fit(X_train, y_train)
        
        # Coefficients and intercept
        self.alpha_ = self.model.alpha_
        self.coef = self.model.coef_
        self.intercept = self.model.intercept_

        # Evaluate Ridge model on test set
        y_pred = self.model.predict(X_test)
        self.r2 = r2_score(y_test, y_pred)
        if self.r2 < 0:
            self.r2 = max(0.0, self.r2)
            
        self.rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        self.mae = mean_absolute_error(y_test, y_pred)

        # Calculate residual standard deviation on training set
        y_train_pred = self.model.predict(X_train)
        self.residual_std = np.std(y_train - y_train_pred)
        if self.residual_std < 0.1:
            self.residual_std = 0.1

        return self

def run_monte_carlo_simulation(model_fit, history_df, district, season, crop, num_simulations=1000):
    """
    Runs Monte Carlo simulation of crop yield using the fitted Ridge model and soil parameters.
    Samples rainfall and temperature using the historical district covariance.
    Predicts crop yields using the full multi-feature model, incorporating weather variance.
    """
    # 1. Extract historical weather subset to compute multivariate parameters (covariance)
    subset = history_df[
        (history_df['district'].str.lower() == district.lower()) &
        (history_df['season'].str.lower() == season.lower())
    ]
    
    if subset.empty:
        # Ultimate fallback if no weather records
        mean_rf = 500.0
        mean_tmax = 30.0
        cov_matrix = [[50**2, 0.0], [0.0, 1.5**2]]
    else:
        weather_data = subset[['rainfall_mm', 'tmax_c']].drop_duplicates()
        if len(weather_data) < 3:
            # Use global averages if not enough points
            mean_rf = subset['rainfall_mm'].mean()
            mean_tmax = subset['tmax_c'].mean()
            cov_matrix = [[subset['rainfall_mm'].var() or 2500.0, 0.0], [0.0, subset['tmax_c'].var() or 2.25]]
        else:
            mean_rf = weather_data['rainfall_mm'].mean()
            mean_tmax = weather_data['tmax_c'].mean()
            cov_matrix = np.cov(weather_data['rainfall_mm'], weather_data['tmax_c'])

    # Ensure covariance matrix is positive semi-definite
    cov_matrix = np.nan_to_num(cov_matrix)
    cov_matrix[0, 0] = max(10.0, cov_matrix[0, 0])
    cov_matrix[1, 1] = max(0.1, cov_matrix[1, 1])

    # 2. Sample 1000 scenarios from Multivariate Normal Distribution
    weather_scenarios = np.random.multivariate_normal([mean_rf, mean_tmax], cov_matrix, num_simulations)
    
    # Clip simulated values to physical bounds
    sim_rf = np.clip(weather_scenarios[:, 0], 10.0, None)
    sim_tmax = np.clip(weather_scenarios[:, 1], 15.0, None)

    # 3. Look up district soil features (constant across simulations)
    soil_features = {}
    soil_cols = [
        'pH', 'organic_carbon', 'phosphorus', 'potassium', 'soil_quality_index',
        'texture_sandy_clay_loam', 'texture_clay_loam', 'texture_clay', 
        'texture_sandy_loam', 'texture_loam', 'texture_sandy_clay'
    ]
    
    dist_subset = history_df[history_df['district'].str.lower() == district.lower()]
    if not dist_subset.empty:
        first_row = dist_subset.iloc[0]
        for col in soil_cols:
            soil_features[col] = first_row[col]
    else:
        # Fallback values if district not found
        fallback_values = {
            'pH': 7.0, 'organic_carbon': 0.5, 'phosphorus': 15.0, 'potassium': 250.0, 'soil_quality_index': 5.0,
            'texture_sandy_clay_loam': 0, 'texture_clay_loam': 0, 'texture_clay': 0,
            'texture_sandy_loam': 1, 'texture_loam': 0, 'texture_sandy_clay': 0
        }
        for col in soil_cols:
            soil_features[col] = fallback_values[col]

    # Create simulation feature matrix matching the trained model's feature structure
    sim_df = pd.DataFrame({
        'rainfall_mm': sim_rf,
        'tmax_c': sim_tmax
    })
    for col in soil_cols:
        sim_df[col] = soil_features[col]
        
    X_sim = sim_df[model_fit.features_list].values

    # 4. Predict yields through Ridge model and add residual uncertainty
    residuals = np.random.normal(0, model_fit.residual_std, num_simulations)
    sim_yields = model_fit.model.predict(X_sim) + residuals
    sim_yields = np.clip(sim_yields, 0.1, None)  # yields cannot be negative

    return {
        'crop': crop,
        'sim_yields': sim_yields,
        'sim_rf': sim_rf,
        'sim_tmax': sim_tmax,
        'mean_yield': np.mean(sim_yields),
        'yield_variance': np.var(sim_yields),
        'yield_std': np.std(sim_yields),
        'downside_yield': np.percentile(sim_yields, 10),  # 10th percentile (downside risk)
        'upside_yield': np.percentile(sim_yields, 90)     # 90th percentile (favorable scenario)
    }
