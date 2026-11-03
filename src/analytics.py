import numpy as np
import pandas as pd

def classify_crop_risk(variance, break_even_prob):
    """
    Classifies risk band (Low, Moderate, High) based on:
    - Yield variance (thresholds: 30 and 60)
    - Break-even probability (thresholds: 0.75 and 0.55)
    The classification takes the worst of the two metrics.
    """
    # Variance band
    if variance < 30.0:
        var_risk = 'Low'
    elif variance < 60.0:
        var_risk = 'Moderate'
    else:
        var_risk = 'High'
        
    # Break-even probability band
    if break_even_prob >= 0.75:
        prob_risk = 'Low'
    elif break_even_prob >= 0.55:
        prob_risk = 'Moderate'
    else:
        prob_risk = 'High'
        
    # Worst-case risk mapping
    risk_rank = {'Low': 1, 'Moderate': 2, 'High': 3}
    final_rank = max(risk_rank[var_risk], risk_rank[prob_risk])
    
    inv_risk_rank = {1: 'Low', 2: 'Moderate', 3: 'High'}
    return inv_risk_rank[final_rank]

def calculate_portfolio_metrics(allocation, sim_results, econ_df, total_area):
    """
    Calculates expected profit, ROI, variance, and break-even probability for a given land allocation.
    We combine the 1,000 simulated yield trials for each crop to compute portfolio-level distributions.
    
    Parameters:
    - allocation: dict, {crop_name: area_ha}
    - sim_results: dict, {crop_name: sim_data_dict_from_monte_carlo}
    - econ_df: DataFrame, containing crop economics
    - total_area: float, total farm area
    
    Returns:
    - dict of portfolio metrics
    """
    num_sims = 1000
    portfolio_profits = np.zeros(num_sims)
    total_cost = 0.0
    
    # Extract economic parameters for crops in allocation
    econ_lookup = econ_df.drop_duplicates(subset=['crop']).set_index('crop').to_dict('index')
    
    for crop, area in allocation.items():
        if area <= 0:
            continue
            
        crop_econ = econ_lookup.get(crop)
        if not crop_econ:
            continue
            
        cost_per_ha = crop_econ['cost_rs_ha']
        price_per_qtl = crop_econ['price_rs_qtl']
        
        # Add to total cost
        total_cost += cost_per_ha * area
        
        # Portfolio yield distribution
        sim_yields = sim_results[crop]['sim_yields']
        
        # Profit trial = (Yield * Price - Cost) * Area
        crop_profits = (sim_yields * price_per_qtl - cost_per_ha) * area
        portfolio_profits += crop_profits

    expected_profit = np.mean(portfolio_profits)
    portfolio_variance = np.var(portfolio_profits)
    portfolio_std = np.std(portfolio_profits)
    
    # Portfolio ROI (%)
    if total_cost > 0:
        expected_roi = (expected_profit / total_cost) * 100
        roi_trials = (portfolio_profits / total_cost) * 100
    else:
        expected_roi = 0.0
        roi_trials = np.zeros(num_sims)
        
    # Break-even probability of the portfolio (P(Profit > 0))
    break_even_prob = np.mean(portfolio_profits > 0)
    
    # 10th and 90th percentiles
    downside_profit = np.percentile(portfolio_profits, 10)
    upside_profit = np.percentile(portfolio_profits, 90)

    # Classify portfolio risk using yield-stability terms
    # We can approximate portfolio risk using the standard deviation and break-even probability
    # Scale portfolio variance back to an "effective yield variance" per hectare
    effective_variance = portfolio_variance / (total_area**2) if total_area > 0 else 0.0
    risk_band = classify_crop_risk(effective_variance, break_even_prob)

    return {
        'expected_profit': expected_profit,
        'profit_std': portfolio_std,
        'expected_roi': expected_roi,
        'break_even_prob': break_even_prob,
        'downside_profit': downside_profit,
        'upside_profit': upside_profit,
        'risk_band': risk_band,
        'total_cost': total_cost,
        'profit_distribution': portfolio_profits
    }

def normalize_features(portfolios_metrics):
    """
    Min-max normalizes expected profit, yield stability (break-even prob), and ROI
    across all portfolios to prepare for RAS computation.
    """
    df = pd.DataFrame(portfolios_metrics)
    
    def min_max_norm(series):
        s_min = series.min()
        s_max = series.max()
        if s_max == s_min:
            return pd.Series(1.0, index=series.index)
        return (series - s_min) / (s_max - s_min)

    df['profit_norm'] = min_max_norm(df['expected_profit'])
    df['stability_norm'] = min_max_norm(df['break_even_prob'])
    df['roi_norm'] = min_max_norm(df['expected_roi'])
    
    return df.to_dict('records')

def compute_risk_adjusted_score(portfolios, w1, w2, w3):
    """
    Computes RAS = w1 * profit_norm + w2 * stability_norm + w3 * roi_norm
    Ranks portfolios by RAS and returns them sorted.
    """
    # 1. Normalize
    normed_portfolios = normalize_features(portfolios)
    
    # 2. Compute RAS
    for p in normed_portfolios:
        p['ras'] = w1 * p['profit_norm'] + w2 * p['stability_norm'] + w3 * p['roi_norm']
        p['ras'] = round(p['ras'], 4)
        
    # 3. Sort
    sorted_portfolios = sorted(normed_portfolios, key=lambda x: x['ras'], reverse=True)
    return sorted_portfolios

def run_ras_sensitivity_analysis(portfolios):
    """
    Evaluates rankings of portfolios across three farmer risk profiles:
    - Profit-first: w1=0.6, w2=0.2, w3=0.2
    - Balanced: w1=0.5, w2=0.3, w3=0.2
    - Risk-averse: w1=0.3, w2=0.5, w3=0.2
    
    Returns a dataframe summarizing the scores and ranks of each portfolio across profiles.
    """
    profiles = {
        'Profit-first': (0.6, 0.2, 0.2),
        'Balanced': (0.5, 0.3, 0.2),
        'Risk-averse': (0.3, 0.5, 0.2)
    }
    
    records = []
    for profile_name, weights in profiles.items():
        w1, w2, w3 = weights
        ranked = compute_risk_adjusted_score(portfolios, w1, w2, w3)
        for rank, p in enumerate(ranked, start=1):
            records.append({
                'Profile': profile_name,
                'Portfolio_Name': p['name'],
                'Score': p['ras'],
                'Rank': rank
            })
            
    return pd.DataFrame(records)
