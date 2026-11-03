import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import our modular engine components
from src.data_ingestion import (
    load_raw_data, preprocess_soil_data, get_zscore_scalers, 
    fetch_live_weather, get_district_weather_fallback,
    get_integrated_dataset, detect_price_outliers_iqr
)
from src.simulation import CropYieldModel, run_monte_carlo_simulation
from src.optimization import (
    solve_crop_optimization, solve_crop_optimization_conservative, 
    run_baseline_comparison
)
from src.analytics import (
    classify_crop_risk, calculate_portfolio_metrics, 
    compute_risk_adjusted_score, run_ras_sensitivity_analysis
)

# 1. Page Configuration
st.set_page_config(
    page_title="Multi-Source Crop Optimization Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS for Premium Design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Outfit:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
    }
    
    /* Title Gradient */
    .title-text {
        background: linear-gradient(135deg, #10B981 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    
    .subtitle-text {
        font-size: 1.1rem;
        color: #6B7280;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Premium Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid #E5E7EB;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 1rem;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05), 0 4px 6px -2px rgba(0, 0, 0, 0.02);
    }
    
    /* Risk Badges */
    .risk-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .risk-low {
        background-color: #D1FAE5;
        color: #065F46;
        border: 1px solid #A7F3D0;
    }
    
    .risk-moderate {
        background-color: #FEF3C7;
        color: #92400E;
        border: 1px solid #FDE68A;
    }
    
    .risk-high {
        background-color: #FEE2E2;
        color: #991B1B;
        border: 1px solid #FCA5A5;
    }
    
    /* Custom divider */
    .custom-hr {
        margin: 1.5rem 0;
        border: 0;
        height: 1px;
        background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(229, 231, 235, 1), rgba(0, 0, 0, 0));
    }
    
    /* Sidebar styling */
    .css-1542z7x {
        background-color: #F9FAFB;
    }
</style>
""", unsafe_allow_html=True)

# 3. Load Datasets
@st.cache_data
def get_all_data():
    dist_df, history_df, econ_df = load_raw_data()
    preprocessed_dist_df = preprocess_soil_data(dist_df)
    integrated_history_df = get_integrated_dataset(history_df, dist_df)
    return preprocessed_dist_df, integrated_history_df, econ_df

dist_df, history_df, econ_df = get_all_data()

# 4. Sidebar Inputs
st.sidebar.markdown("### 🛠️ Farm Configurations")

# District Selection
district_list = sorted(dist_df['district'].tolist())
selected_district = st.sidebar.selectbox("Select District", district_list)
district_row = dist_df[dist_df['district'] == selected_district].iloc[0]
lat, lon = district_row['latitude'], district_row['longitude']

# Season Selection
selected_season = st.sidebar.selectbox("Select Season", ["Kharif", "Rabi", "Summer"])

# Land Area
farm_area = st.sidebar.number_input("Total Farm Area (ha)", min_value=0.5, max_value=100.0, value=5.0, step=0.5)

# Capital Budget
max_budget = 500000.0
capital_budget = st.sidebar.slider("Working Capital Budget (Rs)", min_value=10000, max_value=int(max_budget), value=150000, step=5000)

# Water Availability
water_level = st.sidebar.select_slider("Available Water Level", options=["Low", "Moderate", "High"], value="Moderate")

# Farmer Risk Profile
farmer_profile = st.sidebar.selectbox("Farmer Risk Profile", ["Balanced (default)", "Profit-first", "Risk-averse"])

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Coordinates:** {lat}°N, {lon}°E")
st.sidebar.markdown(f"**State:** {district_row['state']}")

# Trigger computation button
run_system = st.sidebar.button("🚀 Run Optimizer & Simulation", use_container_width=True)

# Main Title Header
st.markdown('<div class="title-text">Risk-Aware Crop Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Multi-Source Crop Optimization using Monte Carlo Simulation and Linear Programming</div>', unsafe_allow_html=True)

# 5. Core Execution Pipeline
if run_system or 'results' not in st.session_state:
    with st.spinner("Executing pipeline: Ingesting live weather, fitting Ridge models, and running simulation..."):
        
        # Step 1: Live weather API check
        weather_data = fetch_live_weather(lat, lon, selected_season)
        if weather_data:
            st.session_state['weather_status'] = ("live", weather_data)
        else:
            weather_data = get_district_weather_fallback(selected_district, selected_season, history_df)
            st.session_state['weather_status'] = ("fallback", weather_data)
            
        current_rf = weather_data['rainfall_mm']
        current_tmax = weather_data['tmax_c']
        
        # Step 2: Get candidate crops for this season
        season_econ_df = econ_df[econ_df['season'] == selected_season]
        available_crops = season_econ_df['crop'].tolist()
        
        # Step 3: Run models and Monte Carlo
        sim_results = {}
        fitted_models = {}
        crops_data_for_lp = []
        
        for crop in available_crops:
            # Fit model
            model = CropYieldModel(crop, selected_district, selected_season)
            model.fit(history_df)
            fitted_models[crop] = model
            
            # Run simulation (1000 trials)
            sim_out = run_monte_carlo_simulation(model, history_df, selected_district, selected_season, crop)
            sim_results[crop] = sim_out
            
            # Look up economics
            econ_row = season_econ_df[season_econ_df['crop'] == crop].iloc[0]
            
            crops_data_for_lp.append({
                'crop': crop,
                'mean_yield': sim_out['mean_yield'],
                'downside_yield': sim_out['downside_yield'],
                'cost': econ_row['cost_rs_ha'],
                'price': econ_row['price_rs_qtl'],
                'water_req': econ_row['water_req_mm']
            })
            
        # Step 4: Solve LP Optimizer
        lp_out = solve_crop_optimization(crops_data_for_lp, farm_area, capital_budget, water_level)
        lp_out_cons = solve_crop_optimization_conservative(crops_data_for_lp, farm_area, capital_budget, water_level)
        
        # Step 5: Setup candidate portfolios for RAS ranking
        # We define:
        # - LP-Optimal Portfolio
        # - Conservative Portfolio
        # - Equal Area (Baseline A)
        # - Single Best Crop (Baseline B)
        # - Focused Crop Portfolios: 60% Crop i, 40% second best crop.
        baselines = run_baseline_comparison(crops_data_for_lp, farm_area, capital_budget, lp_out['expected_profit'], lp_out['allocation'])
        
        portfolios = []
        # Add LP Optimal
        portfolios.append({
            'name': 'LP-Optimal',
            'allocation': lp_out['allocation'],
            **calculate_portfolio_metrics(lp_out['allocation'], sim_results, season_econ_df, farm_area)
        })
        # Add Conservative
        portfolios.append({
            'name': 'Conservative LP',
            'allocation': lp_out_cons['allocation'],
            **calculate_portfolio_metrics(lp_out_cons['allocation'], sim_results, season_econ_df, farm_area)
        })
        # Add Baseline A
        portfolios.append({
            'name': 'Baseline A (Equal-Area)',
            'allocation': baselines['Baseline_A']['allocation'],
            **calculate_portfolio_metrics(baselines['Baseline_A']['allocation'], sim_results, season_econ_df, farm_area)
        })
        # Add Baseline B
        portfolios.append({
            'name': 'Baseline B (Single Best-Crop)',
            'allocation': baselines['Baseline_B']['allocation'],
            **calculate_portfolio_metrics(baselines['Baseline_B']['allocation'], sim_results, season_econ_df, farm_area)
        })
        
        # Add Crop-focused portfolios
        sorted_crops = sorted(crops_data_for_lp, key=lambda x: x['mean_yield'], reverse=True)
        for c_data in crops_data_for_lp:
            c_name = c_data['crop']
            # Create a 60% focused portfolio
            alloc = {crop: 0.0 for crop in available_crops}
            if len(available_crops) > 1:
                second = sorted_crops[1]['crop'] if sorted_crops[0]['crop'] != c_name else sorted_crops[0]['crop']
                alloc[c_name] = 0.6 * farm_area
                alloc[second] = 0.4 * farm_area
            else:
                alloc[c_name] = farm_area
                
            # Scale down if budget exceeded
            total_cost_focused = sum(c_data_lookup['cost'] * alloc[crop] for crop in available_crops for c_data_lookup in crops_data_for_lp if c_data_lookup['crop'] == crop)
            if total_cost_focused > capital_budget:
                scale_factor = capital_budget / total_cost_focused
                alloc = {k: v * scale_factor for k, v in alloc.items()}
                
            portfolios.append({
                'name': f'{c_name}-Focused Mix',
                'allocation': alloc,
                **calculate_portfolio_metrics(alloc, sim_results, season_econ_df, farm_area)
            })
            
        # Remove duplicate portfolios by name
        seen_names = set()
        unique_portfolios = []
        for p in portfolios:
            if p['name'] not in seen_names:
                unique_portfolios.append(p)
                seen_names.add(p['name'])
                
        # Save to session state
        st.session_state['results'] = {
            'sim_results': sim_results,
            'fitted_models': fitted_models,
            'crops_data_for_lp': crops_data_for_lp,
            'lp_out': lp_out,
            'lp_out_cons': lp_out_cons,
            'baselines': baselines,
            'unique_portfolios': unique_portfolios,
            'current_rf': current_rf,
            'current_tmax': current_tmax,
            'available_crops': available_crops
        }

# Retrieve results from session state
results = st.session_state['results']
sim_results = results['sim_results']
fitted_models = results['fitted_models']
crops_data_for_lp = results['crops_data_for_lp']
lp_out = results['lp_out']
lp_out_cons = results['lp_out_cons']
baselines = results['baselines']
unique_portfolios = results['unique_portfolios']
current_rf = results['current_rf']
current_tmax = results['current_tmax']
available_crops = results['available_crops']

# Display Weather Status Bar
weather_status, w_val = st.session_state['weather_status']
if weather_status == "live":
    st.success(f"☀️ **Live Weather Ingestion Connected (Open-Meteo API):** Retreived coordinates {lat}°N, {lon}°E. Current season weather indicators: **Rainfall = {current_rf} mm**, **Mean Max Temp = {current_tmax}°C**.")
else:
    st.info(f"💾 **Local Agro-Meteorological Baseline Averages Loaded (Open-Meteo offline/fallback):** Current season weather indicators: **Rainfall = {current_rf} mm**, **Mean Max Temp = {current_tmax}°C**.")

# 6. Tab Design Layout
tab1, tab2, tab3, tab4 = st.tabs([
    "🌾 Recommended Crop Portfolios", 
    "📈 Economic Metrics & Baselines", 
    "🎲 Monte Carlo Distributions", 
    "🔬 Soil Health & EDA"
])

# Profile Weights Mapping
profile_weights = {
    "Balanced (default)": (0.5, 0.3, 0.2),
    "Profit-first": (0.6, 0.2, 0.2),
    "Risk-averse": (0.3, 0.5, 0.2)
}
w1, w2, w3 = profile_weights[farmer_profile]

# Rank portfolios for active profile
ranked_portfolios = compute_risk_adjusted_score(unique_portfolios, w1, w2, w3)

# ----------------- TAB 1: RECOMMENDATIONS -----------------
with tab1:
    col_l, col_r = st.columns([1, 1])
    
    with col_l:
        st.markdown("### 🏆 Top-Ranked Crop Portfolios")
        st.write("Portfolios are ranked using a **Risk-Adjusted Score (RAS)** which synthesises Expected Profit, Yield Stability (break-even probability), and ROI.")
        
        # Display top 3 portfolios
        for idx in range(min(3, len(ranked_portfolios))):
            p = ranked_portfolios[idx]
            
            # Risk color
            risk_class = "risk-low"
            if p['risk_band'] == "Moderate":
                risk_class = "risk-moderate"
            elif p['risk_band'] == "High":
                risk_class = "risk-high"
                
            st.markdown(f"""
            <div class="metric-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span style="font-size: 1.25rem; font-weight: 700; color: #1E293B;">Rank {idx+1}: {p['name']}</span>
                    <span class="risk-badge {risk_class}">{p['risk_band']} Risk</span>
                </div>
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1rem;">
                    <div>
                        <div style="font-size: 0.85rem; color: #6B7280; font-weight: 500;">Expected Profit</div>
                        <div style="font-size: 1.2rem; font-weight: 700; color: #10B981;">Rs. {p['expected_profit']:,.2f}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.85rem; color: #6B7280; font-weight: 500;">Expected ROI</div>
                        <div style="font-size: 1.2rem; font-weight: 700; color: #3B82F6;">{p['expected_roi']:.1f}%</div>
                    </div>
                    <div>
                        <div style="font-size: 0.85rem; color: #6B7280; font-weight: 500;">Break-Even Prob.</div>
                        <div style="font-size: 1.2rem; font-weight: 700; color: #F59E0B;">{p['break_even_prob']*100:.1f}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    with col_r:
        st.markdown("### 📊 Crop Land Allocation")
        
        # Prepare allocation chart data for Top-3 Portfolios, LP-Optimal, and Conservative
        alloc_records = []
        plot_portfolios = [ranked_portfolios[0], ranked_portfolios[1], ranked_portfolios[2]]
        
        # Add LP Optimal and Conservative to plot list if not already in top 2
        names_in_plot = [p['name'] for p in plot_portfolios]
        
        for p in ranked_portfolios:
            if p['name'] in ['LP-Optimal', 'Conservative LP'] and p['name'] not in names_in_plot:
                plot_portfolios.append(p)
                
        for p in plot_portfolios:
            for crop, area in p['allocation'].items():
                if area > 0:
                    alloc_records.append({
                        'Portfolio': p['name'],
                        'Crop': crop,
                        'Land (ha)': round(area, 2)
                    })
        
        df_plot = pd.DataFrame(alloc_records)
        
        # Draw Bar Chart using Matplotlib/Seaborn
        fig, ax = plt.subplots(figsize=(8, 5.5))
        sns.set_theme(style="whitegrid")
        
        # Custom color palette for crops
        crop_palette = sns.color_palette("muted", len(available_crops))
        crop_colors = dict(zip(available_crops, crop_palette))
        
        # Re-pivot to draw stacked bar
        pivot_df = df_plot.pivot(index='Portfolio', columns='Crop', values='Land (ha)').fillna(0)
        # Ensure all available crops are columns to match color mapping
        for crop in available_crops:
            if crop not in pivot_df.columns:
                pivot_df[crop] = 0.0
        
        pivot_df.plot(kind='bar', stacked=True, ax=ax, color=[crop_colors[c] for c in pivot_df.columns])
        
        ax.set_title("Hectares Allocated Per Crop", fontsize=12, fontweight='bold', pad=15)
        ax.set_ylabel("Hectares (ha)", fontsize=10)
        ax.set_xlabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right', fontsize=9)
        ax.legend(title="Crops", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)
        plt.tight_layout()
        st.pyplot(fig)
        
    st.markdown('<div class="custom-hr"></div>', unsafe_allow_html=True)
    
    # Farmer Advice Column
    st.markdown("### 🗣️ Advisor's Plain-Language Guidance")
    best_portfolio = ranked_portfolios[0]
    best_crops = [f"**{crop}** ({area:.2f} ha)" for crop, area in best_portfolio['allocation'].items() if area > 0.01]
    
    # Write a plain language advice block
    risk_profile_desc = {
        "Balanced (default)": "maintain a secure balance between maximizing income and mitigating risk during poor climate conditions.",
        "Profit-first": "push for maximum net income, accepting higher climate-related yield volatility.",
        "Risk-averse": "ensure maximum safety and yield stability, protecting against droughts or extreme temperatures."
    }
    
    advice_text = f"""
    Dear Farmer / Extension Officer,
    
    Based on the historical soil parameters in **{selected_district}** (pH: {district_row['pH']}, Texture: {district_row['soil_texture']}), 
    live weather inputs (Rainfall: {current_rf} mm, Temperature: {current_tmax}°C), and your preference to **{risk_profile_desc[farmer_profile]}**, 
    the system recommends implementing the **{best_portfolio['name']}** strategy.
    
    **Suggested Allocation:**
    * You should plant: {', '.join(best_crops)}.
    
    **Expectations:**
    * **Expected Net Profit:** Approximately **Rs. {best_portfolio['expected_profit']:,.2f}** for the total farm area ({farm_area} ha), with an expected **ROI of {best_portfolio['expected_roi']:.1f}%**.
    * **Downside Safe Return (10th percentile):** If weather conditions turn adverse (e.g. low rainfall/drought), your net profit is expected to remain above **Rs. {best_portfolio['downside_profit']:,.2f}**.
    * **Climate Risk Classification:** The recommendation is classified as **{best_portfolio['risk_band']} Risk** because of the stable historical yields and high break-even confidence (**{best_portfolio['break_even_prob']*100:.1f}%** chance of positive returns).
    """
    st.info(advice_text)


# ----------------- TAB 2: ECONOMIC METRICS & BASELINES -----------------
with tab2:
    st.markdown("### 📊 Performance Comparison against Baselines")
    st.write("We compare the mathematically optimal allocation (LP-Optimal) against two common baseline strategies to validate decision-support value:")
    st.write("* **Baseline A (Equal-Area):** Available land split equally across all crops suited for the season.")
    st.write("* **Baseline B (Single Best-Crop):** 60% of land (max permitted by diversity) given to the single crop with highest historical yield, 40% to the second best.")
    
    # Baseline comparison dataframe
    compare_records = []
    for b_name, b_data in baselines.items():
        # Get metrics
        if b_name == 'LP_Optimal':
            p_name = 'LP-Optimal'
        elif b_name == 'Baseline_A':
            p_name = 'Baseline A (Equal-Area)'
        else:
            p_name = 'Baseline B (Single Best-Crop)'
            
        p_metrics = [p for p in unique_portfolios if p['name'] == p_name][0]
        
        # Calculate percentage gains relative to Baseline A
        compare_records.append({
            'Strategy': p_name,
            'Expected Profit (Rs.)': f"Rs. {p_metrics['expected_profit']:,.2f}",
            'Total Cost (Rs.)': f"Rs. {p_metrics['total_cost']:,.2f}",
            'Expected ROI': f"{p_metrics['expected_roi']:.2f}%",
            'Break-Even Probability': f"{p_metrics['break_even_prob']*100:.1f}%",
            'Downside Profit (10th percentile)': f"Rs. {p_metrics['downside_profit']:,.2f}",
            'Risk Band': p_metrics['risk_band']
        })
        
    df_compare = pd.DataFrame(compare_records)
    st.dataframe(df_compare, use_container_width=True, hide_index=True)
    
    # Display gains summary
    profit_optimal = baselines['LP_Optimal']['expected_profit']
    profit_a = baselines['Baseline_A']['expected_profit']
    profit_b = baselines['Baseline_B']['expected_profit']
    
    gain_a = ((profit_optimal - profit_a) / profit_a * 100) if profit_a > 0 else 0.0
    gain_b = ((profit_optimal - profit_b) / profit_b * 100) if profit_b > 0 else 0.0
    
    st.markdown(f"""
    💡 **Optimization Performance Gain Analysis:**
    * LP-Optimal allocation improves expected profit by **{gain_a:.1f}%** compared to the naive Equal-Area strategy.
    * LP-Optimal allocation improves expected profit by **{gain_b:.1f}%** compared to the Single Best-Crop strategy.
    """)
    
    st.markdown('<div class="custom-hr"></div>', unsafe_allow_html=True)
    
    # Sensitivity & Shadow Prices
    col_shadow_l, col_shadow_r = st.columns([1, 1])
    with col_shadow_l:
        st.markdown("### ⛓️ Resource Shadow Price Sensitivity")
        st.write("Shadow prices represent the marginal change in Expected Profit for every additional unit of resource added (dual values):")
        
        shadow_prices = lp_out['shadow_prices']
        
        shadow_records = [
            {'Resource Constraint': 'Land Area (per additional Hectare)', 'Shadow Price': f"Rs. {shadow_prices['Land']:,.2f}"},
            {'Resource Constraint': 'Working Capital (per additional Rupee)', 'Shadow Price': f"Rs. {shadow_prices['Capital']:.4f}"},
            {'Resource Constraint': 'Irrigation Water (per additional Hectare-mm)', 'Shadow Price': f"Rs. {shadow_prices['Water']:,.2f}"}
        ]
        st.table(pd.DataFrame(shadow_records))
        
    with col_shadow_r:
        # Explain shadow price insights
        binding_resources = []
        for res, val in shadow_prices.items():
            if val > 1e-4:
                binding_resources.append(res)
                
        binding_text = "None"
        if binding_resources:
            binding_text = ", ".join(binding_resources)
            
        st.markdown("#### 💡 Decision-Support Insights:")
        st.markdown(f"**Binding Constraints identified:** `{binding_text}`")
        if 'Land' in binding_resources:
            st.markdown(f"* **Land is binding:** This farm is size-constrained. Acquiring more land or optimizing crop spacing will directly yield **Rs. {shadow_prices['Land']:,.2f}** more per hectare.")
        if 'Capital' in binding_resources:
            st.markdown(f"* **Capital is binding:** Working capital is limiting crop choices. Increasing the budget by just Rs 1.00 will return **Rs. {shadow_prices['Capital']:.2f}** in net profit (marginal rate of return of **{shadow_prices['Capital']*100:.1f}%**).")
        if 'Water' in binding_resources:
            st.markdown(f"* **Water is binding:** Drought conditions or restricted water access are holding back profit. Setting up rainwater harvesting or better irrigation will return **Rs. {shadow_prices['Water']:,.2f}** per hectare-mm.")
        if not binding_resources:
            st.markdown("* The current farm configuration is fully optimized and has surplus resources. Neither land, capital, nor water constraints are fully binding at this moment.")


# ----------------- TAB 3: MONTE CARLO DISTRIBUTIONS -----------------
with tab3:
    st.markdown("### 🎲 Monte Carlo Yield & Profit Simulations")
    st.write("We run 1,000 simulated weather seasons (Multivariate Normal sampling of Rainfall & Tmax) and add Ridge regression residuals to model yield uncertainty:")
    
    # Dropdown to choose crop yield distribution
    selected_crop_plot = st.selectbox("Select Crop to view Yield Distribution", available_crops)
    
    col_plot_l, col_plot_r = st.columns([1.1, 1])
    
    with col_plot_l:
        # Plot Crop Yield Histogram
        fig_mc, ax_mc = plt.subplots(figsize=(7, 4.5))
        sim_data = sim_results[selected_crop_plot]
        
        # Draw histogram and KDE
        sns.histplot(sim_data['sim_yields'], kde=True, color='#10B981', ax=ax_mc, bins=30, edgecolor='white')
        
        # Add percentile lines
        ax_mc.axvline(sim_data['mean_yield'], color='blue', linestyle='--', label=f"Mean Yield: {sim_data['mean_yield']:.2f} q/ha")
        ax_mc.axvline(sim_data['downside_yield'], color='red', linestyle=':', label=f"10th Pct (Downside): {sim_data['downside_yield']:.2f} q/ha")
        ax_mc.axvline(sim_data['upside_yield'], color='green', linestyle=':', label=f"90th Pct (Upside): {sim_data['upside_yield']:.2f} q/ha")
        
        # Labels and legend
        ax_mc.set_title(f"1,000-Trial Yield Distribution: {selected_crop_plot}", fontsize=11, fontweight='bold')
        ax_mc.set_xlabel("Yield (q/ha)", fontsize=9)
        ax_mc.set_ylabel("Trial Count", fontsize=9)
        ax_mc.legend(fontsize=8, loc='upper right')
        plt.tight_layout()
        st.pyplot(fig_mc)
        
    with col_plot_r:
        # Summary statistics table of simulation results
        st.markdown(f"#### 📊 Simulated Statistics for {selected_crop_plot}")
        
        stats_df = pd.DataFrame([
            {'Metric': 'Historically Predicted Mean Yield', 'Value': f"{sim_data['mean_yield']:.2f} q/ha"},
            {'Metric': 'Yield Standard Deviation (Volatility)', 'Value': f"{sim_data['yield_std']:.2f} q/ha"},
            {'Metric': 'Yield Variance (Variance)', 'Value': f"{sim_data['yield_variance']:.2f} (q/ha)²"},
            {'Metric': 'Downside Yield (10% weather safety)', 'Value': f"{sim_data['downside_yield']:.2f} q/ha"},
            {'Metric': 'Upside Yield (90% weather favorability)', 'Value': f"{sim_data['upside_yield']:.2f} q/ha"},
        ])
        st.table(stats_df)
        
        # Show regression validation metrics
        model = fitted_models[selected_crop_plot]
        st.markdown("#### 🔬 Ridge vs. OLS Validation Comparison:")
        metrics_df = pd.DataFrame({
            'Validation Metric': ['Out-of-sample R² Score', 'Root Mean Squared Error (RMSE)', 'Mean Absolute Error (MAE)'],
            'RidgeCV (Proposed)': [f"{model.r2:.3f}", f"{model.rmse:.3f} q/ha", f"{model.mae:.3f} q/ha"],
            'OLS Baseline': [f"{model.ols_r2:.3f}", f"{model.ols_rmse:.3f} q/ha", f"{model.ols_mae:.3f} q/ha"]
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        # Display Ridge alpha and fitted equation
        st.caption(f"ℹ️ Ridge Regularization Strength Selected (λ / alpha): **{model.alpha_:.4f}**")
        coef_str = f"Y = {model.intercept:.2f} + ({model.coef[0]:.3f}*RF) + ({model.coef[1]:.3f}*Tmax)"
        if len(model.coef) > 2:
            coef_str += f" + ({model.coef[2]:.3f}*pH) + ({model.coef[6]:.3f}*SQI) + ..."
        st.caption(f"Fitted Ridge Equation: `{coef_str}`")
        
    st.markdown('<div class="custom-hr"></div>', unsafe_allow_html=True)
    
    # Portfolio Sensitivity Matrix
    st.markdown("### ⚡ Farmer Risk Profile Sensitivity Heatmap")
    st.write("We evaluate how the ranking of candidate portfolios changes as we alter the weights of the Risk-Adjusted Score (RAS) across different farmer profiles:")
    
    sens_df = run_ras_sensitivity_analysis(unique_portfolios)
    
    # Pivot to show rankings
    pivot_sens = sens_df.pivot(index='Portfolio_Name', columns='Profile', values='Rank')
    # Order profiles
    pivot_sens = pivot_sens[['Profit-first', 'Balanced', 'Risk-averse']]
    
    fig_heat, ax_heat = plt.subplots(figsize=(8, 4))
    sns.heatmap(pivot_sens, annot=True, cmap="YlGnBu_r", cbar=False, fmt='g', linewidths=.5, ax=ax_heat)
    ax_heat.set_title("Portfolio Rank across Farmer Risk Profiles (Rank 1 is Best)", fontsize=11, fontweight='bold', pad=10)
    ax_heat.set_ylabel("")
    ax_heat.set_xlabel("Farmer Profile Weights", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig_heat)


# ----------------- TAB 4: SOIL HEALTH & EDA -----------------
with tab4:
    st.markdown("### 🔬 Soil Health Parameter Metrics")
    st.write("Soil Health Card Portal records for the selected district (including the engineered Soil Quality Index):")
    
    # Metric cards for soil (6 columns)
    col_soil1, col_soil2, col_soil3, col_soil4, col_soil5, col_soil6 = st.columns(6)
    with col_soil1:
        st.metric("Soil pH (Acidity)", f"{district_row['pH']}", help="Ideal pH range is usually 6.0 to 7.5.")
    with col_soil2:
        st.metric("Organic Carbon (OC)", f"{district_row['organic_carbon']}%", help="Higher organic carbon indicates richer soil.")
    with col_soil3:
        st.metric("Phosphorus (P)", f"{district_row['phosphorus']} kg/ha")
    with col_soil4:
        st.metric("Potassium (K)", f"{district_row['potassium']} kg/ha")
    with col_soil5:
        st.metric("Soil Texture Class", f"{district_row['soil_texture']}")
    with col_soil6:
        st.metric("Soil Quality Index (SQI)", f"{district_row['soil_quality_index']}/10", help="Engineered index combining OC, P, and K.")
        
    st.markdown('<div class="custom-hr"></div>', unsafe_allow_html=True)
    
    # IQR price outlier detection alert
    outliers, lower_b, upper_b = detect_price_outliers_iqr(econ_df)
    if outliers:
        st.warning(f"⚠️ **Commodity Price Outlier Analysis (IQR Method):** Outlier prices detected outside the typical range [Rs. {lower_b:,.2f} - Rs. {upper_b:,.2f}] per quintal:")
        outlier_text = []
        for o in outliers:
            outlier_text.append(f"**{o['crop']}** ({o['season']}): Rs. {o['price']:,.2f}")
        st.markdown("  •  " + "  |  ".join(outlier_text))
    else:
        st.success(f"✅ **Commodity Price Outlier Analysis (IQR Method):** All commodity prices fall within the standard range [Rs. {lower_b:,.2f} - Rs. {upper_b:,.2f}] per quintal (no statistical outliers).")

    st.markdown('<div class="custom-hr"></div>', unsafe_allow_html=True)

    # Historical summary statistics table
    st.markdown("### 📊 Historical Dataset Summary Statistics")
    st.write("Exploratory summary metrics of the integrated dataset (combining yield records, soil health, and weather variables):")
    
    summary_cols = {
        'rainfall_mm': 'Seasonal Rainfall (mm)',
        'tmax_c': 'Mean Max Temp (°C)',
        'yield_q_ha': 'Crop Yield (q/ha)',
        'pH': 'Soil pH',
        'organic_carbon': 'Organic Carbon (%)',
        'soil_quality_index': 'Soil Quality Index'
    }
    
    stats_records = []
    for col, name in summary_cols.items():
        if col in history_df.columns:
            series = history_df[col]
            stats_records.append({
                'Variable': name,
                'Mean': f"{series.mean():.2f}",
                'Median': f"{series.median():.2f}",
                'Std Dev': f"{series.std():.2f}",
                'Skewness': f"{series.skew():.2f}",
                'Kurtosis': f"{series.kurtosis():.2f}"
            })
            
    df_stats = pd.DataFrame(stats_records)
    st.dataframe(df_stats, use_container_width=True, hide_index=True)
    
    st.markdown('<div class="custom-hr"></div>', unsafe_allow_html=True)
    
    col_eda_l, col_eda_r = st.columns([1, 1])
    
    with col_eda_l:
        st.markdown("### 🌦️ Weather-Yield Historical Correlation")
        st.write("Correlation analysis showing how weather variables (seasonal Rainfall and max Temperature) correlate with yields for all candidate crops:")
        
        # Extract subset of history for correlation
        corr_subset = history_df[
            (history_df['district'].str.lower() == selected_district.lower()) &
            (history_df['season'].str.lower() == selected_season.lower())
        ]
        
        if corr_subset.empty:
            st.warning("No local data found for correlation plot.")
        else:
            # We pivot the dataframe to get crops as columns
            pivot_corr = corr_subset.pivot_table(index='year', columns='crop', values='yield_q_ha')
            # Add rainfall and tmax
            weather_subset = corr_subset.groupby('year')[['rainfall_mm', 'tmax_c']].mean()
            combined_corr = pd.concat([pivot_corr, weather_subset], axis=1)
            
            # Compute correlation matrix
            corr_mat = combined_corr.corr()
            
            fig_corr, ax_corr = plt.subplots(figsize=(7, 5))
            sns.heatmap(corr_mat[['rainfall_mm', 'tmax_c']].drop(['rainfall_mm', 'tmax_c']), annot=True, cmap="coolwarm", vmin=-1.0, vmax=1.0, ax=ax_corr)
            ax_corr.set_title("Correlation of Crop Yield with Weather Parameters", fontsize=10, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig_corr)
            
    with col_eda_r:
        st.markdown("### 📈 Historical Yield Trends (1990-2024)")
        st.write("Temporal trend analysis of yields per crop in the selected district:")
        
        fig_trend, ax_trend = plt.subplots(figsize=(7, 5))
        
        for crop in available_crops:
            crop_trend = corr_subset[corr_subset['crop'] == crop].sort_values('year')
            if not crop_trend.empty:
                ax_trend.plot(crop_trend['year'], crop_trend['yield_q_ha'], marker='o', label=crop, linewidth=1.5, markersize=4)
                
        ax_trend.set_title("Crop Yield Timeline (quintals per hectare)", fontsize=10, fontweight='bold')
        ax_trend.set_xlabel("Year", fontsize=9)
        ax_trend.set_ylabel("Yield (q/ha)", fontsize=9)
        ax_trend.legend(fontsize=8, loc='upper left', frameon=True)
        plt.tight_layout()
        st.pyplot(fig_trend)
