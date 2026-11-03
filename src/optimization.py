import pulp
import pandas as pd
import numpy as np

def solve_crop_optimization(crops_data, total_area, capital_budget, water_availability_level):
    """
    Formulates and solves the crop allocation problem using PuLP with CBC.
    Parameters:
    - crops_data: dict of crop metrics, each having:
        { 'crop': str, 'mean_yield': float, 'downside_yield': float, 'cost': float, 'price': float, 'water_req': float }
    - total_area: float (hectares)
    - capital_budget: float (Rs)
    - water_availability_level: str ('Low', 'Moderate', 'High')
    
    Returns:
    - dict containing: status, allocation (dict), expected_profit, expected_cost, expected_water, shadow_prices
    """
    # 1. Calculate available water based on slider
    avg_water_req = np.mean([c['water_req'] for c in crops_data])
    water_multipliers = {'Low': 0.4, 'Moderate': 0.8, 'High': 1.2}
    available_water = water_multipliers[water_availability_level] * avg_water_req * total_area

    # Create PuLP problem
    prob = pulp.LpProblem("Crop_Profit_Maximization", pulp.LpMaximize)

    # Decision variables: area allocated to each crop (x_i >= 0)
    crop_names = [c['crop'] for c in crops_data]
    x = pulp.LpVariable.dicts("area", crop_names, lowBound=0, cat='Continuous')

    # Objective function: Maximize Expected Net Profit
    # Z = Sum_i (mean_yield_i * price_i - cost_i) * x_i
    net_returns = {}
    for c in crops_data:
        # Profit per hectare = Yield (q/ha) * Price (Rs/q) - Cost (Rs/ha)
        net_returns[c['crop']] = c['mean_yield'] * c['price'] - c['cost']
        
    prob += pulp.lpSum([net_returns[name] * x[name] for name in crop_names])

    # Constraints
    # 1. Land constraint: Sum_i x_i <= total_area
    land_constraint = pulp.lpSum([x[name] for name in crop_names]) <= total_area
    prob += land_constraint, "Land_Constraint"

    # 2. Capital constraint: Sum_i cost_i * x_i <= capital_budget
    costs = {c['crop']: c['cost'] for c in crops_data}
    capital_constraint = pulp.lpSum([costs[name] * x[name] for name in crop_names]) <= capital_budget
    prob += capital_constraint, "Capital_Constraint"

    # 3. Water constraint: Sum_i water_req_i * x_i <= available_water
    water_reqs = {c['crop']: c['water_req'] for c in crops_data}
    water_constraint = pulp.lpSum([water_reqs[name] * x[name] for name in crop_names]) <= available_water
    prob += water_constraint, "Water_Constraint"

    # 4. Diversity constraint: x_i <= 0.6 * total_area for all i
    for name in crop_names:
        prob += x[name] <= 0.6 * total_area, f"Diversity_Constraint_{name}"

    # Solve the problem
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # Extract results
    allocation = {name: x[name].varValue if x[name].varValue is not None else 0.0 for name in crop_names}
    
    # Calculate summary metrics
    expected_profit = sum(net_returns[name] * allocation[name] for name in crop_names)
    expected_cost = sum(costs[name] * allocation[name] for name in crop_names)
    expected_water = sum(water_reqs[name] * allocation[name] for name in crop_names)

    # Shadow prices (Dual values)
    # Note: shadow price represents change in objective value per unit change in constraint limit
    shadow_prices = {
        'Land': prob.constraints['Land_Constraint'].pi if prob.constraints['Land_Constraint'].pi is not None else 0.0,
        'Capital': prob.constraints['Capital_Constraint'].pi if prob.constraints['Capital_Constraint'].pi is not None else 0.0,
        'Water': prob.constraints['Water_Constraint'].pi if prob.constraints['Water_Constraint'].pi is not None else 0.0,
    }

    return {
        'status': pulp.LpStatus[status],
        'allocation': allocation,
        'expected_profit': round(expected_profit, 2),
        'expected_cost': round(expected_cost, 2),
        'expected_water': round(expected_water, 2),
        'shadow_prices': shadow_prices,
        'available_water': round(available_water, 2),
        'net_returns_per_ha': net_returns
    }

def solve_crop_optimization_conservative(crops_data, total_area, capital_budget, water_availability_level):
    """
    Solves the crop allocation problem using 10th-percentile (downside) yields for conservative planning.
    """
    # Calculate available water based on slider
    avg_water_req = np.mean([c['water_req'] for c in crops_data])
    water_multipliers = {'Low': 0.4, 'Moderate': 0.8, 'High': 1.2}
    available_water = water_multipliers[water_availability_level] * avg_water_req * total_area

    prob = pulp.LpProblem("Crop_Profit_Maximization_Conservative", pulp.LpMaximize)

    crop_names = [c['crop'] for c in crops_data]
    x = pulp.LpVariable.dicts("area", crop_names, lowBound=0, cat='Continuous')

    # Objective function: Maximize Conservative expected net profit
    # Z_cons = Sum_i (downside_yield_i * price_i - cost_i) * x_i
    net_returns_cons = {}
    for c in crops_data:
        net_returns_cons[c['crop']] = c['downside_yield'] * c['price'] - c['cost']
        
    prob += pulp.lpSum([net_returns_cons[name] * x[name] for name in crop_names])

    # Constraints
    prob += pulp.lpSum([x[name] for name in crop_names]) <= total_area, "Land_Constraint"
    
    costs = {c['crop']: c['cost'] for c in crops_data}
    prob += pulp.lpSum([costs[name] * x[name] for name in crop_names]) <= capital_budget, "Capital_Constraint"
    
    water_reqs = {c['crop']: c['water_req'] for c in crops_data}
    prob += pulp.lpSum([water_reqs[name] * x[name] for name in crop_names]) <= available_water, "Water_Constraint"
    
    for name in crop_names:
        prob += x[name] <= 0.6 * total_area, f"Diversity_Constraint_{name}"

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    allocation = {name: x[name].varValue if x[name].varValue is not None else 0.0 for name in crop_names}
    
    # Calculate metrics (evaluated using normal mean yields for fair economic comparison)
    mean_net_returns = {c['crop']: c['mean_yield'] * c['price'] - c['cost'] for c in crops_data}
    expected_profit = sum(mean_net_returns[name] * allocation[name] for name in crop_names)
    expected_cost = sum(costs[name] * allocation[name] for name in crop_names)
    expected_water = sum(water_reqs[name] * allocation[name] for name in crop_names)

    return {
        'status': pulp.LpStatus[status],
        'allocation': allocation,
        'expected_profit': round(expected_profit, 2),
        'expected_cost': round(expected_cost, 2),
        'expected_water': round(expected_water, 2),
    }

def run_baseline_comparison(crops_data, total_area, capital_budget, lp_profit, lp_allocation):
    """
    Computes returns for two baseline strategies:
    - Baseline A (Equal-Area): Land split equally across all candidate crops.
    - Baseline B (Single Best-Crop): Allocates maximum possible area (60% due to diversity) to the 
      highest mean-yield crop, and the remainder (40%) to the second highest mean-yield crop.
      If capital budget is exceeded, allocations are scaled down to fit budget.
    """
    crop_names = [c['crop'] for c in crops_data]
    n = len(crop_names)
    
    # 1. Baseline A: Equal Area Allocation
    base_a_alloc = {name: total_area / n for name in crop_names}
    
    # Check if budget constraints require scaling down equal area
    cost_per_ha = {c['crop']: c['cost'] for c in crops_data}
    total_cost_a = sum(cost_per_ha[name] * base_a_alloc[name] for name in crop_names)
    if total_cost_a > capital_budget:
        scale_factor = capital_budget / total_cost_a
        base_a_alloc = {name: base_a_alloc[name] * scale_factor for name in crop_names}
        
    net_returns = {c['crop']: c['mean_yield'] * c['price'] - c['cost'] for c in crops_data}
    profit_a = sum(net_returns[name] * base_a_alloc[name] for name in crop_names)

    # 2. Baseline B: Single Best-Crop (subject to diversity constraint)
    # Rank crops by historical mean yield
    sorted_by_yield = sorted(crops_data, key=lambda x: x['mean_yield'], reverse=True)
    best_crop = sorted_by_yield[0]['crop']
    second_best = sorted_by_yield[1]['crop'] if len(sorted_by_yield) > 1 else best_crop
    
    base_b_alloc = {name: 0.0 for name in crop_names}
    if best_crop == second_best:
        base_b_alloc[best_crop] = total_area
    else:
        # Maximize best crop up to 60%, and remainder 40% to second best
        base_b_alloc[best_crop] = 0.6 * total_area
        base_b_alloc[second_best] = 0.4 * total_area
        
    # Scale down if budget exceeded
    total_cost_b = sum(cost_per_ha[name] * base_b_alloc[name] for name in crop_names)
    if total_cost_b > capital_budget:
        scale_factor = capital_budget / total_cost_b
        base_b_alloc = {name: base_b_alloc[name] * scale_factor for name in crop_names}
        
    profit_b = sum(net_returns[name] * base_b_alloc[name] for name in crop_names)

    return {
        'Baseline_A': {
            'allocation': base_a_alloc,
            'expected_profit': round(profit_a, 2),
            'cost': round(sum(cost_per_ha[name] * base_a_alloc[name] for name in crop_names), 2)
        },
        'Baseline_B': {
            'allocation': base_b_alloc,
            'expected_profit': round(profit_b, 2),
            'cost': round(sum(cost_per_ha[name] * base_b_alloc[name] for name in crop_names), 2)
        },
        'LP_Optimal': {
            'allocation': lp_allocation,
            'expected_profit': round(lp_profit, 2),
            'cost': round(sum(cost_per_ha[name] * lp_allocation[name] for name in crop_names), 2)
        }
    }
