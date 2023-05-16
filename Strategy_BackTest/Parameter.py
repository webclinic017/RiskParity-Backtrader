import numpy as np
from OptimizerBackTest import *

# Define range and step size for Max_weight and N_More
max_weight_range = np.arange(0.1, 1, 0.1)
n_more_range = np.arange(1, 10)

# Set the number of iterations
num_iterations = 100

# Initialize variables to store results
results = []

# Monte Carlo simulation
for _ in range(num_iterations):
    # Generate random values for Max_weight and N_More
    max_weight = np.random.choice(max_weight_range)
    n_more = np.random.choice(n_more_range)
    
    # Call optimizerbacktest function with the generated values
    portfolio_return_concat, weight_concat, sharpe_array_concat = optimizerbacktest(monthly_returns_log, dummy_L_df, daily_returns_log)
    
    # Calculate performance metric (e.g., Sharpe ratio) and store the results
    #sharpe_ratio = old_sharpe(portfolio_return_concat)
    results.append({'Max_weight': max_weight, 'N_More': n_more, 'Sharpe_ratio': sharpe_ratio})

# Analyze the results
best_result = max(results, key=lambda x: x['Sharpe_ratio'])
optimal_max_weight = best_result['Max_weight']
optimal_n_more = best_result['N_More']

print(f"Optimal Max_weight: {optimal_max_weight}")
print(f"Optimal N_More: {optimal_n_more}")