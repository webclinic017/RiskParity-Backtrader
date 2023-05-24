
import numpy as np
from OptimizerBackTest import *
from datamanagement import *

monthly_returns, asset_classes, asset, assets = data_management(Start, End, '1mo')

monthly_returns = monthly_returns.dropna()

monthly_returns_log = ret(monthly_returns)
daily_returns_log   = ret(daily_returns)


# Define range and step size for Max_weight and N_More
max_weight_range = np.arange(0, 1, 0.1).tolist()
n_more_range = np.arange(1, 10, 1).tolist()

# Set the number of iterations
num_iterations = 100

# Initialize variables to store results
results = []
print("Starting backtest")
# Monte Carlo simulation

for nmore in  n_more_range:
    for mweight in max_weight_range:
        # Generate random values for Max_weight and N_More
        benchmark = 'Benchmark'
        # Call optimizerbacktest function with the generated values
        portfolio_return_concat, weight_concat, sharpe_array_concat = optimizerbacktest(daily_returns_log, dummy_L_df, daily_returns_log, nmore, mweight, monthly_returns_log)
        Bench, Merged_df = bench(portfolio_return_concat.index.min(), benchmark, portfolio_return_concat)

        portfolio_return = (Merged_df.iloc[-1]-10000)/10000  # Get the final portfolio return
        results.append({'Max_weight': nmore, 'N_More': mweight, 'Portfolio_return': portfolio_return.values})
print("we at")
print(results)
# Analyze the results
best_result = max(results, key=lambda x: x['Portfolio_return'])
optimal_max_weight = best_result['Max_weight']
optimal_n_more = best_result['N_More']

print(f"Optimal Max_weight: {optimal_max_weight}")
print(f"Optimal N_More: {optimal_n_more}")
