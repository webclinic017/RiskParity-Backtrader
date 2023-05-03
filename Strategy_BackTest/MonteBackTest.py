import pandas as pd
import numpy as np 
import warnings
import dash_html_components as html
from Utils import *
from Trend_Following import *
warnings.filterwarnings("ignore")
from OptimizerBackTest import *

#setup (1 = True):
ls        = 1
monte     = 1
trend     = 'sma'
Rf        = 0.04
benchmark = ['VTI','BND']
Scalar    = 500
Dist      = 'direchlit' #'standard_t'

# Setup dates


# Setup DFs
merged_df = sharpe_array = df_dummy_sum = df_dummy_sum = this_month_weight = pd.DataFrame([])

# Monte Carlo
def monte_carlo(Y):
    log_return  = np.log(Y/Y.shift(1))
    sample      = Y.shape[0]
    num_ports   = 10 # number_of_iter * Scalar 
    all_weights = np.zeros((num_ports, len(Y.columns)))
    ret_arr     = np.zeros(num_ports)
    vol_arr     = np.zeros(num_ports)
    sharpe_arr  = np.zeros(num_ports)
    num_assets = len(Y.columns)
    all_weights = np.zeros((num_ports, num_assets))
    df = 10

    for ind in range(num_ports): 
        # weights 
        if Dist == 'standard_t':
            weights = np.random.standard_t(df, size=num_assets)
        else:
            weights = np.random.dirichlet(np.ones(len(Y.columns)), size=1)
        #Rules
        weights[weights < 0.05] = 0
        weights[weights > 0.6] = 0.6

        weights = np.squeeze(weights)

        weights = weights/np.sum(weights)

        all_weights[ind,:] = weights
        
        # expected return 
        ret_arr[ind] = np.sum((log_return.mean()*weights)*sample)

        # expected volatility 
        vol_arr[ind] = np.sqrt(np.dot(weights.T,np.dot(log_return.cov()*sample, weights)))

        # Sharpe Ratio 
        sharpe_arr[ind] = (ret_arr[ind] - (Rf/12))/vol_arr[ind]
    max_sh = sharpe_arr.argmax()
    sharpe_ratio = (ret_arr[max_sh]- (Rf/12))/vol_arr[max_sh]
    return all_weights[max_sh,:], sharpe_ratio, vol_arr,ret_arr,sharpe_arr

# Calculate sharpe for next month
def next_sharpe(weights, log_return, sharpe_list):
    sample = log_return.shape[0]
    ret_arr2 = np.sum((log_return.mean()*weights)*sample)
    # expected volatility 
    vol_arr2 = np.sqrt(np.dot(weights.T,np.dot(log_return.cov()*sample, weights)))
    sharpe_arr2 = ret_arr2/vol_arr2
    return sharpe_arr2

def forfrontier(arr, i):
    arr = pd.DataFrame(arr).T
    arr['index'] = i
    arr.set_index('index', inplace=True)
    return arr

def monte_backtest(Y_adjusted, i, b, trend_df, sharpe_array_concat, ret_pct, weight_concat):
    w, sharpe_ratio, vol_arr,ret_arr,sharpe_arr = monte_carlo(Y_adjusted) #Long
    next_i,next_b = next_month(i)
    weight_concat, w_df = weightings(w, Y_adjusted, next_i, weight_concat, sharpe_array_concat, sharpe_ratio)
    y_next = ret_pct[next_i:next_b]
    Y_adjusted_next_L = asset_trimmer(b, trend_df, y_next) #Long
    portfolio_return = portfolio_returns(w_df, Y_adjusted_next_L, i, w) #Long
    vol_arr = forfrontier(vol_arr, i)
    ret_arr = forfrontier(ret_arr, i)
    sharpe_arr = forfrontier(sharpe_arr, i)
    return portfolio_return, vol_arr, ret_arr, sharpe_arr, weight_concat

def concat(vol_arr, vol_arr_concat, ret_arr, ret_arr_concat, sharpe_arr, sharpe_arr_concat, i, b):
    vol_arr_concat = pd.concat([vol_arr, vol_arr_concat], axis=0)
    ret_arr_concat = pd.concat([ret_arr, ret_arr_concat], axis=0)
    sharpe_arr_concat = pd.concat([sharpe_arr, sharpe_arr_concat], axis=0)
    prev_i = i
    prev_b = b
    return vol_arr_concat, ret_arr_concat, sharpe_arr_concat, prev_i, prev_b

# Backtesting
def backtest(rng_start, ret, ret_pct, trend_df):
    print("Iterating: ", number_of_iter * Scalar)
    vol_arr = vol_arr_concat = ret_arr_concat = sharpe_arr_concat = ret_arr = sharpe_arr = y_next = portfolio_return_concat = portfolio_return = weight_concat = sharpe_array_concat = pd.DataFrame([])
    for i in rng_start:
        rng_end = pd.date_range(i, periods=1, freq='M')
        for b in rng_end:
            # cleanup here
            if rng_start[-1] == i and prev_i is not None and prev_b is not None:
                print(f"Last month {i}")
                Y = ret[prev_i:prev_b]
                w, sharpe_ratio, vol_arr,ret_arr,sharpe_arr = monte_carlo(Y_adjusted) #Long
            else:
                Y = ret[i:b]
                Y_adjusted = asset_trimmer(b, trend_df, Y)
                if not Y_adjusted.empty:
                    portfolio_return, vol_arr, ret_arr, sharpe_arr, weight_concat = monte_backtest(Y_adjusted, i, b, trend_df, sharpe_array_concat, ret_pct, weight_concat)

                vol_arr_concat, ret_arr_concat, sharpe_arr_concat, prev_i, prev_b = concat(vol_arr, vol_arr_concat, ret_arr, ret_arr_concat, sharpe_arr, sharpe_arr_concat, i, b)
                portfolio_return_concat = pd.concat([portfolio_return_concat, portfolio_return], axis=0) #Long
    portfolio_return_concat = pd.DataFrame(portfolio_return_concat)
    portfolio_return_concat.index = pd.to_datetime(portfolio_return_concat.index)

    return portfolio_return_concat, weight_concat, vol_arr_concat,ret_arr_concat,sharpe_arr_concat

# Calling my functions
if   trend == 'rsi':
    rolling_long_df = rsi_df
elif trend == 'sma':
    rolling_long_df = dummy_L_df

portfolio_return_concat, weight_concat, vol_arr, ret_arr, sharpe_arr  = backtest(rng_start, ret, ret.pct_change(), rolling_long_df)

'''
Next steps:

-Actually get the optimization working, the montecarlo clearly isn't right


-More assets enabled.
-More asset selection culling.
-Incorporate the capm model for each assets expected returns.

-Incorporate a rally pivot concept, whereby it will pivot out of an asset if it its recent prices of the month are poor, e.g., a pivot.
-It is a bubble indicator.

New project:
-For each month, rate us on how well we selected assets based on the next months weightings, if the weightings are within a bounds then we are ok, if they are 
    below the previous month then take note that we were in too deep with this asset class, so next time we think of re-balancing by increasing this asset, we can essentially rate our scores.

    '''