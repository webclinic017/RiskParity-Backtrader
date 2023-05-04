import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import minimize, Bounds, LinearConstraint
from Utils import *
import warnings
from dateutil.relativedelta import relativedelta
from datamanagement import *
from Trend_Following import dummy_L_df
warnings.filterwarnings("ignore")

#from Trend_Following import * #Start, End, ret, dummy_L_df, months_between, next_month

monthly_returns, asset_classes, asset = data_management(Start, End, '1mo')
daily_returns, asset_classes, asset = data_management(Start, End, '1d')

monthly_returns = monthly_returns.dropna()

def optimize(func, W, exp_ret, cov, target_return):
    opt_bounds = Bounds(0, 1)
    opt_constraints = ({'type': 'eq',
                        'fun': lambda W: 1.0 - np.sum(W)})
    
    optimal_weights = minimize(func, W, 
                               args=(exp_ret, cov),
                               method='SLSQP',
                               bounds=opt_bounds,
                               constraints=opt_constraints,
                               options={'maxiter': 100000000, 'ftol': 1e-3})
    print(optimal_weights['message'])
    return optimal_weights

def ret_risk(W, exp_ret, cov):
    print(W)
    print(type(exp_ret.values))
    ret     = W@exp_ret.values
    risk    = W.T@cov@W
    return -(ret/np.sqrt(risk))

def optimize_portfolio(returns_data):
    # Number of assets in the portfolio
    num_assets = len(returns_data.columns)

    # Calculate mean daily returns
    mean_returns = returns_data.mean()

    # Calculate covariance matrix
    cov_matrix = returns_data.cov()

    # Set random seed for reproducibility
    np.random.seed(0)

    # Define optimization function
    def portfolio_variance(weights):
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        return portfolio_variance

    # Define optimization constraints
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Constraint: weights sum to 1
    )

    # Define optimization bounds
    bounds = tuple((0, 1) for _ in range(num_assets))  # Bounds: 0 <= weights <= 1

    # Initial guess for weights
    initial_weights = np.ones(num_assets) / num_assets

    # Perform portfolio optimization
    result = minimize(portfolio_variance, initial_weights, method='SLSQP', constraints=constraints, bounds=bounds)

    # Extract optimized weights
    optimized_weights = result.x

    return optimized_weights


def ret(monthly_returns):
    monthly_returns_log = monthly_returns.pct_change() #np.log(monthly_returns/monthly_returns.shift(1))
    monthly_returns_log = monthly_returns_log.dropna()
    monthly_returns_log.index = pd.to_datetime(monthly_returns_log.index).strftime('%Y-%m-%d')
    return monthly_returns_log

monthly_returns_log = ret(monthly_returns)
daily_returns_log   = ret(daily_returns)

def asset_trimmer(b, df_split_monthly, Y):
    cols_to_drop = [col for col in df_split_monthly.columns if df_split_monthly[col].max() < 0.8]
    Y = Y.drop(columns=cols_to_drop)
    return Y

def optimizerbacktest(Y_adjusted, trend_df, daily_returns_log):
    weight_concat = sharpe_array_concat = portfolio_return_concat = pd.DataFrame()
    stopper = len(monthly_returns_log)
    monthly_returns_log.index = pd.to_datetime(monthly_returns_log.index)  # Convert index to datetime
    daily_returns_log.index = pd.to_datetime(daily_returns_log.index)

    for row_number, (index, row) in enumerate(monthly_returns_log.iterrows(), 1):

        current_month = index.month
        next_month = current_month + 1

        portfolio_return = month_returns_log = trend_df_2 = Y_adjusted = current_month_returns = pd.DataFrame()

        current_month_returns = daily_returns_log[daily_returns_log.index.month == current_month]
        next_month_returns = daily_returns_log[daily_returns_log.index.month == next_month]

        if row_number != stopper:
            trend_df_2 = pd.DataFrame(trend_df.iloc[row_number+1])
            Y_adjusted = asset_trimmer(row_number+1, trend_df_2.T, current_month_returns)
            if not Y_adjusted.empty:
                month_returns_log   = current_month_returns.iloc[row_number]
                w = optimize_portfolio(Y_adjusted)

                weight_concat, w_df, sharpe_array_concat = weightings(w, Y_adjusted, index, weight_concat, sharpe_array_concat, 1)
                month_returns_log   = pd.DataFrame(month_returns_log)

                Y_adjusted_next_L   = pd.DataFrame(asset_trimmer(row_number+2, trend_df, next_month_returns)) #Long

                w = w_df.drop('sharpe', axis=1)

                for col in w:
                    new_df = pd.DataFrame(w[col].values * Y_adjusted_next_L[col], columns=[col])
                    portfolio_return.index = new_df.index
                    if new_df.index.equals(portfolio_return.index):
                        portfolio_return = portfolio_return.merge(new_df, left_index = True, right_index=True)
                    else:
                        print("The index of new_df f{new_df.index} doesnt match the index of portfolio_return f{portfolio_return}")

            portfolio_return = pd.DataFrame(portfolio_return.sum(axis=1), columns=['portfolio_return'])
        portfolio_return_concat = pd.concat([portfolio_return_concat, portfolio_return], axis=0) #Long
    return portfolio_return_concat, weight_concat, sharpe_array_concat


portfolio_return_concat, weight_concat, sharpe_array_concat = optimizerbacktest(monthly_returns_log, dummy_L_df, daily_returns_log)


merged_df = portfolio_return_concat.sort_index(ascending=True)
merged_df.iloc[0] = 0
merged_df = (1 + merged_df).cumprod() * 10000
print(merged_df)
