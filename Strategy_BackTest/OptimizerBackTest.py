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
    return -(np.dot(W.T.flatten(), exp_ret) / np.sqrt(np.dot(np.dot(W.T, cov), W) * np.dot(exp_ret.T, exp_ret)))

def ret(monthly_returns):
    monthly_returns_log = monthly_returns.pct_change() #np.log(monthly_returns/monthly_returns.shift(1))
    monthly_returns_log = monthly_returns_log.dropna()
    monthly_returns_log.index = pd.to_datetime(monthly_returns_log.index).strftime('%Y-%m-%d')
    return monthly_returns_log

monthly_returns_log = ret(monthly_returns)
daily_returns_log   = ret(daily_returns)

def optimizerbacktest(Y_adjusted, trend_df, daily_returns_log):
    weight_concat = sharpe_array_concat = portfolio_return_concat = pd.DataFrame()
    stopper = len(monthly_returns_log)
    monthly_returns_log.index = pd.to_datetime(monthly_returns_log.index)  # Convert index to datetime
    daily_returns_log.index = pd.to_datetime(daily_returns_log.index)

    for row_number, (index, row) in enumerate(monthly_returns_log.iterrows(), 1):
        current_month = index.month
        next_month = current_month + 1
        portfolio_return = month_returns_log = cov = pd.DataFrame()

        current_month_returns = daily_returns_log[daily_returns_log.index.month == current_month]
        next_month_returns = daily_returns_log[daily_returns_log.index.month == next_month]
        if row_number != stopper:
            print(index)
            Y_adjusted          = asset_trimmer(index, trend_df, current_month_returns)
            print(Y_adjusted)
            if not Y_adjusted.empty:
                month_returns_log   = current_month_returns.iloc[row_number]
                cov                 = month_returns_log.T.cov(other=month_returns_log)
                exp_ret             = month_returns_log.values
                W                   = np.ones((month_returns_log.shape[0],1))*(1.0/month_returns_log.shape[0])
                x                   = optimize(ret_risk, W, exp_ret, cov, target_return=0.055)
                w                   = x['x']
                weight_concat, w_df, sharpe_array_concat = weightings(w, Y_adjusted, index, weight_concat, sharpe_array_concat, 1)
                month_returns_log   = pd.DataFrame(month_returns_log)

                # current_month_returns here should be next month returns...
                Y_adjusted_next_L   = pd.DataFrame(asset_trimmer(row_number, trend_df, next_month_returns)) #Long

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
print(merged_df.to_string())