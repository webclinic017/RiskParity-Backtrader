import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import minimize, Bounds, LinearConstraint
from Utils import *
from datamanagement import *
from Trend_Following import dummy_L_df
#from Trend_Following import * #Start, End, ret, dummy_L_df, months_between, next_month

monthly_returns, asset_classes, asset = data_management(Start, End, '1mo')

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

monthly_returns_log = np.log(monthly_returns/monthly_returns.shift(1))
monthly_returns_log = monthly_returns_log.dropna()
monthly_returns_log.index = pd.to_datetime(monthly_returns_log.index).strftime('%Y-%m-%d')

def optimizerbacktest(Y_adjusted, trend_df):
    weight_concat = sharpe_array_concat = portfolio_return_concat = pd.DataFrame()
    stopper = len(monthly_returns_log)
    for row_number, (index, row) in enumerate(monthly_returns_log.iterrows(), 1):
        print(row_number, stopper)
        if row_number != stopper:
            Y_adjusted = asset_trimmer(index, trend_df, monthly_returns_log)

            month_returns_log = cov = pd.DataFrame()
            month_returns_log = monthly_returns_log.iloc[row_number-1]
            next_month_returns = monthly_returns_log.iloc[row_number]
            cov = month_returns_log.T.cov(other=month_returns_log)

            exp_ret = month_returns_log.values
            W = np.ones((7,1))

            x = optimize(ret_risk, W, exp_ret, cov, target_return=0.055)
            w = x['x']
            weight_concat, w_df, sharpe_array_concat = weightings(w, Y_adjusted, index, weight_concat, sharpe_array_concat, 1)
            month_returns_log = pd.DataFrame(month_returns_log)
            Y_adjusted_next_L = pd.DataFrame(asset_trimmer(row_number, trend_df, next_month_returns)) #Long

            ######### this needs to use daily data, I need a daily df

            portfolio_return = portfolio_returns(w_df, Y_adjusted_next_L.T, index, w) #Long
        portfolio_return_concat = pd.concat([portfolio_return_concat, portfolio_return], axis=0) #Long
    return portfolio_return_concat, weight_concat, sharpe_array_concat

portfolio_return_concat, weight_concat, sharpe_array_concat = optimizerbacktest(monthly_returns_log, dummy_L_df)

'''
Here will be the optimization,

It needs to have monthly data fed into the optimization.

'''