import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.optimize import minimize
from scipy.optimize import minimize, Bounds, LinearConstraint
from Utils import *
import warnings
from dateutil.relativedelta import relativedelta
from datamanagement import *
from Trend_Following import dummy_L_df, ret as daily_returns, asset_classes, asset
warnings.filterwarnings("ignore")

#from Trend_Following import * #Start, End, ret, dummy_L_df, months_between, next_month

monthly_returns, asset_classes, asset = data_management(Start, End, '1mo')
#daily_returns, asset_classes, asset = data_management(Start, End, '1d')
monthly_returns = monthly_returns.dropna()

### New optimization, max sharpe
def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0):
    portfolio_return, portfolio_var, portfolio_std = portfolio(weights, mean_returns, cov_matrix)
    sr = ((portfolio_return - risk_free_rate)/portfolio_std) * (len(mean_returns)**0.5) # annualized
    return(-sr)

def calc_returns_stats(returns):
    """
    Parameters
        returns: returns timeseries pd.DataFrame object

    Returns:
        mean_returns: Avereage of returns
        cov_matrix: returns Covariance matrix
    """
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    print(mean_returns)
    return(mean_returns, cov_matrix)

def portfolio(weights, mean_returns, cov_matrix):

    portfolio_return = np.dot(weights.reshape(1,-1), mean_returns.values.reshape(-1,1))
    portfolio_var = np.dot(np.dot(weights.reshape(1,-1), cov_matrix.values), weights.reshape(-1,1))
    portfolio_std = np.sqrt(portfolio_var)

    return(np.squeeze(portfolio_return),np.squeeze(portfolio_var),np.squeeze(portfolio_std))


def optimize_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate=0, w_bounds=(0,1)):
    "This function finds the portfolio weights which minimize the negative sharpe ratio"

    init_guess = np.array([1/len(mean_returns) for _ in range(len(mean_returns))])
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints =   ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                    {'type': 'ineq', 'fun': lambda x: 0.6 - x},
                    {'type': 'ineq', 'fun': lambda x: np.sum(x > 0.01) - 3},  # Constraint: at least 3 assets must be invested in
                   )
    result = opt.minimize(fun=neg_sharpe_ratio,
                          x0=init_guess,
                          args=args,
                          method='SLSQP',
                          bounds=tuple(w_bounds for _ in range(len(mean_returns))),
                          constraints=constraints,
                          )
    
    if result['success']:
        print(result['message'])
        opt_sharpe = - result['fun']
        opt_weights = result['x']
        opt_return, opt_variance, opt_std = portfolio(opt_weights, mean_returns, cov_matrix)
        print(opt_sharpe)
        return(opt_sharpe, opt_weights, opt_return.item()*252, opt_variance.item()*252, opt_std.item()*(252**0.5))
    else:
        print("Optimization was not succesfull!")
        print(result['message'])
        return(None)



### Old optimization method, min variance
def optimize_portfolio(returns_data):
    # Number of assets in the portfolio
    num_assets = len(returns_data.columns)

    # Calculate covariance matrix
    cov_matrix = returns_data.cov()

    # Set random seed for reproducibility

    # Define optimization function
    def portfolio_variance(weights):
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        return portfolio_variance

    # Define optimization constraints
    constraints = (
    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Constraint: weights sum to 1
    )

    # Define optimization bounds
    bounds = tuple((0, 1) for _ in range(num_assets))  # Bounds: 0 <= weights <= 1

    # Initial guess for weights
    initial_weights = np.ones(num_assets) / num_assets

    # Perform portfolio optimization
    result = minimize(portfolio_variance, initial_weights, method='SLSQP', constraints=constraints, bounds=bounds)

    print(result.message)

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


def optimizerbacktest(Y_adjusted, trend_df, daily_returns_log):
    weight_concat = sharpe_array_concat = portfolio_return_concat = pd.DataFrame()
    stopper = len(monthly_returns_log)
    monthly_returns_log.index = pd.to_datetime(monthly_returns_log.index)  # Convert index to datetime
    daily_returns_log.index = pd.to_datetime(daily_returns_log.index)
    date_bench = daily_returns_log.iloc[199:200].index
    for row_number, (index, row) in enumerate(monthly_returns_log.iterrows(), 1):
        if pd.DatetimeIndex([index]) > date_bench:

            next_month = index + relativedelta(months=1)

            #Set up empty DF
            portfolio_return = trend_df_2 = Y_adjusted = current_month_returns = pd.DataFrame()

            #Split my df into current and next month, current for calculating the portfolio, and next for the backtest.
            current_month_returns = daily_returns_log[(daily_returns_log.index.month == index.month) & (daily_returns_log.index.year == index.year)]
            next_month_returns = daily_returns_log[(daily_returns_log.index.month == next_month.month) & (daily_returns_log.index.year == next_month.year)]

            if row_number != stopper:

                #This is my trend tracker df, and we adjust the returns and next returns later on by this to select only trending assets.
                trend_df_2 = pd.DataFrame(trend_df.iloc[row_number+1])
                Y_adjusted = asset_trimmer(trend_df_2.T, current_month_returns)

                if not Y_adjusted.empty:

                    mean_returns, cov_matrix = calc_returns_stats(Y_adjusted)

                    # max sharpe portfolio
                    opt_sharpe, w, opt_return, opt_variance, opt_std = optimize_sharpe_ratio(
                                                                                    mean_returns,
                                                                                    cov_matrix,
                                                                                    risk_free_rate=0, w_bounds=(0,1))

                    weight_concat, w_df, sharpe_array_concat = weightings(w, Y_adjusted, next_month, weight_concat, sharpe_array_concat, 1)
                    Y_adjusted_next_L   = pd.DataFrame(asset_trimmer(pd.DataFrame(trend_df.iloc[row_number+1]), next_month_returns)) #Long
                    w = w_df.drop('sharpe', axis=1)
                    for col in w:
                        new_df = pd.DataFrame(w[col].values * Y_adjusted_next_L[col], columns=[col])
                        portfolio_return.index = new_df.index
                        portfolio_return = portfolio_return.merge(new_df, left_index = True, right_index=True)
                    portfolio_return = pd.DataFrame(portfolio_return.sum(axis=1), columns=['portfolio_return'])
                portfolio_return_concat = pd.concat([portfolio_return_concat, portfolio_return], axis=0) #Long

    return portfolio_return_concat, weight_concat, sharpe_array_concat

portfolio_return_concat, weight_concat, sharpe_array_concat = optimizerbacktest(monthly_returns_log, dummy_L_df, daily_returns_log)

print(weight_concat)