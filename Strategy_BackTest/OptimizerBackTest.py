import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.optimize import minimize
from scipy.optimize import minimize, Bounds, LinearConstraint
from Utils import *
import warnings
from dateutil.relativedelta import relativedelta
import quantstats as qs
from Trend_Following import dummy_L_df, ret as daily_returns, asset_classes, asset
warnings.filterwarnings("ignore")
### New optimization, max sharpe

'''
I could get a list of each asset group and set the maximum sum weight = x?

'''

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0):
    portfolio_return, portfolio_var, portfolio_std = portfolio(weights, mean_returns, cov_matrix)
    sr = ((portfolio_return - risk_free_rate)/portfolio_std) * (len(mean_returns)**0.5) # annualized
    return(-sr)

def calc_returns_stats(returns):

    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    return(mean_returns, cov_matrix)

def portfolio(weights, mean_returns, cov_matrix):
    portfolio_return = np.dot(weights.reshape(1,-1), mean_returns.values.reshape(-1,1))
    portfolio_var = np.dot(np.dot(weights.reshape(1,-1), cov_matrix.values), weights.reshape(-1,1))
    portfolio_std = np.sqrt(portfolio_var)

    return(np.squeeze(portfolio_return),np.squeeze(portfolio_var),np.squeeze(portfolio_std))

def optimize_sharpe_ratio(Y_adjusted, mean_returns, cov_matrix, mweight, asset_constraints, nmore, risk_free_rate=0, w_bounds=(0,1)):
    init_guess = np.array([1/len(mean_returns) for _ in range(len(mean_returns))])
    args = (mean_returns, cov_matrix, risk_free_rate)

    constraints =   [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, # Sum = 1
                    {'type': 'ineq', 'fun': lambda x, max_value=0.5: max_value - np.max(x)}, # max value for a variable > than max_value
                    {'type': 'ineq', 'fun': lambda x, nmore=3: (nmore - np.count_nonzero(x == 0))}, # The n > 0 must be at least nmore
    ]

    # I could add some constraints about how much of each asset class so x -
    for industry in asset_constraints['Industry'].unique():
        assets_in_industry = asset_constraints[asset_constraints['Industry'] == industry]['Asset']
        max_weight = asset_constraints[asset_constraints['Industry'] == industry]['Max_Weight'].iloc[0]
        
        existing_assets = [asset for asset in assets_in_industry if asset in Y_adjusted.columns]
        print(max_weight, existing_assets)

    result = opt.minimize(fun=neg_sharpe_ratio,
                          x0=init_guess,
                          args=args,
                          method='SLSQP',
                          bounds=tuple(w_bounds for _ in range(len(mean_returns))),
                          constraints=constraints,
                          options={'maxiter': 100},
                          )
    
    if result['success']:
        opt_sharpe = - result['fun']
        opt_weights = result['x']
        opt_return, opt_variance, opt_std = portfolio(opt_weights, mean_returns, cov_matrix)
        b = True
        return(opt_sharpe, opt_weights, opt_return.item()*252, opt_variance.item()*252, opt_std.item()*(252**0.5), b)
    else:
        print(result['message'])

        opt_weights = [0.6,0.4]
        opt_sharpe = opt_return = opt_variance = opt_std = [0]

        b = False
        return(opt_sharpe, opt_weights, opt_return, opt_variance, opt_std, b)


def ret(monthly_returns):
    monthly_returns_log = monthly_returns.pct_change() #np.log(monthly_returns/monthly_returns.shift(1))
    monthly_returns_log = monthly_returns_log.dropna()
    monthly_returns_log.index = pd.to_datetime(monthly_returns_log.index).strftime('%Y-%m-%d')
    return monthly_returns_log



def optimizerbacktest(Y_adjusted, trend_df, daily_returns_log, N_More, Max_weight, monthly_returns_log, asset_constraints):
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
            start_cur   = current_month_returns.index[0]
            end_cur     = current_month_returns.index[-1]

            if row_number != stopper:
                start_next_month  = next_month_returns.index[0]
                end_next    = next_month_returns.index[-1]
                #This is my trend tracker df, and we adjust the returns and next returns later on by this to select only trending assets.
                trend_df_2 = pd.DataFrame(trend_df.iloc[row_number+1])
                Y_adjusted = asset_trimmer(trend_df_2.T, current_month_returns)

                if not Y_adjusted.empty:

                    mean_returns, cov_matrix = calc_returns_stats(Y_adjusted)

                    # max sharpe portfolio
                    opt_sharpe, w, opt_return, opt_variance, opt_std, b = optimize_sharpe_ratio(Y_adjusted,
                                                                                    mean_returns,
                                                                                    cov_matrix,
                                                                                    Max_weight,
                                                                                    asset_constraints,
                                                                                    N_More,
                                                                                    risk_free_rate=0, w_bounds=(0,1))
                    if b == False:
                        data = np.array([[0.6, 0.4, 1]])
                        next_month  = start_next_month.month
                        next_year   = start_next_month.year
                        start_next_month = pd.to_datetime(datetime(next_year, next_month, 1))
                        w = pd.DataFrame(data, columns=['VTI', 'BND', 'sharpe'])
                        w.index = [start_next_month]
                        Y_adjusted = pd.DataFrame()
                        Y_adjusted_next_L = pd.DataFrame()
                        Y_adjusted['VTI'] = yf.download("VTI", start=start_cur, end=end_cur)['Adj Close'].pct_change()
                        Y_adjusted['BND'] = yf.download("BND", start=start_cur, end=end_cur)['Adj Close'].pct_change()
                        Y_adjusted_next_L["VTI"] = yf.download("VTI", start=start_next_month, end=end_next)['Adj Close'].pct_change().dropna()#* 0.6
                        Y_adjusted_next_L['BND'] = yf.download("BND", start=start_next_month, end=end_next)['Adj Close'].pct_change().dropna()#* 0.4
                        weight_concat = pd.concat([weight_concat, w]).fillna(0)
                        opt_sharpe = (daily_returns.mean() - 0.04) / daily_returns.std()
                        sharpe_array = w
                        sharpe_array['sharpe'] = opt_sharpe
                        sharpe_array_concat = pd.concat([sharpe_array_concat, sharpe_array]).fillna(0)

                    else:
                        Y_adjusted_next_L   = pd.DataFrame(asset_trimmer(pd.DataFrame(trend_df.iloc[row_number+1]), next_month_returns)) #Long
                        weight_concat, w, sharpe_array_concat = weightings(w, Y_adjusted, next_month, weight_concat, sharpe_array_concat, 1, b)

                    w = w.drop('sharpe', axis=1)
                    for col in w:
                        new_df = pd.DataFrame(w[col].values * Y_adjusted_next_L[col], columns=[col])
                        portfolio_return.index = new_df.index
                        portfolio_return = portfolio_return.merge(new_df, left_index = True, right_index=True)
                    portfolio_return = pd.DataFrame(portfolio_return.sum(axis=1), columns=['portfolio_return'])
                portfolio_return_concat = pd.concat([portfolio_return_concat, portfolio_return], axis=0) #Long

    return portfolio_return_concat, weight_concat, sharpe_array_concat