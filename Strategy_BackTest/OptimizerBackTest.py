import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.optimize import minimize
from scipy.optimize import minimize, Bounds, LinearConstraint
from Utils import *
import warnings
from dateutil.relativedelta import relativedelta
from datamanagement import *
import quantstats as qs
from Trend_Following import dummy_L_df, ret as daily_returns, asset_classes, asset
warnings.filterwarnings("ignore")

#from Trend_Following import * #Start, End, ret, dummy_L_df, months_between, next_month

monthly_returns, asset_classes, asset, assets = data_management(Start, End, '1mo')
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
                    {'type': 'ineq', 'fun': lambda x: 0.5 - x},
                    {'type': 'ineq', 'fun': lambda x: np.sum(x > 0.01) - 3}  # x > 0.01) - 3 looks good
                   )
    result = opt.minimize(fun=neg_sharpe_ratio,
                          x0=init_guess,
                          args=args,
                          method='SLSQP',
                          bounds=tuple(w_bounds for _ in range(len(mean_returns))),
                          constraints=constraints,
                          options={'maxiter': 100},
                          )
    
    if result['success']:
        print(result['message'])
        opt_sharpe = - result['fun']
        opt_weights = result['x']
        opt_return, opt_variance, opt_std = portfolio(opt_weights, mean_returns, cov_matrix)
        b = True
        return(opt_sharpe, opt_weights, opt_return.item()*252, opt_variance.item()*252, opt_std.item()*(252**0.5), b)
    else:
        print("Optimization was not succesfull!")
        print(result['message'])

        opt_weights = [0.6,0.4]
        opt_sharpe = opt_return = opt_variance = opt_std = [0]

        b = False
        return(opt_sharpe, opt_weights, opt_return, opt_variance, opt_std, b)



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
                    opt_sharpe, w, opt_return, opt_variance, opt_std, b = optimize_sharpe_ratio(
                                                                                    mean_returns,
                                                                                    cov_matrix,
                                                                                    risk_free_rate=0, w_bounds=(0,1))
                    if b == False:
                        print("Failed")
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

portfolio_return_concat, weight_concat, sharpe_array_concat = optimizerbacktest(monthly_returns_log, dummy_L_df, daily_returns_log)
weight_concat = weight_concat.sort_index(axis=1)
assets        = assets.sort_values(by=['Industry', 'Asset'], key=lambda x: x.str.lower())
assets        = assets.reset_index(drop=True)
df_2 = pd.DataFrame(columns = assets['Asset'].to_list())
weight_concat = weight_concat.reindex(columns=df_2.columns)

asset_pick = assets['Industry'].to_list()

bonds       = asset_pick.index("Bonds")
commodities = asset_pick.index("Commodities")
defense     = asset_pick.index("Defense")
energies    = asset_pick.index("Energies")
equities    = asset_pick.index("Equities")
housing     = asset_pick.index("Housing")
metals      = asset_pick.index("Metals")

leng         = len(asset_pick)

benchmark = 'Benchmark'
Bench, Merged_df = bench(portfolio_return_concat.index.min(), benchmark, portfolio_return_concat)
# Old method:
def old_sharpe(df):
    Port_ret = ((df.iloc[-1] - 10000)/10000)
    num_years = (df.index.max() - df.index.min()).days / 365
    num_days = len(df)
    average_number_days = num_days/num_years
    Sharpe_Ratio =  (np.sqrt(average_number_days)*(Port_ret - 0.04) / df.pct_change().std())
    
    return Sharpe_Ratio.to_string()

Port_sh = old_sharpe(Merged_df)
#Bench_sh = old_sharpe(Bench)


qs.reports.html(Merged_df.iloc[:,0], Bench.pct_change().iloc[:,0], output='F:/outputs/quantstats-tearsheet.html')
#qs.reports.html(Merged_df, Bench)
weight_concat['Index'] = weight_concat.index
weight_concat['Index'] = pd.to_datetime(weight_concat['Index'])
weight_concat['Index'] = weight_concat['Index'].dt.strftime('%Y-%m-%d')
weight_concat.index = weight_concat['Index']

weight_concat = weight_concat.drop('Index', axis=1)

weight_concat = weight_concat.round(2)
#04B018

column_styles = [
    {"selector": f".col{idx}", "props": [("border-right", "1px solid magenta"), ("border-left", "1px solid magenta")]}
    for idx in range(0, bonds)
] + [
    {"selector": f".col{idx}", "props": [("border-right", "1px solid brown"), ("border-left", "1px solid brown")]}
    for idx in range(bonds, commodities)
] + [
    {"selector": f".col{idx}", "props": [("border-right", "1px solid grey"), ("border-left", "1px solid grey")]}
    for idx in range(commodities, defense)
] + [
    {"selector": f".col{idx}", "props": [("border-right", "1px solid orange"), ("border-left", "1px solid orange")]}
    for idx in range(defense, energies)
] + [
    {"selector": f".col{idx}", "props": [("border-right", "1px solid green"), ("border-left", "1px solid green")]}
    for idx in range(energies, equities)
] + [
    {"selector": f".col{idx}", "props": [("border-right", "1px solid brown"), ("border-left", "1px solid brown")]}
    for idx in range(equities, housing)
] + [
    {"selector": f".col{idx}", "props": [("border-right", "1px solid gold"), ("border-left", "1px solid gold")]}
    for idx in range(housing, metals)
] + [
    {"selector": f".col{idx}", "props": [("border-right", "1px solid purple"), ("border-left", "1px solid purple")]}
    for idx in range(metals, leng)
]+ [
    {"selector": ".highlight-last-row tr:last-child", "props": [("border", "1px solid black")]}
]

html_table = (
    weight_concat.style
    .applymap(lambda x: (
        f'background-color: {"#04B018" if x > 0.5 else "#9ACD32" if x > 0.3 else "#6FD17A" if x > 0.1 else "#D6FF97" if x > 0.05 else ""}'
    ), subset=pd.IndexSlice[:, :])
    .format("{:.2f}")
    .set_properties(**{'text-align': 'right'})
    .set_table_styles(column_styles)
    .render()
)

with open(r'C:/Users/Kit/RPVSCode/RiskParity/quantstats-tearsheet.html') as file:
    html_content = file.read()

modified_content = html_content + html_table

with open('C:/Users/Kit/RPVSCode/RiskParity/quantstats-tearsheet.html','w') as file:
    file.write(modified_content)
