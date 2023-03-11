import pandas as pd
import datetime
import yfinance as yf
import numpy as np 
import warnings
import matplotlib.pyplot as plt
from datetime import datetime
from calendar import monthrange
from dateutil.relativedelta import relativedelta
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import concurrent.futures
from scipy.optimize import minimize
#from optimizer import optimizer_backtest
from Trend_Following import dummy_L_df, ret, Start, End, dummy_LS_df, number_of_iter, asset_classes
warnings.filterwarnings("ignore")
############################################################
# Variables and setup
############################################################
counter = 4

date1 = datetime.strptime(Start, "%Y-%m-%d")
date2 = datetime.strptime(End, "%Y-%m-%d")
diff = relativedelta(date2, date1)

Start_bench = date1 + relativedelta(months=1)

months_between = (diff.years)*12 + diff.months + 1

############################################################
# Setting up empty DFs
############################################################

merged_df = pd.DataFrame([])
sharpe_array = pd.DataFrame([])
df_dummy_sum = pd.DataFrame()
df_dummy_sum = pd.DataFrame()
this_month_weight = pd.DataFrame([])

# Optimizer

def optimize_risk_parity(Y, counter, i):

    Ycov = Y.cov()
    n = Y.shape[1]
    # Define the risk contribution as a constraint
    def risk_contribution(w):
        sigma = np.sqrt(np.matmul(np.matmul(w, Ycov), w))
        return (np.matmul(w, Ycov) * w) / sigma
    # Define the optimization objective
    def objective(w):
        return -np.sum(w * Y.mean())
    # Define the optimization constraints
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': lambda w: np.sum(risk_contribution(w)) - (1/n)},
            ]
    bounds = [(0, 1) for i in range(n)]
    # Call the optimization solver
    res = minimize(objective, np.ones(n)/n, constraints=cons, bounds=bounds, method='SLSQP',
                   options={'disp': False, 'eps': 1e-12, 'maxiter': 10000})
    print(res.message)
    print(res.success)

    return res.x

############################################################
# Monte carlo
############################################################

def monte_carlo(Y):
    log_return = np.log(Y/Y.shift(1))
    sample = Y.shape[0]
    num_ports = number_of_iter
    all_weights = np.zeros((num_ports, len(Y.columns)))
    ret_arr = np.zeros(num_ports)
    vol_arr = np.zeros(num_ports)
    sharpe_arr = np.zeros(num_ports)

    for ind in range(num_ports): 
        # weights 
        weights = np.random.dirichlet(np.ones(len(Y.columns)), size=1)
        weights = np.squeeze(weights)
        
        # Enforce minimum weight
        weights = np.maximum(weights, 0.05)
        weights = weights/np.sum(weights)
        all_weights[ind,:] = weights
        
        # expected return 
        ret_arr[ind] = np.sum((log_return.mean()*weights)*sample)

        # expected volatility 
        vol_arr[ind] = np.sqrt(np.dot(weights.T,np.dot(log_return.cov()*sample, weights)))

        # Sharpe Ratio 
        sharpe_arr[ind] = ret_arr[ind]/vol_arr[ind]
    max_sh = sharpe_arr.argmax()
    #plot_frontier(vol_arr,ret_arr,sharpe_arr)
    sharpe_ratio = ret_arr[max_sh]/vol_arr[max_sh]
    #To-do:
    #enable short selling
    #enable leverage
    return all_weights[max_sh,:], sharpe_ratio

def monte_carlo_SL(Y, short_df):
    log_return = np.log(Y/Y.shift(1))
    sample = Y.shape[0]
    num_ports = 1000
    all_weights = np.zeros((num_ports, len(Y.columns)))
    ret_arr = np.zeros(num_ports)
    vol_arr = np.zeros(num_ports)
    sharpe_arr = np.zeros(num_ports)

    for ind in range(num_ports): 
        # weights 
        weights = np.array(np.random.random(len(Y.columns))) 

        # set maximum weight of 1 for assets with value of 1 or greater in short_df
        mask = short_df >= 1
        weights[mask] = weights[mask] / np.sum(weights[mask])
        weights[~mask] = weights[~mask] / np.sum(np.abs(weights[~mask]))

        # save the weights
        all_weights[ind,:] = weights

        # expected return 
        ret_arr[ind] = np.sum((log_return.mean()*weights)*sample)

        # expected volatility 
        vol_arr[ind] = np.sqrt(np.dot(weights.T,np.dot(log_return.cov()*sample, weights)))

        # Sharpe Ratio 
        sharpe_arr[ind] = ret_arr[ind]/vol_arr[ind]

    max_sh = sharpe_arr.argmax()

    return all_weights[max_sh,:]

############################################################

def plot_frontier(vol_arr,ret_arr,sharpe_arr):
    plt.figure(figsize=(12,8))
    plt.scatter(vol_arr,ret_arr,c=sharpe_arr,cmap='plasma')
    plt.colorbar(label='Sharpe Ratio')
    max_sr_ret = ret_arr[sharpe_arr.argmax()]
    max_sr_vol = vol_arr[sharpe_arr.argmax()]
    max_sr_sr  = sharpe_arr[sharpe_arr.argmax()]
    print("Max values", max_sr_ret,max_sr_vol, max_sr_sr, "Max possible Sharpe:", max(sharpe_arr))
    # plot the dataplt.figure(figsize=(12,8))
    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.scatter(max_sr_vol, max_sr_ret, c='red', s=50, edgecolors='black')

############################################################
# Building a loop that estimate optimal portfolios on
# rebalancing dates
############################################################

rng_start = pd.date_range(Start, periods=months_between, freq='MS')

def next_month(i):
    i_str = i.strftime('%Y-%m')
    dt = datetime.strptime(i_str, '%Y-%m')
    next_month = dt + relativedelta(months=1)
    next_i = datetime(next_month.year, next_month.month, 1)
    next_b = pd.date_range(start=next_i, periods=1, freq='M')
    next_b = next_b[0]
    return next_i,next_b

############################################################
# Calculate sharpe for next month
############################################################

def next_sharpe(weights, log_return, sharpe_list):
    sample = log_return.shape[0]
    ret_arr2 = np.sum((log_return.mean()*weights)*sample)
    # expected volatility 
    vol_arr2 = np.sqrt(np.dot(weights.T,np.dot(log_return.cov()*sample, weights)))
    sharpe_arr2 = ret_arr2/vol_arr2
    return sharpe_arr2

############################################################
# Backtesting
############################################################
def backtest(rng_start, ret, ret_pct, dummy_L_df, dummy_LS_df, ls, monte):
    y_next                  = pd.DataFrame([])
    portfolio_return_concat = pd.DataFrame([])
    portfolio_return        = pd.DataFrame([])
    weight_concat           = pd.DataFrame([])
    sharpe_array_concat     = pd.DataFrame([])
    for i in rng_start:
        rng_end = pd.date_range(i, periods=1, freq='M')
        for b in rng_end:
            # cleanup here
            if rng_start[-1] == i and prev_i is not None and prev_b is not None:
                print(f"Last month {i}")
            else:
                if ls == 0:
                    Y_LS = ret[i:b]
                    Y_adjusted_LS = asset_trimmer_LS(b, dummy_LS_df, Y_LS)
                    if not Y_adjusted_LS.empty:
                        w_SL = monte_carlo_SL(Y_adjusted_LS, dummy_LS_df)
                        next_i,next_b = next_month(i)
                        y_next = ret_pct[next_i:next_b]
                        Y_adjusted_next_SL = asset_trimmer_LS(b, dummy_LS_df, y_next)
                        portfolio_return = portfolio_returns(w_SL, Y_adjusted_next_SL, b)
                else:
                    Y = ret[i:b]
                    Y_adjusted = asset_trimmer(b, dummy_L_df, Y)
                    if not Y_adjusted.empty:
                        if monte == 0:
                            print("Calculating using optimization")
                            w = optimize_risk_parity(Y_adjusted, counter, i)
                            sharpe_ratio = 1
                        else:
                            print("Calculating using MonteCarlo")
                            w = threader(Y)
                            w, sharpe_ratio = monte_carlo(Y_adjusted) #Long
                        next_i,next_b = next_month(i)
                        weight_concat = weightings(w, Y_adjusted, next_i, weight_concat, sharpe_array_concat, sharpe_ratio)
                        y_next = ret_pct[next_i:next_b]
                        Y_adjusted_next_L = asset_trimmer(b, dummy_L_df, y_next) #Long
                        portfolio_return = portfolio_returns(w, Y_adjusted_next_L) #Long
                prev_i = i
                prev_b = b
                portfolio_return_concat = pd.concat([portfolio_return, portfolio_return_concat], axis=0) #Long
    return portfolio_return_concat, weight_concat

# Function to drop if the asset is not trending.
    # I guess the next step is to set up if statement so we can just have both functions working with an "if"
def asset_trimmer(b, df_monthly, Y):
    df_split_monthly = df_monthly[b:b]
    cols_to_drop = [col for col in df_split_monthly.columns if df_split_monthly[col].max() < 0.8]
    Y = Y.drop(columns=cols_to_drop)
    return Y

# This is the LS version of above, needs sorting, low prio.

def asset_trimmer_LS(b, df_monthly, Y):
    df_split_monthly = df_monthly[b:b]
    print("are we here???")
    cols_to_drop = [col for col in df_split_monthly.columns if (-0.8 < df_split_monthly[col].max() < 0.8)]
    print("Trend DF", df_split_monthly.drop(columns=cols_to_drop))
    Y = Y.drop(columns=cols_to_drop)
    return Y
# Function to manage weights.

def weightings(w, Y_adjusted, i, weight_concat, sharpe_array_concat, sharpe_ratio):
    w_df = pd.DataFrame(w).T
    w_df.columns = Y_adjusted.columns
    w_df['date'] = w_df.index
    w_df['date'] = i
    w_df.set_index('date', inplace=True)
    sharpe_array = w_df
    sharpe_array['sharpe'] = sharpe_ratio
    sharpe_array_concat = pd.concat([sharpe_array_concat, sharpe_array])
    weight_concat = pd.concat([weight_concat,w_df]).fillna(0)
    return weight_concat

# Function to calculate portfolio returns

def portfolio_returns(w, Y_adjusted_next):
    print(w)
    df_daily_return = w.T*Y_adjusted_next
    df_portfolio_return = pd.DataFrame(df_daily_return.sum(axis=1), columns=['portfolio_return'])
    return df_portfolio_return

# Multithreading, this needs to be sorted soon.
def threader(Y):
    num_threads = 8
    w = monte_carlo(Y)
    # split the data into num_threads chunks
    chunks = np.array_split(Y, num_threads)

    # create a ThreadPoolExecutor with num_threads workers
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)

    # submit a monte_carlo job for each chunk of data
    futures = [executor.submit(threader, chunk) for chunk in chunks]

    # wait for all jobs to complete
    concurrent.futures.wait(futures)
    return(w)

# Correlation matrix used in the plotly dash.

def correlation_matrix(sharpe_array, column):
    corr_matrix = sharpe_array.corr()
    corr_matrix = corr_matrix[f'{column}']
    return corr_matrix

############################################################
# Calling my functions
############################################################
ret_pct = ret.pct_change()

ls = 1 # Dummy for long short, shorting needs to be shorted out, but not high prio.
monte = 0
# Data management of weights and returns.
portfolio_return_concat, weight_concat = backtest(rng_start, ret, ret_pct, dummy_L_df, dummy_LS_df, ls, monte)


sharpe_array = weight_concat.copy()
weight_concat.drop('sharpe', axis=1, inplace=True)

this_month_weight = weight_concat.iloc[-2]
this_month_weight = pd.DataFrame([this_month_weight])
weight_concat = weight_concat.drop(index=weight_concat.index[-1])

############################################################
# Spy returns & portfolio returns
############################################################

portfolio_return_concat = pd.DataFrame(pd.DataFrame(portfolio_return_concat))

Bench_start = portfolio_return_concat.index.min()
Bench_end   = portfolio_return_concat.index.max()

SPY = yf.download('SPY', start=Bench_start, Bench_end=End)['Adj Close'].pct_change()
SPY = pd.DataFrame(pd.DataFrame(SPY))

merged_df = SPY.merge(portfolio_return_concat, left_index=True, right_index=True)
merged_df.iloc[0] = 0

merged_df = (1 + merged_df).cumprod() * 10000
merged_df = merged_df.rename(columns={'Adj Close': 'SPY_Return'})


def long_names(asset_classes, weight):
    mapping_dict = dict(zip(asset_classes['Asset'], asset_classes['Full_name']))
    weight_long = weight.rename(columns=mapping_dict)
    return weight_long

# Generate the table of weights
def df_merger(weights_df, weight_long):
    for asset_df, asset_long in zip(weights_df, weight_long):
        column_name = f"{asset_long} ({asset_df})"
        weights_df.rename(columns={asset_df: column_name}, inplace=True)
        weight_long.rename(columns={asset_long: column_name}, inplace=True)
    return weights_df, weight_long

def generate_weights_table(weights_df, asset_classes):
    weight_long = long_names(asset_classes, weights_df)
    weights_df2 = weights_df.copy()
    weights_df, weight_long = df_merger(weights_df, weight_long)
    weights_table = html.Table(
        style={'border': '1px solid black', 'padding': '10px'},
        children=[
            # create table header row
            html.Tr(
                style={'background-color': 'grey',                              # Header
                       'color': 'white',
                       'border': '120px solid black',
                       'padding': '120px',
                       'font-family': 'Arial',
                       'font-size': '14px'},
                children=[
                    html.Th('Date:'),
                    *[html.Th(col, style={'text-align': 'center'}) for col in weights_df2.columns]
                ]
            ),
            # create table body rows
            *[html.Tr(
                children=[
                    html.Td(index, style={'font-weight': 'bold',                # Left index
                                          'border': '1px solid black',
                                          'padding': '1px',
                                          'font-family': 'Arial',
                                          'font-size': '14px',}),
                    *[html.Td(round(weights_df.loc[index, col], 4),
                              style={'text-align': 'center',
                                     'border': '1px solid grey',
                                     'padding': '1px',
                                     'font-family': 'Arial',
                                     'font-size': '12px',
                                     'background-color': '#0DBF00' if weights_df.loc[index, col] > 0.5 
                                       else '#9ACD32' if weights_df.loc[index, col] > 0.2 
                                       else '#6FD17A' if weights_df.loc[index, col] > 0.1
                                       else '#D6FF97' if weights_df.loc[index, col] > 0.04
                                       else 'white',
                                       },
                                       title=col,
                                ) for col in weights_df.columns],
                ]
            ) for index in weights_df.index.strftime('%Y-%m-%d')]
        ]
    )
    return weights_table


# Create the plotly dash

def portfolio_returns_app(returns_df, weights_df, this_month_weight, sharpe_array):
    # Calculate summary statistics for portfolio returns
    num_years = (returns_df.index.max() - returns_df.index.min()).days / 365
    num_days = len(returns_df)
    average_number_days = num_days/num_years
    returns = returns_df.pct_change()
    returns.dropna(inplace=True)

    # Portfolio data:
    Portfolio_Net_Returns = returns['portfolio_return'].mean()* num_days
    Portfolio_Average_Returns = returns['portfolio_return'].mean() * average_number_days
    Portfolio_std = returns['portfolio_return'].std() * average_number_days
    Portfolio_Sharpe_Ratio =  np.sqrt(average_number_days) * (Portfolio_Average_Returns / Portfolio_std)
    
    # SPY data:
    SPY_Net_Returns = returns['SPY_Return'].mean()*num_days
    SPY_Average_Returns = returns['SPY_Return'].mean() * average_number_days
    SPY_std = returns['SPY_Return'].std() * average_number_days
    SPY_Sharpe_Ratio = np.sqrt(average_number_days) * (SPY_Average_Returns / SPY_std)
    
    # Calculate monthly Sharpe ratio for last month
    last_month_returns = returns.loc[returns.index.month == returns.index[-2].month]
    last_month_mean_returns = last_month_returns['portfolio_return'].mean()
    last_month_std_returns = last_month_returns['portfolio_return'].std()
    last_month_sharpe_ratio = np.sqrt(12) * (last_month_mean_returns / last_month_std_returns)
    
    # Create a line chart of portfolio and benchmark returns
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=returns_df.index, y=returns_df['portfolio_return'], mode='lines', name='Portfolio Return'))
    fig.add_trace(go.Scatter(x=returns_df.index, y=returns_df['SPY_Return'], mode='lines', name='SPY Returns'))
    corr_matrix = correlation_matrix(sharpe_array, 'sharpe')
    corr_matrix = corr_matrix.to_frame()
    corr_matrix = corr_matrix.sort_values(by='sharpe', ascending=True)

    data = [go.Heatmap(z=corr_matrix.values,
                   x=corr_matrix.columns,
                   y=corr_matrix.index,
                   colorscale='RdBu',
                   hoverongaps=False,
                   hovertemplate='%{y}: %{x}<br>Correlation: %{z:.2f}<extra></extra>',
                   showscale=True,
                   zmin=-1,
                   zmax=1,
                   text=corr_matrix.round(2).values.astype(str),
                   texttemplate="%{text}",
                   textfont={"size":10})]
    # Create a table of summary statistics for portfolio and benchmark returns
    returns_table = html.Table(children=[
            html.Tr(children=[
                html.Th('Statistic'),
                html.Th('Portfolio'),
                html.Th('SPY')
            ]),
            html.Tr(children=[
                html.Td('Net Returns'),
                html.Td(round(Portfolio_Net_Returns, 4)),
                html.Td(round(SPY_Net_Returns, 4)),
            ]),
            html.Tr(children=[
                html.Td('Avg Yr Returns'),
                html.Td(round(Portfolio_Net_Returns / num_years, 4)),
                html.Td(round(SPY_Net_Returns / num_years, 4))
            ]),
            html.Tr(children=[
                html.Td('Std Returns'),
                html.Td(round(Portfolio_std, 4)),
                html.Td(round(SPY_std, 4))
            ]),
            html.Tr(children=[
                html.Td('Sharpe Ratio'),
                html.Td(str(round(Portfolio_Sharpe_Ratio, 4))),
                html.Td(str(round(SPY_Sharpe_Ratio, 4)))
            ]),
            html.Tr(children=[
                html.Td('L/M sharpe Ratio'),
                html.Td(str(round(last_month_sharpe_ratio, 4))),
                html.Td(str(round(np.sqrt(252) * (last_month_returns['SPY_Return'].mean() / last_month_returns['SPY_Return'].std()), 4)))
            ])
        ])
       
    app = dash.Dash(__name__)
    app.layout = html.Div(children=[
    html.H1(children='Portfolio Returns'),

    dcc.Graph(
        id='returns-chart',
        figure=fig
    ),
    html.H2(children='Weights'),

    generate_weights_table(weights_df, asset_classes),

    html.H2(children="Next Month Weights"),
    
    generate_weights_table(this_month_weight, asset_classes),

    html.H2(children='Summary Statistics', style={'font-size': '24px'}),
    returns_table,
    #I would like the index to be the ticker, and the hover on the chart to be the full asset name, it would also be nice in the weights table.
    html.H2(children='Correlation Matrix'),
    dcc.Graph(id='correlation-matrix', figure={'data': data},
            style={'width': '40vh',
                   'height': '90vh',
                   'font-family': 'Arial',
                   'font-size': '12px',}
            )
    ])
    return app

app = portfolio_returns_app(merged_df, weight_concat, this_month_weight, sharpe_array)
app.run_server(debug=False)

'''
Next steps:
-Enable short selling.
-Enable leverage.
-Instead of just printing the W of the best sharpe ratio, show me the top 10 (not strictly 10, but the principal counts) sharpes W and show me the average top 10,
-More assets enabled.
-More asset selection culling.
-Hovering of asset names on the weights table and the correlation matrix.
'''


'''
New project:
-For each month, rate us on how well we selected assets based on the next months weightings, if the weightings are within a bounds then we are ok, if they are 
    below the previous month then take note that we were in too deep with this asset class, so next time we think of re-balancing by increasing this asset, we can essentially rate our scores.

'''