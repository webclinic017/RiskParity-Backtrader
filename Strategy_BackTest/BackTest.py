import pandas as pd
import datetime
import yfinance as yf
import backtrader as bt
import numpy as np 
import warnings
import riskfolio as rp
import requests
import matplotlib.pyplot as plt
import qgrid
import plotly.graph_objects as go
from calendar import monthrange
from dateutil.relativedelta import relativedelta
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# Date range
Start = '2022-12-01'
End = '2023-01-31'
counter = 3

start = Start
end = End

date1 = datetime.datetime.strptime(Start, "%Y-%m-%d")
date2 = datetime.datetime.strptime(End, "%Y-%m-%d")
diff = relativedelta(date2, date1)

months_between = (diff.years)*12 + diff.months + 1
print(months_between)
# Tickers of assets

Model='Classic' # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
Rm = 'MV' # Risk measure used, this time will be variance
Obj = 'Sharpe' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
Hist = True # Use historical scenarios for risk measures that depend on scenarios
Rf = 0.04 # Risk free rate
L = 1 # Risk aversion factor, only useful when obj is 'Utility'
Points = 50 # Number of points of the frontier
method_mu ='hist' # Method to estimate expected returns based on historical data.
method_cov ='hist' # Method to estimate covariance matrix based on historical data.

def excel_download():
    holdings_url = "https://github.com/ra6it/RiskParity/blob/main/RiskParity_Holdings_Constraints.xlsx?raw=true"
    holdings_url = requests.get(holdings_url).content
    assets = pd.read_excel(holdings_url,'Holdings',usecols="A:B", engine='openpyxl')
    assets = assets.reindex(columns=['Asset', 'Industry'])
    asset_classes = {'Asset': assets['Asset'].values.tolist(), 
                     'Industry': assets['Industry'].values.tolist()}
    asset_classes = pd.DataFrame(asset_classes)
    asset_classes = asset_classes.sort_values(by=['Asset'])
    asset = assets['Asset'].values.tolist()
    asset = [x for x in asset if str(x) != 'nan']
    return asset_classes, asset

asset_classes, asset = excel_download()

assets = asset
print(assets)
# Downloading data

################## Here, I need this to be in my backtesting loop, where start and end is called with each month.
df_list = []
for i in assets:
    asset_2 = yf.download(i, start=start, end=end)['Adj Close']
    df_list.append(pd.DataFrame(asset_2))

new_df = pd.concat(df_list, axis=1)
new_df.columns = assets
print(new_df)

prices = new_df

############################################################
# Calculate assets returns
############################################################

def optimize_risk_parity(Y, Ycov, counter, i):
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
            {'type': 'ineq', 'fun': lambda w: np.sum(w[:counter] > 0.05) -counter},
            {'type': 'ineq', 'fun': lambda w: np.amax(w) - 0.8},
            {'type': 'ineq', 'fun': lambda w: 3 - np.sum(w > 0)},
            ]
    bounds = [(0, 1) for i in range(n)]
    # Call the optimization solver
    res = minimize(objective, np.ones(n)/n, constraints=cons, bounds=bounds, method='SLSQP',
                   options={'disp': False, 'eps': 1e-12, 'maxiter': 100000})
    print(res.message)
    print(res.success)
    return res.x

data = prices

returns = data.pct_change()

valid_assets = asset_classes['Asset'].isin(asset)

asset_classes = asset_classes[valid_assets]

asset_classes = pd.DataFrame(asset_classes)
asset_classes = asset_classes.sort_values(by=['Asset'])

############################################################
# Building a loop that estimate optimal portfolios on
# rebalancing dates
############################################################

rms = ['MV']

rng_start = pd.date_range(start, periods=months_between, freq='MS')
ret = returns

def next_month(i):
    next_i = i + pd.Timedelta(days=31)
    next_b = pd.date_range(start=next_i, periods=1, freq='M')
    next_b = next_b[0]
    return next_i,next_b

############################################################
# Setting up empty DFs
############################################################

weights = pd.DataFrame([])
x = pd.DataFrame([])
asset_pr = pd.DataFrame([])
sum_returns = pd.DataFrame([])
we_df = pd.DataFrame([])
y_next = pd.DataFrame([])
weight_df = pd.DataFrame([])
weighted_df = pd.DataFrame([])
time_df = pd.DataFrame([])
timed_df = pd.DataFrame([])
wght = pd.DataFrame([])
merged_df = pd.DataFrame([])

for i in rng_start:
    rng_end = pd.date_range(i, periods=1, freq='M')
    for b in rng_end:
        Y = ret[i:b]
        Ycov = Y.cov()
        optimized_weights = optimize_risk_parity(Y, Ycov, counter, i)
        w = optimized_weights.round(6)

        next_i,next_b = next_month(i)
        y_next = ret[next_i:next_b]
        weight_printer = pd.DataFrame(w).T
        weight_printer.columns = Y.columns.T
        wgt = pd.concat([weight_printer.T, pd.DataFrame(rng_end, index={"Date"})]).T
        wgt = wgt.set_index(wgt["Date"])
        wght = pd.concat([wght, wgt], axis = 0)
        #Convert the returns and weightings to numpy.
        myreturns = np.dot(w, y_next.T)
        myret = pd.DataFrame(myreturns.T, index = y_next.index)
        x = pd.concat([x, myret], axis = 0)


wght.drop(columns=['Date'], axis = 1, inplace = True)
wght.drop(wght.columns[wght.sum() == 0], axis=1, inplace=True)
print(wght)

############################################################
# To normalize the charts to the same dfs.
############################################################

specific_year  = wght.index[0].year
specific_month = wght.index[0].month
specific_month_1  = wght.index[0].month - 1

############################################################
# Portfolio returns
############################################################
x.iloc[0,:] = 0
cumret = (1 + x).cumprod() * 10000
cumret = pd.DataFrame(cumret)
cumret.columns = ['Returns total']
cumret.iloc[0,:] = 10000

############################################################
# Spy returns
############################################################

SPY = prices['SPY'].dropna()
mask = (SPY.index.year == specific_year) & (SPY.index.month == specific_month)
SPY.drop(SPY[mask].index, inplace=True)
mask_2 = (SPY.index.year == specific_year) & (SPY.index.month == specific_month_1)
SPY.drop(SPY[mask_2].index, inplace=True)
SPY = SPY/SPY.iloc[0]*10000

############################################################
# Create 1 df
############################################################

merged_df = pd.merge(SPY, cumret, left_index=True, right_index=True, how='inner')
print(merged_df)

############################################################
# Plot
############################################################

fig, ax = plt.subplots()
cumret.plot(ax=ax, label='Portfolio Returns')
SPY.plot(ax=ax, label='SPY')

# Set the x-axis to show monthly ticks
ax.xaxis.set_major_locator(plt.MaxNLocator(12))

plt.legend()
#plt.show()