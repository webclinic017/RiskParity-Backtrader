#Start with a MA.
import pandas as pd
from datetime import datetime
from datamanagement import *
import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
Start = '2022-01-01'
End = date.today().strftime("%Y-%m-%d")
number_of_iter = 1000
long    = 200
medium  = 100
short   = 30

date1 = datetime.strptime(Start, "%Y-%m-%d")
date2 = datetime.strptime(End, "%Y-%m-%d")
diff = relativedelta(date2, date1)
Start_bench = date1 + relativedelta(months=1)
months_between = (diff.years)*12 + diff.months + 1
rng_start = pd.date_range(Start, periods=months_between, freq='MS')

def asset_trimmer(b, trend_df, Y):
    df_split_monthly = trend_df[b:b]
    cols_to_drop = [col for col in df_split_monthly.columns if df_split_monthly[col].max() < 0.8]
    Y = Y.drop(columns=cols_to_drop)
    return Y

def next_month(i):
    i_str = i.strftime('%Y-%m')
    dt = datetime.strptime(i_str, '%Y-%m')
    next_month = dt + relativedelta(months=1)
    next_i = datetime(next_month.year, next_month.month, 1)
    next_b = pd.date_range(start=next_i, periods=1, freq='M')
    next_b = next_b[0]
    return next_i,next_b

def portfolio_returns(w, Y_adjusted_next, i, oldw):
    'there is a bug here, I need the column to all be there. till next time.'
    w = w.drop('sharpe', axis=1)
    w_arr = np.array(w)

    if "VTI" in w and (w['VTI'] == 0.6).any and "VTI" not in Y_adjusted_next.columns:
        Y_adjusted_next['VTI'] = yf.download("VTI", start=Start, end=End)['Adj Close'].pct_change()
    if "BIL" in w and (w['BIL'] == 0.4).any and "BND" not in Y_adjusted_next.columns:
        Y_adjusted_next['BIL'] = yf.download("BND", start=Start, end=End)['Adj Close'].pct_change()


    #Y_adjusted_next = np.array(Y_adjusted_next)
    df_daily_return = w_arr*Y_adjusted_next
    df_portfolio_return = pd.DataFrame(df_daily_return.sum(axis=1), columns=['portfolio_return'])
    return df_portfolio_return

prices, asset_classes, asset = datamanagement_1(Start, End)
ret = data_management_2(prices, asset_classes, asset).dropna()

def calculate_rolling_average(ret, days):
    ret = ret.dropna()
    rolling_df = pd.DataFrame()
    for column in ret.columns:
        rolling_df[column] = ret[column].rolling(window=200).mean()
    rolling_df = dummy_sma(rolling_df, ret)
    return rolling_df

# Now we need to use this to determine what assets to hold. So for each month, we need to know if the asset is trending.
# To do this, I think for each day, we can have a dummy, 1 for above sma and 2 for below sma.

def dummy_sma(rolling_df, ret):
    dummy_L_df = pd.DataFrame(index=rolling_df.index)
    for asset_name in rolling_df.columns:
    # Skip non-numeric columns
        if not np.issubdtype(rolling_df[asset_name].dtype, np.number):
            continue
        # Compare the prices of the asset for each date
        dummy_L_df[asset_name] = (rolling_df[asset_name] < ret[asset_name]).astype(int)
    dummy_L_df  = dummy_L_df.resample('M').mean()

    return dummy_L_df
dummy_L_df = calculate_rolling_average(ret, 200)

def calculate_monthly_rsi(df):
    # Calculate monthly RSI for each column (i.e., asset)
    rsi_df = rsi_value_df = rsi_value = pd.DataFrame([])
    delta = pd.DataFrame([])
    time_period = 5

    for col in df.columns: 
        # Calculate monthly returns for this asset
        delta[col] = df[col].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        # Calculate the average gains and losses over the time period
        avg_gain = gain.rolling(window=time_period).mean()
        avg_loss = loss.rolling(window=time_period).mean()

        # Calculate the relative strength (RS)
        rs = avg_gain / avg_loss
        # Calculate the RSI
        rsi = 100 - (100 / (1 + rs))

        # Combine the RSI value and signal into a single DataFrame
    rsi_df = pd.concat([rsi_df, rsi], axis=1)

    # Combine RSI DataFrames for all assets into one DataFrame
    return rsi_df

rsi_df = calculate_monthly_rsi(ret)
# Now, if the row for a specific contract is <0, then we can exclude it from our sample set, and it is not needed. This is part of the asset selection component.
n = 5
count = rsi_df.groupby(pd.Grouper(freq='M')).apply(lambda x: (x > 70).sum())
new_cool_df = count.where(count <= n, 1).where(count > n, 0).resample('M').last()
