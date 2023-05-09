#Start with a MA.
import pandas as pd
from datetime import datetime
from datamanagement import *
import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
from Utils import Start, End

number_of_iter = 1000
long    = 200
medium  = 100
short   = 30

ret, asset_classes, asset, assets = data_management(Start, End, '1d')

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
