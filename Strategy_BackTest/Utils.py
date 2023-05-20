from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
from datetime import date
import numpy as np
import yfinance as yf

Start = '2013-01-01'
End = date.today().strftime("%Y-%m-%d")

date1 = datetime.strptime(Start, "%Y-%m-%d")
date2 = datetime.strptime(End, "%Y-%m-%d")
diff = relativedelta(date2, date1)
Start_bench = date1 + relativedelta(months=1)
months_between = (diff.years)*12 + diff.months + 1
rng_start = pd.date_range(Start, periods=months_between, freq='MS')

def asset_trimmer(df_split_monthly, Y):
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

def weightings(w, Y_adjusted, i, weight_concat, sharpe_array_concat, sharpe_ratio, b):
    w_df = pd.DataFrame(w)#.T
    w_df = w_df.T
    w_df.columns = Y_adjusted.columns
    if b == False:
        w_df['VTI']    =   0.6
        w_df['BND']    =   0.4
    w_df['date'] = w_df.index
    w_df['date'] = i
    w_df.set_index('date', inplace=True)
    sharpe_array = w_df
    sharpe_array['sharpe'] = sharpe_ratio
    sharpe_array_concat = pd.concat([sharpe_array_concat, sharpe_array])
    weight_concat = pd.concat([weight_concat,w_df]).fillna(0)
    return weight_concat, w_df, sharpe_array_concat

def output_mgmt(weight_concat):
    #weight_concat.drop('sharpe', axis=1, inplace=True)

    this_month_weight = weight_concat.iloc[-1]
    this_month_weight = pd.DataFrame([this_month_weight])
    weight_concat = weight_concat.drop(index=weight_concat.index[-1])
    return weight_concat, this_month_weight

def bench(Bench_start, benchmark, portfolio_return_concat):
    Bench_W = Bench = pd.DataFrame([])
    print(benchmark)
    benchasset = ['VTI','BND']
    for i in benchasset:
        if i == 'VTI':
            Bench_W = yf.download(i, start=Bench_start, end=End)['Adj Close'].pct_change() * 0.6
        else:
            Bench_W = yf.download(i, start=Bench_start, end=End)['Adj Close'].pct_change() * 0.4
        Bench = pd.concat([Bench, Bench_W], axis=1)
    print(benchmark)
    Bench = pd.DataFrame(pd.DataFrame(Bench)).sum(axis=1)
    Bench.iloc[0] = 0
    Bench = (1 + Bench).cumprod() * 10000
    Bench = pd.DataFrame(Bench)
    Bench.columns = ['Bench_Return']

    merged_df = portfolio_return_concat
    merged_df.iloc[0] = 0
    merged_df = (1 + merged_df).cumprod() * 10000
    return Bench, merged_df
