from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
from datetime import date

Start = '2022-01-01'
End = date.today().strftime("%Y-%m-%d")

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
