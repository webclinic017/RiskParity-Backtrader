import pandas as pd
import yfinance as yf
import requests
from calendar import monthrange
from scipy.optimize import minimize

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

def datamanagement_1(start, end):
    asset_classes, asset = excel_download()
    df_list = []
    asset = list(set(asset))
    print(asset)
    tickers = yf.Tickers(asset)

    for asset, ticker in zip(asset, tickers.tickers):
        asset_2 = yf.download(asset, start=start, end=end)['Adj Close']
        
        ticker = ticker.fast_info
        df_list.append(pd.DataFrame(asset_2))
        asset_2 = pd.DataFrame(asset_2)
        print(ticker)
    prices = pd.concat(df_list, axis=1)
    prices.columns = asset
    return prices, asset_classes, asset

def data_management_2(prices, asset_classes, asset):
    returns = prices
    valid_assets = asset_classes['Asset'].isin(asset)
    asset_classes = asset_classes[valid_assets]
    asset_classes = pd.DataFrame(asset_classes)
    asset_classes = asset_classes.sort_values(by=['Asset'])
    return returns