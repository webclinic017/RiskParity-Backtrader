import pandas as pd
import yfinance as yf
import requests
from calendar import monthrange
from scipy.optimize import minimize
from Utils import Start, End
##

def data_management(start, end):
    holdings_url = "https://github.com/ra6it/RiskParity/blob/main/RiskParity_Holdings_Constraints.xlsx?raw=true"
    assets = pd.read_excel(holdings_url, 'Holdings', usecols="A:B", engine='openpyxl')
    assets = assets.dropna(subset=['Asset'])
    assets = assets.rename(columns={'Asset': 'Ticker'})
    
    asset_classes = assets.sort_values(by='Ticker')[['Ticker', 'Industry', 'Full_name']]
    asset_classes.reset_index(drop=True, inplace=True)
    
    asset_list = asset_classes['Ticker'].tolist()
    df_list = []
    
    for asset in asset_list:
        asset_data = yf.download(asset, start=start, end=end, interval="1mo")['Adj Close']
        df_list.append(pd.DataFrame(asset_data))
    
    prices = pd.concat(df_list, axis=1)
    prices.columns = asset_list
    
    valid_assets = asset_classes['Ticker'].isin(asset_list)
    asset_classes = asset_classes[valid_assets]
    asset_classes.reset_index(drop=True, inplace=True)
    
    returns = prices
    
    return prices, asset_classes, returns

prices, asset_classes, returns = data_management(Start, End)

# Print the results
print("Prices:")
print(prices)
print("\nAsset Classes:")
print(asset_classes)
print("\nReturns:")
print(returns)