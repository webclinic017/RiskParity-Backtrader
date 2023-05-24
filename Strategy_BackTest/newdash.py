from OptimizerBackTest import optimizerbacktest, ret
import quantstats as qs
from Trend_Following import dummy_L_df, ret as daily_returns, asset_classes, asset
import pandas as pd
from datamanagement import *
from Utils import bench

monthly_returns, asset_classes, asset, assets = data_management(Start, End, '1mo')

monthly_returns = monthly_returns.dropna()

monthly_returns_log = ret(monthly_returns)
daily_returns_log   = ret(daily_returns)

mweight  = 0
nmore    = 0.2

assets        = assets.sort_values(by=['Industry', 'Asset'], key=lambda x: x.str.lower())
assets        = assets.reset_index(drop=True)
df_2 = pd.DataFrame(columns = assets['Asset'].to_list())

asset_pick = assets['Industry'].to_list()

bonds       = asset_pick.index("Bonds")
commodities = asset_pick.index("Commodities")
defense     = asset_pick.index("Defense")
energies    = asset_pick.index("Energies")
equities    = asset_pick.index("Equities")
housing     = asset_pick.index("Housing")
metals      = asset_pick.index("Metals")

# Max weightings:

# I think this needs to be dynamicly adjusted.
max_ind_weights = {
    "Bonds": 0.5,
    "Commodities": 1,
    "Defense": 1,
    "Energies": 0.2,
    "Equities": 1,
    "Housing": 1,
    "Metals": 0.8
}

assets['Max_Weight'] = assets['Industry'].map(max_ind_weights)

asset_constraint = assets.copy()

print(dummy_L_df)

portfolio_return_concat, weight_concat, sharpe_array_concat = optimizerbacktest(daily_returns_log, dummy_L_df, daily_returns_log, nmore, mweight, monthly_returns_log, asset_constraint)
weight_concat = weight_concat.sort_index(axis=1)
weight_concat = weight_concat.reindex(columns=df_2.columns)

leng         = len(asset_pick)

benchmark = 'Benchmark'
Bench, Merged_df = bench(portfolio_return_concat.index.min(), benchmark, portfolio_return_concat)

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
    {"selector": f".col{idx}", "props": [("border-right", "1px solid brown"), ("border-left", "1px solid grey")]}
    for idx in range(commodities, defense)
] + [
    {"selector": f".col{idx}", "props": [("border-right", "1px solid grey"), ("border-left", "1px solid orange")]}
    for idx in range(defense, energies)
] + [
    {"selector": f".col{idx}", "props": [("border-right", "1px solid orange"), ("border-left", "1px solid green")]}
    for idx in range(energies, equities)
] + [
    {"selector": f".col{idx}", "props": [("border-right", "1px solid green"), ("border-left", "1px solid brown")]}
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

with open(f'F:\outputs\RiskParity-tearsheet_{Start}-{End}.html','w') as file: #({nmore})_maxweight({mweight})
    file.write(modified_content)
