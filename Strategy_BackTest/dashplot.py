import numpy as np 
import warnings
from calendar import monthrange
import plotly.graph_objs as go
from Utils import *
from Trend_Following import *
import numpy as np
import dash
import dash_core_components as dcc
from dash import html
from Utils import bench
from OptimizerBackTest import portfolio_return_concat, portfolio_return_concat, weight_concat, asset_classes, sharpe_array_concat

warnings.filterwarnings("ignore")

benchmark = 'Bench_Return'

Bench, merged_df = bench(portfolio_return_concat.index.min(), benchmark, portfolio_return_concat)


weight_concat, this_month_weight = output_mgmt(weight_concat)

def correlation_matrix(sharpe_array, column):
    corr_matrix = sharpe_array.corr()
    corr_matrix = corr_matrix[f'{column}']
    return corr_matrix

def long_names(asset_classes, weight):
    mapping_dict = dict(zip(asset_classes['Asset'], asset_classes['Full_name']))
    weight_long = weight.rename(columns=mapping_dict)
    return weight_long

# Generate the table of weights
def df_merger(weights_df, weight_long):
    for asset_df, asset_long in zip(weights_df, weight_long):
        column_name = f"{asset_long}"
        weights_df.rename(columns={asset_df: column_name}, inplace=True)
        weight_long.rename(columns={asset_long: column_name}, inplace=True)
    return weights_df, weight_long

def generate_weights_table(weights_df, asset_classes):
    weight_long = long_names(asset_classes, weights_df)
    weights_df2 = weights_df.copy()
    weights_df, weight_long = df_merger(weights_df, weight_long)
    print(weights_df)
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
                                     'background-color': '#0DBF00' #if weights_df.loc[index, col] > 0.5 
                                       #else '#9ACD32' if weights_df.loc[index, col] > 0.2 
                                       #else '#6FD17A' if weights_df.loc[index, col] > 0.1
                                       #else '#D6FF97' if weights_df.loc[index, col] > 0.04
                                       #else 'white',
                                       },
                                       title=col,
                                ) for col in weights_df.columns],
                ]
            ) for index in weights_df.index.strftime('%Y-%m-%d')]
        ]
    )
    return weights_table

def portfolio_data(df, col, num_days, average_number_days):
    Net_Returns = df[f'{col}'].mean()* num_days
    Average_Returns = df[f'{col}'].mean() * average_number_days
    std = df[f'{col}'].std() * average_number_days
    Sharpe_Ratio =  np.sqrt(average_number_days) * (Average_Returns / std)
    return Net_Returns, std, Sharpe_Ratio

def last_month_data(df, col):
    last_month_returns = df.resample('M').mean().iloc[-2]
    last_month_std_returns = df.resample('M').std().iloc[-2]
    last_month_sharpe_ratio = np.sqrt(12) * (last_month_returns / last_month_std_returns)
    last_month_sharpe_ratio = last_month_sharpe_ratio.astype(np.float64).values
    return last_month_sharpe_ratio

def frontier_chart(vol_arr, ret_arr, sharpe_arr, selected_index):
    vol_arr = vol_arr.T
    ret_arr = ret_arr.T
    sharpe_arr = sharpe_arr.T
    trace = go.Scatter(x=vol_arr[selected_index], 
                        y=ret_arr[selected_index], 
                        mode='markers', 
                        marker=dict(color=sharpe_arr[selected_index], 
                        colorscale='Viridis', 
                        size=8),
                        name = 'Feasable Set')

    layout = go.Layout(xaxis_title='Volatility',
                        yaxis_title='Returns',
                        coloraxis_colorbar=dict(title='Sharpe Ratio'))
    sharpe_max = sharpe_arr[selected_index].max()
    location = sharpe_arr[sharpe_arr == sharpe_max].stack().index#.tolist()
    ret_max = ret_arr[selected_index]
    vol_max = vol_arr[selected_index]
    location = location[0][0]
    max_ret = ret_max[location]
    max_vol = vol_max[location]
    frontier = go.Figure(data=[trace], layout=layout)
    frontier.add_trace(go.Scatter(x=[max_vol], y=[max_ret], mode = 'markers',
                         marker_symbol = 'circle',
                         marker_size = 10, name = 'Portfolio Estimate'))
    return frontier

def portfolio_returns_app(returns_df, weights_df, this_month_weight, Bench, sharpe_array):
    num_years = (returns_df.index.max() - returns_df.index.min()).days / 365
    num_days = len(returns_df)
    average_number_days = num_days/num_years
    returns = returns_df.pct_change()
    returns.dropna(inplace=True)
    Portfolio_Net_Returns, Portfolio_std, Portfolio_Sharpe_Ratio = portfolio_data(returns, 'portfolio_return', num_days, average_number_days)
    last_month_sharpe_ratio = last_month_data(returns, 'portfolio_return')
    Bench_Net_Returns, Bench_std, Bench_Sharpe_Ratio = portfolio_data(Bench.pct_change(), f'{benchmark}', num_days, average_number_days)
    last_month_sharpe_ratio_bench = last_month_data(Bench.pct_change(), f'{benchmark}')
 
    fig = go.Figure()

    returns_df = returns_df.sort_index(ascending=False)
    fig.add_trace(go.Scatter(x=returns_df.index, y=returns_df['portfolio_return'], mode='lines', name='Portfolio Return'))
    fig.add_trace(go.Scatter(x=Bench.index, y=Bench[f'{benchmark}'], mode='lines', name=f'{benchmark}'))

    corr_matrix = correlation_matrix(sharpe_array, 'sharpe')
    corr_matrix = corr_matrix.to_frame()
    corr_matrix = corr_matrix.sort_values(by='sharpe', ascending=True)
    corr_matrix_long = long_names(asset_classes, corr_matrix.T).T
    corr_matrix, corr_matrix_long = df_merger(corr_matrix, corr_matrix_long)

    data = [
        go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix_long.index,
            colorscale='RdBu',
            hoverongaps=False,
            hovertemplate='%{y} <br>Correlation: %{z:.2f}<br>',
            showscale=True,
            zmin=-1,
            zmax=1,
            text=corr_matrix.round(2).values.astype(str),
            texttemplate="%{text}",
            textfont={"size":10},
            name = ''
        )
    ]

    layout = go.Layout(
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(corr_matrix.index))),
            ticktext=corr_matrix.index.to_list()
        )
    )
    # Create a table of summary statistics for portfolio and benchmark returns
    returns_table = html.Table(children=[
            html.Tr(children=[
                html.Th('Statistic'),
                html.Th('Portfolio'),
                html.Th(benchmark)
            ]),
            html.Tr(children=[
                html.Td('Net Returns'),
                html.Td(round(Portfolio_Net_Returns, 4)),
                html.Td(round(Bench_Net_Returns, 4)),
            ]),
            html.Tr(children=[
                html.Td('Avg Yr Returns'),
                html.Td(round(Portfolio_Net_Returns / num_years, 4)),
                html.Td(round(Bench_Net_Returns / num_years, 4))
            ]),
            html.Tr(children=[
                html.Td('Std Returns'),
                html.Td(round(Portfolio_std, 4)),
                html.Td(round(Bench_std, 4))
            ]),
            html.Tr(children=[
                html.Td('Sharpe Ratio'),
                html.Td(str(round(Portfolio_Sharpe_Ratio, 4))),
                html.Td(str(round(Bench_Sharpe_Ratio, 4)))
            ]),
            html.Tr(children=[
                html.Td('L/M sharpe Ratio'),
                html.Td(str(round(float(last_month_sharpe_ratio), 4))),
                html.Td(str(round(float(last_month_sharpe_ratio_bench), 4)))
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
    
    html.Div(children=[
        html.Div(children=[
            html.H2(children='Sharpe Ratio Correlation Matrix'),
            dcc.Graph(
                id='correlation-matrix',
                figure={'data': data, 'layout': layout},
                style={
                    'width': '40vh',
                    'height': '90vh',
                    'font-family': 'Arial',
                    'font-size': '12px',
                },
            ),
        ], style={'flex': '1'})
                
    ])
        ])

    return app

app = portfolio_returns_app(merged_df, weight_concat, this_month_weight, Bench, sharpe_array_concat)
app.run_server(debug=False)


'''
    html.Div(children=[
            html.H3(children='Efficient Frontier', style={'font-size': '24px'}),
            dcc.Dropdown(
                id='vol-dropdown',
                options=ret_arr_list,
                value=ret_arr_list[0],
                style={'width': '200px',
                       'font-size': '12px'},
            ),
            dcc.Graph(
                id='efficient-frontier',
                figure=frontier_chart(vol_arr, ret_arr, sharpe_arr, ret_arr.index[0]),
            ),
        ], style={'flex': '1'}),
    ], style={'display': 'flex'}),
    ])
    @app.callback(
            Output('efficient-frontier', 'figure'),
            [Input('vol-dropdown', 'value')])
    def update_graph(selected_index):
        frontier = frontier_chart(vol_arr, ret_arr, sharpe_arr, selected_index)
        return frontier
    '''

