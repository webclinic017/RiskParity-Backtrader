import numpy as np 
import warnings
from calendar import monthrange
import plotly.graph_objs as go
from Utils import *
from Trend_Following import *
import numpy as np
import dash_table
import dash
import dash_core_components as dcc
from dash import html
from Utils import bench
from OptimizerBackTest import old_sharpe, Merged_df, Port_sh, Bench,  Bench_sh, portfolio_return_concat, weight_concat, asset_classes, sharpe_array_concat, bonds, commodities, defense, leng, energies, equities, housing, metals

warnings.filterwarnings("ignore")

benchmark = 'Bench_Return'

weight_concat, this_month_weight = output_mgmt(weight_concat)
weight_concat.dropna()

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

def generate_weights_table_table(weights_df, asset_classes, ider):
    weight_long = long_names(asset_classes, weights_df)
    weights_df2 = weights_df.copy()
    weights_df, weight_long = df_merger(weights_df, weight_long)
    weights_df2 = weights_df2.round(3)
    weights_df2['Index'] = weights_df2.index
    weights_df2['Index'] = pd.to_datetime(weights_df2['Index'])
    weights_df2['Index'] = weights_df2['Index'].dt.strftime('%Y-%m-%d')
    # I need to add 0 to the empty columns, or drop?
    weights_df2 = weights_df2[['Index'] + [col for col in weights_df2.columns if col != 'Index']]
    weights_table = dash_table.DataTable(
        id = ider,
        data=weights_df2.to_dict('records'),
        columns=[{'name': col, 'id': col} for col in weights_df2.columns],
        tooltip_data=[{weights_df2.columns[i+1]: {'value': str(weight_long.columns[i] + " (" + weights_df2.columns[i+1] + ")"),'type': 'markdown'}
                for i, column in enumerate(weights_df2.columns[1:])  # Exclude the first column
            }
            for _ in weights_df2.iterrows()
        ],
        css=[{
        'selector': '.dash-table-tooltip',
        'rule': 'background-color: lightgrey; font-family: monospace; color: black, width: 10px; height: 50px'
        }],
        tooltip_delay=0,
        tooltip_duration=None,
        style_cell={'font-family': 'Arial', 'font-size': '10px', 'font-color':'black', 'minWidth': '1px', 'width': '2px', 'maxWidth': '2px'},
        fixed_rows={'headers': True},
        style_header={'position': 'sticky','top': '0','background-color': 'grey', 'color': 'white','font-size': '16px', 'font-family': 'Arial', 'font-size': '10px','fontColor':'black', 'fontWeight': 'bold'},
        style_table = {'overflowX': 'auto',
                        'overflowY': 'scroll',
                        'border': '1px solid grey',
                        'borderRadius': '5px',
                        'padding': '5px'
        },
        style_data_conditional=
            [{'if': {'column_id': col},'border': '1px solid magenta',}for col in weights_df2.columns[bonds+1:commodities+1]] +
            [{'if': {'column_id': col},'border': '1px solid brown',}for col in weights_df2.columns[commodities+1:defense+1]] + 
            [{'if': {'column_id': col},'border': '1px solid grey',}for col in weights_df2.columns[defense+1:energies+1]] + 
            [{'if': {'column_id': col},'border': '1px solid orange',}for col in weights_df2.columns[energies+1:equities+1]] + 
            [{'if': {'column_id': col},'border': '1px solid green',}for col in weights_df2.columns[equities+1:housing+1]] + 
            [{'if': {'column_id': col},'border': '1px solid brown',}for col in weights_df2.columns[housing+1:metals+1]] + 
            [{'if': {'column_id': col},'border': '1px solid gold',}for col in weights_df2.columns[metals+1:leng+1]]+
            [{'if': {'column_id': col, 'filter_query': '{{{}}} > 0.05'.format(col)}, 'backgroundColor': '#D6FF97', 'color': 'black'}
            for col in weights_df2.columns] + 
            [{'if': {'column_id': col, 'filter_query': '{{{}}} > 0.1'.format(col)}, 'backgroundColor': '#6FD17A', 'color': 'black'}
            for col in weights_df2.columns] + 
            [{'if': {'column_id': col, 'filter_query': '{{{}}} > 0.2'.format(col)}, 'backgroundColor': '#9ACD32', 'color': 'black'}
            for col in weights_df2.columns] + 
            [{'if': {'column_id': col, 'filter_query': '{{{}}} > 0.5'.format(col)}, 'backgroundColor': '#0DBF00', 'color': 'black'}
            for col in weights_df2.columns] +        
            [{'if': {'column_id': col}, 'textAlign': 'center'} for col in weights_df2.columns]+
            [{'if': {'column_id': weights_df2.columns[0]},'fontWeight': 'bold', 'font-size': '10px', 'minWidth': '10px', 'maxWidth': '10px'}])
    return weights_table

def portfolio_data(df, col, num_days, average_number_days, portfoliotreturn):
    last_month_returns = df.mean()
    last_month_std_returns = df.std()
    last_month_sharpe_ratio = np.sqrt(240) * (last_month_returns / last_month_std_returns)
    last_month_sharpe_ratio = last_month_sharpe_ratio.astype(np.float64).values
    Net_Returns = 1
    std = 1
    print(last_month_sharpe_ratio)
    return Net_Returns == 1, std, last_month_sharpe_ratio

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

def portfolio_returns_app(returns_df, weights_df, this_month_weight, Bench, sharpe_array, portfolio_return_concat):
    num_years = (returns_df.index.max() - returns_df.index.min()).days / 365
    num_days = len(returns_df)
    average_number_days = num_days/num_years
    returns = returns_df.pct_change()
    bench_pct = Bench.pct_change()
    bench_pct.dropna(inplace=True)
    returns.dropna(inplace=True)
    
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
                html.Td(round((Merged_df.iloc[-1]-10000)/10000, 4)),
                html.Td(round((Bench.iloc[-1]-10000)/10000, 4)),
            ]),
            html.Tr(children=[
                html.Td('Avg Yr Returns'),
                html.Td(round((Merged_df.iloc[-1]-10000)/10000 / num_years, 4)),
                html.Td(round((Bench.iloc[-1]-10000)/10000 / num_years, 4))
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

    generate_weights_table_table(weights_df, asset_classes, 'weights'),

    html.H2(children="Next Month Weights"),
    
    generate_weights_table_table(this_month_weight, asset_classes, 'nmweights'),

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

app = portfolio_returns_app(Merged_df, weight_concat, this_month_weight, Bench, sharpe_array_concat, portfolio_return_concat)
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

