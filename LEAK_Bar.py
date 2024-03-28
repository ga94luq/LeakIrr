from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots

chart_types = [
    {'label': 'Scatter Plot', 'value': 'scatter'},
    {'label': 'Bar Chart', 'value': 'bar'}
]
from tkinter import filedialog
df = pd.read_csv(filedialog.askopenfilename())

df['Bezeichnung'] = 'SOC:' + df['SOC'].astype(str) + '% SiO:' + df['SiO'].astype(str) + '% D:' + df['D'].astype(str)
data = df

app = Dash(__name__)
server = app.server

dropdown_options = [{'label': col, 'value': col} for col in df.columns]

initial_min_y = df['Q_irr'].min()
LowerGrenze = math.ceil(initial_min_y / 10.0) * 10
initial_max_y = df['Q_irr'].max()
UpperGrenze = math.ceil(initial_max_y / 10) * 10

app.layout = html.Div(style={'backgroundColor': 'white'}, children=[
    html.H4('Auswertungstool'),
    dcc.Dropdown(
        id='chart-type-dropdown',
        options=[
            {'label': 'Scatter Plot', 'value': 'scatter'},
            {'label': 'Line Plot', 'value': 'line'},
            {'label': 'Bar Chart', 'value': 'bar'}
        ],
        value='bar',
        clearable=False
    ),
    html.Div([
        html.H6("Achsenbeschriftung ändern"),
        dcc.Input(id="x-axis-title", type="text", placeholder="X-Achsenbeschriftung"),
        dcc.Input(id="y-axis-title", type="text", placeholder="Y-Achsenbeschriftung"),
        dcc.Input(id="title", type="text", placeholder="Titel des Plots"),
        html.Button(id="submit-title", n_clicks=0, children="Titel anwenden"),
    ]),
    dcc.Graph(id="chart-graph"),
    html.P("Auswahl der Durchläufe"),
    dcc.RangeSlider(
        id='range-slider-x',
        min=1, max=6, step=1,
        marks={i: str(i) for i in range(1, 7)},
        value=[1, 5],
        className='slider-x',
        tooltip={"placement": "bottom", "always_visible": False}
    ),
    html.Div([
        html.H6("Y-Achsenbereich"),
        dcc.RangeSlider(
            id='range-slider-y',
            min=initial_min_y,
            max=initial_max_y,
            step=0.05,
            marks={LowerGrenze: str(LowerGrenze), UpperGrenze: str(UpperGrenze)},
            value=[initial_min_y, initial_max_y],
            className='slider-y',
            tooltip={"placement": "bottom", "always_visible": True}
        ),
    ]),
    html.Div([
        html.H6("Spaltenauswahl"),
        dcc.Dropdown(
            id='x-axis-dropdown',
            options=dropdown_options,
            value='D',
            clearable=False
        ),
        dcc.Dropdown(
            id='Y_Achsen',
            options=dropdown_options,
            multi=True,
            value=['Q_irr'],
            clearable=False
        ),
    ]),
    html.Div([
        html.H6("Farbauswahl"),
        dcc.Dropdown(
            id='color-dropdown',
            options=["SOC", "SiO", "D", "Bezeichnung", "Typ"],#,[{'label': col, 'value': col} for col in df.columns],
            value='SOC',
            clearable=False
        ),
    ]),
    html.Div([
        html.H6("Reihenauswahl"),
        dcc.Dropdown(
            id='Reihen-dropdown',
            options=["SOC", "SiO", "D", "Bezeichnung", 'Typ'],
            value='SOC',
            clearable=False
        ),
    ]),
    html.Div([
        html.H6("Spaltenauswahl"),
        dcc.Dropdown(
            id='Spalten-dropdown',
            options=["SOC", "SiO", "D", "Bezeichnung", 'Typ'],
            value='SOC',
            clearable=False
        ),
    ]),
    html.Div([
        html.H6("Y-Achsen in einen Plot"),
        dcc.RadioItems(
            id='marker-radio',
            options=[
                {'label': 'Ja', 'value': True},
                {'label': 'Nein', 'value': False}
            ],
            value=False,
            labelStyle={'display': 'block'}
        ),
    ]),
    html.Div([
        html.H6("Werte für SOC auswählen"),
        dcc.Checklist(
            id='soc-checklist',
            options=[
                {'label': '10%', 'value': 10},
                {'label': '30%', 'value': 30},
                {'label': '50%', 'value': 50},
                {'label': '70%', 'value': 70},
                {'label': '90%', 'value': 90}
            ],
            value=[50],
            labelStyle={'display': 'block'}
        ),
    ]),
    html.Div([
        html.H6("Werte für SiO auswählen"),
        dcc.Checklist(
            id='sio-checklist',
            options=[
                {'label': '0%', 'value': 0},
                {'label': '10%', 'value': 10},
                {'label': '15%', 'value': 15}
            ],
            value=[15],
            labelStyle={'display': 'block'}
        ),
    ]),
    html.Div([
        html.H6("Werte für Bezeichnung auswählen"),
        dcc.Checklist(
            id='bezeichnung-checklist',
            options=[{'label': label, 'value': label} for label in df['Bezeichnung'].unique()],
            value=['SOC:50% SiO:15% D:1', 'SOC:50% SiO:15% D:2','SOC:50% SiO:15% D:3', 'SOC:50% SiO:15% D:4','SOC:50% SiO:15% D:5]'],
            labelStyle={'display': 'block'}
        ),
    ]),

])

@app.callback(
    [Output("chart-graph", "figure"),
     Output("range-slider-y", "min"),
     Output("range-slider-y", "max"),
     Output("range-slider-y", "marks"),
     Output('bezeichnung-checklist', 'value'),
     Output('bezeichnung-checklist', 'options')],
    [Input("range-slider-x", "value"),
     Input("range-slider-y", "value"),
     Input('x-axis-dropdown', 'value'),
     Input('Y_Achsen', 'value'),
     Input('color-dropdown', 'value'),
     Input('Reihen-dropdown', 'value'),
     Input('Spalten-dropdown', 'value'),
     Input('soc-checklist', 'value'),
     Input('sio-checklist', 'value'),
     Input('chart-type-dropdown', 'value'),
     Input('bezeichnung-checklist', 'value'),
     Input('marker-radio', 'value'),
     Input('x-axis-title', 'value'),
     Input('y-axis-title', 'value'),
     Input('title', 'value')]
)

def update_bar_chart(x_range, y_range, x_column, y_columns, color_column, Reihencolumn, Spaltencolumn, soc_values, sio_values,
                     chart_type, bezeichnung_values, Oneplot, xTitle, yTitle, Title):

    df = data
    Min = 0
    Max = 0

    for col in y_columns:
        if df[col].min() < Min:
            Min = df[col].min()
            Min_col = col
        if df[col].max() > Max:
            Max = df[col].max()
            Max_col = col

    min_y = df[y_columns].min().min()
    LowerGrenze = math.floor(min_y / 10.0) * 10
    max_y = df[y_columns].max().max()
    UpperGrenze = math.ceil(max_y / 10) * 10

    low_x, high_x = x_range
    low_y, high_y = y_range

    df_1 = df[df['SOC'].isin(soc_values) & df['SiO'].isin(sio_values) ]
    df_1 = df_1[df_1['D'] >= low_x]
    df_1 = df_1[df_1['D'] <= high_x]
    Bez_values = df_1['Bezeichnung'].unique()
    bezeichnung_values = Bez_values

    mask_x = (df['D'] >= low_x) & (df['D'] <= high_x)
    mask_y = (df[y_columns] >= low_y) & (df[y_columns] <= high_y)
    for col in y_columns:
        mask_y = (df[col] >= low_y) & (df[col] <= high_y)


    df = df[df['SOC'].isin(soc_values) & df['SiO'].isin(sio_values) & df['Bezeichnung'].isin(bezeichnung_values) & mask_y]
    df[color_column] = df[color_column].astype(str)

    step_size = 10
    NumberofSteps = int((UpperGrenze - LowerGrenze) / step_size) + 1
    marks = {LowerGrenze + i * step_size: str(round(LowerGrenze + i * step_size, 2)) for i in range(NumberofSteps)}

    if chart_type=='line':
        fig = px.line(df[mask_x & mask_y], x=x_column, y=y_columns[0],
                        color=color_column, hover_data=['Bezeichnung'], width=2000, height=800)

    if chart_type=='bar':
            if Oneplot:
                fig = go.Figure()
                for yaxis in y_columns:
                    fig.add_trace(go.Bar(x=df[x_column],
                                         y=df[yaxis],
                                         name=yaxis))

                #fig.update_layout(barmode='group')
            else:
                df_2 = pd.DataFrame()
                for yxis in y_columns:
                    df_1 = df[["SOC", "SiO", "D", "Bezeichnung",yxis]]
                    df_1['Values'] = df_1[yxis]
                    df_1[yxis] = yxis
                    df_1.rename(columns={yxis : 'Typ'}, inplace=True)
                    df_2 = pd.concat([df_2, df_1])
                df_N= df_2
                df_N['Typ'] = df_N['Typ'].astype(str)
                df_N = df_N.fillna(0)
                df_N= df_N.reset_index()
                fig = px.bar(df_N,
                        x=x_column,
                        y='Values',
                       color=color_column,
                       barmode='group',
                       facet_row=Reihencolumn, facet_col=Spaltencolumn,
                        width=2000,
                        height=800, hover_data='Values',  text='Values')


                fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
                fig.update_layout(uniformtext_minsize=14, uniformtext_mode='hide')
                fig.update_layout(barmode='group')

    if xTitle:
        fig.update_yaxes(title=xTitle)
    if yTitle:
        fig.update_xaxes(title=yTitle)
    if Title:
        fig.update_layout(title=Title)

    large_rockwell_template = dict(
        layout=go.Layout(
            title_font=dict(family="Arial", size=24),
            xaxis=dict(showline=True, zeroline=False, linecolor='black'),
            yaxis=dict(showline=True, zeroline=False, linecolor='black'),
            plot_bgcolor='white',
        )
    )
    fig.update_layout(template=large_rockwell_template)

    return fig, LowerGrenze, UpperGrenze, marks, bezeichnung_values, [{'label': label, 'value': label} for label in Bez_values]


if __name__ == '__main__':
    app.run_server(debug=True)
