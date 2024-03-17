from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import pandas as pd
from datetime import datetime
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

df = pd.read_csv('https://raw.githubusercontent.com/ga94luq/LeakIrr/main/Leakage.csv')

chart_types = [
    {'label': 'Scatter Plot', 'value': 'scatter'},
    {'label': 'Bar Chart', 'value': 'bar'}
]

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
        value='line',  # Standardwert: Scatter Plot
        clearable=False
    ),
    dcc.RadioItems(
        id='y-axis-radio',
        options=[
            {'label': 'Y-Achse Automatisch setzen', 'value': False},
            {'label': 'Y-Achse manuell', 'value': True}
        ],
        value=True,  # Setze den vordefinierten Wert für die Y-Achse
        labelStyle={'display': 'block'}  # Stelle das Label unter den Radio-Button
    ),
    html.Div([
        html.H6("Achsenbeschriftung ändern"),
        dcc.Input(id="x-axis-title", type="text", placeholder="X-Achsenbeschriftung"),
        dcc.Input(id="y-axis-title", type="text", placeholder="Y-Achsenbeschriftung"),
        dcc.Input(id="title", type="text", placeholder="Titel des Plots"),
        html.Button(id="submit-title", n_clicks=0, children="Titel anwenden"),
    ]),
    html.Div(id='selected-y-axis-output'),
    dcc.Graph(id="chart-graph"),
    html.P("Auswahl der Durchläufe"),
    dcc.RangeSlider(
        id='range-slider-x',
        min=1, max=6, step=1,
        marks={i: str(i) for i in range(1, 6)},
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
        html.H6("Achsenauswahl"),
        dcc.Dropdown(
            id='x-axis-dropdown',
            options=dropdown_options,
            value='A',  # Standardwert für die x-Achse
            clearable=False
        ),
        dcc.Dropdown(
            id='Y_Achsen',
            options=dropdown_options,
            multi=True,  # Erlaubt die Auswahl mehrerer Optionen
            value=['Q_irr'],  # Standardwert für die y-Achse
            clearable=False
        ),
    ]),
    html.Div([
        html.H6("Reihenauswahl"),
        dcc.Dropdown(
            id='color-dropdown',
            options=[{'label': col, 'value': col} for col in df.columns],
            value='SOC',  # Standardwert für die Farbauswahl
            clearable=False
        ),
    ]),
    html.Div([
        html.H6("Spaltenauswahl"),
        dcc.Dropdown(
            id='symbol-dropdown',
            options= [{'label': '', 'value': ''}] + [{'label': col, 'value': col} for col in df.columns],
            value='SiO',  # Standardwert für die Symbolauswahl
            clearable=False
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
            value=[10, 30, 50, 70, 90],  # Standardmäßig alle Werte ausgewählt
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
            value=[0, 10, 15],  # Standardmäßig alle Werte ausgewählt
            labelStyle={'display': 'block'}
        ),
    ]),
    html.Div([
        html.H6("Werte für Bezeichnung auswählen"),
        dcc.Checklist(
            id='bezeichnung-checklist',
            options=[{'label': label, 'value': label} for label in df['Bezeichnung'].unique()],
            value=df['Bezeichnung'].unique(),  # Alle Einträge vorausgewählt
            labelStyle={'display': 'block'}
        ),
    ]),
])

@app.callback(
    [Output("chart-graph", "figure")],
    [Output("range-slider-y", "min"),
     Output("range-slider-y", "max"),
     Output("range-slider-y", "value"),
     Output("range-slider-y", "marks"),
     Output('color-dropdown', 'value'),
     Output('bezeichnung-checklist', 'value'),
     Output('bezeichnung-checklist', 'options')],
    [Input("range-slider-x", "value"),
     Input("range-slider-y", "value"),
     Input('x-axis-dropdown', 'value'),
     Input('Y_Achsen', 'value'),
     Input('color-dropdown', 'value'),
     Input('symbol-dropdown', 'value'),
     Input('soc-checklist', 'value'),
     Input('sio-checklist', 'value'),
     Input('chart-type-dropdown', 'value'),
     Input('bezeichnung-checklist', 'value'),
     Input('y-axis-radio', 'value'),
     Input('x-axis-title', 'value'),
     Input('y-axis-title', 'value'),
     Input('title', 'value'),]
)
def update_bar_chart(x_range, y_range, x_column, y_columns, Reihen_select, Spalten_select, soc_values, sio_values,
                     chart_type, bezeichnung_values, radio_select, xAchse_Label, yAchse_Label, title_Label):

    df = data
    df = df.fillna(0)

    min_y = df[y_columns].min().min()
    LowerGrenze = math.floor(min_y / 10.0) * 10
    max_y = df[y_columns].max().max()
    UpperGrenze = math.ceil(max_y / 10) * 10

    if Reihen_select in x_column:
        if Reihen_select=='SiO':
            Reihen_select = 'SOC'
        if Reihen_select=='SOC':
            Reihen_select ='SiO'

    low_x, high_x = x_range
    low_y, high_y = y_range

    mask_x = (df['A'] >= low_x) & (df['A'] <= high_x)
    mask_y = (df[y_columns[0]] >= low_y) & (df[y_columns[0]] <= high_y)
    for col in y_columns:
        mask = (df[col] >= low_y) & (df[col] <= high_y)
        mask_y = mask_y & mask
        if sum(mask_y)==0:
            y_range_slider_set = [LowerGrenze, UpperGrenze]
            for i in range(len(mask_y)):
                mask_y[i] = True
        else:
            y_range_slider_set = y_range



    df = df[df['SOC'].isin(soc_values) & df['SiO'].isin(sio_values) & 
            df['Bezeichnung'].isin(bezeichnung_values) & mask_x & mask_y]

    if df.shape[0]<2:
        y_range_slider_set = [LowerGrenze, UpperGrenze]
        for i in range(len(mask_y)):
            mask_y[i] = True

        df = data
        df = df[df['SOC'].isin(soc_values) & df['SiO'].isin(sio_values) &
                df['Bezeichnung'].isin(bezeichnung_values)]


    Bez_values = df['Bezeichnung'].unique()

    step_size = 10
    NumberofSteps = int((UpperGrenze - LowerGrenze) / step_size) + 1
    marks = {LowerGrenze + i * step_size: str(round(LowerGrenze + i * step_size, 2)) for i in range(NumberofSteps)}
    total_steps = 10
    step_size = (UpperGrenze - LowerGrenze) / (total_steps - 1)
    NumberofSteps = total_steps
    marks = {LowerGrenze + i * step_size: str(round(LowerGrenze + i * step_size, 2)) for i in range(NumberofSteps)}

    df[Reihen_select] = df[Reihen_select].astype(str)
    ReihenWert = df[Reihen_select].unique()


    AnzReihen = len(ReihenWert)

    if Spalten_select!='':
        df[Spalten_select] = df[Spalten_select].astype(str)
        SpaltenWert = df[Spalten_select].unique()
        AnzSpalten = len(SpaltenWert)
    else:
        AnzSpalten =1
        SpaltenWert = y_columns

    if len(y_columns)>1:
        AnzSpalten = AnzSpalten*len(y_columns)

    Name_subplots = []



    for row in ReihenWert:
        for y in y_columns:
            for c, col in enumerate(SpaltenWert, start=1):

                if len(y_columns)>1:
                    name = y + ' ' + row + ' ' + Reihen_select + ' ' + col + ' ' + Spalten_select
                else:
                    name = y + ' ' + row + ' ' + Reihen_select
                try:
                    name = name.replace(" SOC", "% SOC")
                except:
                    pass
                try:
                    name = name.replace(" SiO", "% SiO")
                except:
                    pass
                Name_subplots.append(name)

    fig = make_subplots(rows=AnzReihen, cols=AnzSpalten, subplot_titles=(Name_subplots))

    if chart_type == 'line':
        if AnzSpalten==1:
            if AnzReihen > 1:
                for r, Reihe in enumerate(ReihenWert, start=1):
                    for y_column in y_columns:
                        fig.add_trace(go.Scatter(x=df[x_column],
                                                 y=df[y_column][(df[Reihen_select].isin(ReihenWert[r-1].split()))],
                                                 mode='lines',
                                                 name=str(y_column) + ' ' + Reihe + ' ' + Reihen_select),
                                      row=r, col=1)
        if AnzSpalten==1:
            if AnzReihen==1:
                for y_column in y_columns:
                    fig.add_trace(go.Scatter(x=df[x_column],
                                             y=df[y_column][(df[Reihen_select].isin(ReihenWert))],
                                             mode='lines',
                                             name=str(y_column) + ' ' + Reihen_select),
                                  row=1, col=1)

        if AnzReihen==1:
            if AnzSpalten > 1:
                for s, Spalte in enumerate(SpaltenWert, start=1):
                    for y_column in y_columns:
                        fig.add_trace(go.Scatter(x=df[x_column],
                                                 y=df[y_column][(df[Spalten_select].isin(SpaltenWert[s - 1].split()))],
                                                 mode='lines',
                                                 name=str(y_column) + ' ' + Spalte + ' ' + Spalten_select),
                                      row=1, col=s)

        if AnzSpalten > 1:
            if AnzReihen > 1:
                try:
                    for y, y_column in enumerate(y_columns, start=0):
                        for s, Spalte in enumerate(SpaltenWert, start=1):
                            for r, Reihe in enumerate(ReihenWert, start=1):
                                SpaltenNummer = s+y*len(SpaltenWert)
                                fig.add_trace(go.Scatter(x=df[x_column],
                                                     y=df[y_column][(df[Reihen_select].isin(ReihenWert[r - 1].split())) & (df[Spalten_select].isin(SpaltenWert[s - 1].split()))],
                                                     mode='lines',
                                                     name=str(y_column) + ' ' + Reihe + ' ' + Reihen_select),
                                                     row=r, col=SpaltenNummer)
                except:
                    print('Fehler!')

    elif chart_type == 'bar':
        if chart_type == 'bar':
            if AnzSpalten == 1:
                if AnzReihen > 1:
                    for r, Reihe in enumerate(ReihenWert, start=1):
                        for y_column in y_columns:
                            fig.add_trace(go.Bar(x=df[x_column],
                                                 y=df[y_column][(df[Reihen_select].isin(ReihenWert[r - 1].split()))],
                                                 name=str(y_column) + ' ' + Reihe + ' ' + Reihen_select),
                                          row=r, col=1)

            if AnzSpalten == 1:
                if AnzReihen == 1:
                    print('Geht rein')
                    for y_column in y_columns:
                        fig.add_trace(go.Bar(x=df[x_column],
                                             y=df[y_column][(df[Reihen_select].isin(ReihenWert))],
                                             name=str(y_column) + ' ' + Reihen_select),
                                      row=1, col=1)

            if AnzReihen == 1:
                if AnzSpalten > 1:
                    for s, Spalte in enumerate(SpaltenWert, start=1):
                        for y_column in y_columns:
                            fig.add_trace(go.Bar(x=df[x_column],
                                                 y=df[y_column][(df[Spalten_select].isin(SpaltenWert[s - 1].split()))],
                                                 name=str(y_column) + ' ' + Spalte + ' ' + Spalten_select),
                                          row=1, col=s)

            if AnzSpalten > 1:
                if AnzReihen > 1:
                    try:
                        for y, y_column in enumerate(y_columns, start=0):
                            for s, Spalte in enumerate(SpaltenWert, start=1):
                                for r, Reihe in enumerate(ReihenWert, start=1):
                                    SpaltenNummer = s + y * len(SpaltenWert)

                                    fig.add_trace(go.Bar(x=df[x_column],
                                                         y=df[y_column][
                                                             (df[Reihen_select].isin(ReihenWert[r - 1].split())) & (
                                                                 df[Spalten_select].isin(SpaltenWert[s - 1].split()))],
                                                         name=str(y_column) + ' ' + Reihe + ' ' + Reihen_select),
                                                  row=r, col=SpaltenNummer)
                    except:
                        print('Fehler!')

    elif chart_type == 'scatter':
        if chart_type == 'scatter':
            if AnzSpalten == 1:
                if AnzReihen > 1:
                    for r, Reihe in enumerate(ReihenWert, start=1):
                        for y_column in y_columns:
                            fig.add_trace(go.Scatter(x=df[x_column],
                                                     y=df[y_column][
                                                         (df[Reihen_select].isin(ReihenWert[r - 1].split()))],
                                                     mode='markers',
                                                     name=str(y_column) + ' ' + Reihe + ' ' + Reihen_select),
                                          row=r, col=1)

            if AnzSpalten == 1:
                if AnzReihen == 1:
                    print('Geht rein')
                    for y_column in y_columns:
                        fig.add_trace(go.Scatter(x=df[x_column],
                                                 y=df[y_column][(df[Reihen_select].isin(ReihenWert))],
                                                 mode='markers',
                                                 name=str(y_column) + ' ' + Reihen_select),
                                      row=1, col=1)

            if AnzReihen == 1:
                if AnzSpalten > 1:
                    for s, Spalte in enumerate(SpaltenWert, start=1):
                        for y_column in y_columns:
                            fig.add_trace(go.Scatter(x=df[x_column],
                                                     y=df[y_column][
                                                         (df[Spalten_select].isin(SpaltenWert[s - 1].split()))],
                                                     mode='markers',
                                                     name=str(y_column) + ' ' + Spalte + ' ' + Spalten_select),
                                          row=1, col=s)

            if AnzSpalten > 1:
                if AnzReihen > 1:
                    try:
                        for y, y_column in enumerate(y_columns, start=0):
                            for s, Spalte in enumerate(SpaltenWert, start=1):
                                for r, Reihe in enumerate(ReihenWert, start=1):
                                    SpaltenNummer = s + y * len(SpaltenWert)

                                    fig.add_trace(go.Scatter(x=df[x_column],
                                                             y=df[y_column][
                                                                 (df[Reihen_select].isin(ReihenWert[r - 1].split())) & (
                                                                     df[Spalten_select].isin(
                                                                         SpaltenWert[s - 1].split()))],
                                                             mode='markers',
                                                             name=str(y_column) + ' ' + Reihe + ' ' + Reihen_select),
                                                  row=r, col=SpaltenNummer)
                    except:
                        print('Fehler!')

    large_rockwell_template = dict(
        layout=go.Layout(
            title_font=dict(family="Arial", size=24),
            xaxis=dict(showline=True, zeroline=False, linecolor='black'),
            yaxis=dict(showline=True, zeroline=False, linecolor='black'),
            plot_bgcolor='white',
        )
    )

    for row in range(AnzReihen):
        AnzSpalten = AnzSpalten*len(Reihen_select)
        for spalte in range(1, AnzSpalten + 1):
            AnzSpaltenNummer = spalte
            if radio_select==True:
                fig.update_yaxes(range=[low_y, high_y], row=row + 1, col=AnzSpaltenNummer)

    fig.update_layout(template=large_rockwell_template)
    fig.update_layout(title=f'{", ".join(y_columns)} - {x_column}')
    fig.update_layout(height=1000, width=2500)

    if yAchse_Label:
        fig.update_yaxes(title=yAchse_Label)
    if xAchse_Label:
        fig.update_xaxes(title=xAchse_Label)
    if title_Label:
        fig.update_layout(title=title_Label)

    return fig, LowerGrenze, UpperGrenze,y_range_slider_set, marks, Reihen_select , bezeichnung_values, [{'label': label, 'value': label} for label in
                                                                      Bez_values]

if __name__ == '__main__':
    app.run_server(debug=True)
