from dash import dcc
from dash import html
from urllib.request import urlopen
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from components import *
from model import *

import plotly.graph_objects as go
import dash


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(children=[
    navbar(),
    header(),
    body(),
    footer()
], style={'background-color': '	#eeeeee'})

@app.callback(Output("choropleth", "figure"), [Input("month-slider", "value"),
                                               Input("year-tickbox", "value"),
                                               Input("filters-dropdown", "value"),
                                               Input("level-radio", "value"),
                                               Input("crime-tickbox", "value"),
                                               ])
def update_choropleth(selected_months, selected_years, filters, level, crime):
    filtered_df = filter_dataset(selected_months, selected_years, filters, level, crime)
    print(filtered_df.AVERAGE_MONTHLY_CRIME_RATE)
    fig = go.Figure()
    if level == 'All' :
        fig = go.Figure(go.Choroplethmapbox(
            geojson=regions,
            locations=filtered_df.CODE,
            z=filtered_df.AVERAGE_MONTHLY_CRIME_RATE,
            colorscale="Viridis", zmin=0, zmax=100, marker_line_width=0))
        fig.update_layout(mapbox_style="light", mapbox_accesstoken=token,
                          mapbox_zoom=5, mapbox_center = {"lat": 12.8797, "lon": 121.7740})
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    else:
        fig = go.Figure(go.Choroplethmapbox(
            geojson=regions,
            locations=filtered_df.CODE,
            z=filtered_df.AVERAGE_MONTHLY_CRIME_RATE,
            colorscale="Viridis", zmin=0, zmax=100, marker_line_width=0))
        fig.update_layout(mapbox_style="light", mapbox_accesstoken=token,
                          mapbox_zoom=5, mapbox_center={"lat": 12.8797, "lon": 121.7740})
        fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    return fig

@app.callback(Output("linegraph", "figure"), Input("graph-radioButton", "value"))
def update_linegraph(graph_type):
    fig = go.Figure()

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)