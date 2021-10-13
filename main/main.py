from components import navbar, header, body, footer
from dash.dependencies import Input, Output, State
from model import *
from dash import html
import plotly.graph_objects as go

import dash_bootstrap_components as dbc
import dash


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(children=[
    navbar(),
    header(),
    body(),
    footer()
], style={'background-color': '	#eeeeee'})


@app.callback([Output("month-slider", "min"),
               Output("month-slider", "max"),
               Output("month-slider", "value"),
               Output("month-slider", "marks")],
              Input("year-tickbox", "value"))
def update_slider(selected_years):
    temp_df = crime_df.loc[crime_df['YEAR'].isin(selected_years)]
    min = temp_df['M'].min()
    max = temp_df['M'].max()
    value = [min, max]
    marks = {index+int(min): period for index, period in enumerate(temp_df['PERIOD'].unique())}
    return min, max, value, marks

@app.callback(Output("choropleth", "figure"), [Input("month-slider", "value"),
                                               Input("year-tickbox", "value"),
                                               Input("filters-dropdown", "value"),
                                               Input("level-radio", "value"),
                                               Input("crime-tickbox", "value"),
                                               ])
def update_choropleth(selected_months, selected_years, filters, level, crime):
    filtered_df = filter_dataset(selected_months, selected_years, filters, level, crime)
    if level == 'All' :
        fig = go.Figure(go.Choroplethmapbox(
            geojson=regions,
            locations=filtered_df.CODE,
            z=filtered_df.AVERAGE_MONTHLY_CRIME_RATE,
            colorscale="Viridis",
            zmin=filtered_df.AVERAGE_MONTHLY_CRIME_RATE.min(),
            zmax=filtered_df.AVERAGE_MONTHLY_CRIME_RATE.max(),
            marker_line_width=0,
            customdata=np.stack((filtered_df['CODE'],
                                 filtered_df['PRO'],
                                 filtered_df['POPULATION'],
                                 filtered_df['CRIME_RATE'],
                                 filtered_df['AVERAGE_MONTHLY_CRIME_RATE']), axis=-1)))
        fig.update_layout(mapbox_style="light", mapbox_accesstoken=token,
                          mapbox_zoom=5, mapbox_center = {"lat": 12.8797, "lon": 121.7740})
        fig.update_traces(
            hovertemplate="<br>".join([
                "CODE: %{customdata[0]}",
                "PRO: %{customdata[1]}",
                "POPULATION: %{customdata[2]}",
                "CRIME_RATE: %{customdata[3]}",
                "AMCR: %{customdata[4]}",
            ])
        )
        fig.update_layout(
            margin={"r":0,"t":0,"l":0,"b":0},
            hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ))
    else:
        fig = go.Figure(go.Choroplethmapbox(
            geojson=regions,
            locations=filtered_df.CODE,
            z=filtered_df.AVERAGE_MONTHLY_CRIME_RATE,
            colorscale="Viridis",
            zmin=filtered_df.AVERAGE_MONTHLY_CRIME_RATE.min(),
            zmax=filtered_df.AVERAGE_MONTHLY_CRIME_RATE.max(),
            marker_line_width=0,
            customdata=np.stack((filtered_df['CODE'],
                                 filtered_df['PRO'],
                                 filtered_df['POPULATION'],
                                 filtered_df['CRIME_RATE'],
                                 filtered_df['AVERAGE_MONTHLY_CRIME_RATE']),axis = -1)))
        fig.update_layout(mapbox_style="light", mapbox_accesstoken=token,
                          mapbox_zoom=5, mapbox_center={"lat": 12.8797, "lon": 121.7740})
        fig.update_traces(
            hovertemplate="<br>".join([
                "CODE: %{customdata[0]}",
                "PRO: %{customdata[1]}",
                "POPULATION: %{customdata[2]}",
                "CRIME_RATE: %{customdata[3]}",
                "AMCR: %{customdata[4]}<extra></extra>",
            ])
        )
        fig.update_layout(
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            hoverlabel=dict(
                bgcolor="white",
                font_size=18,
            ))

    return fig

@app.callback(Output("forecast-time-series", "figure"), [Input("graph-radioButton", "value"),
                                                         Input("choropleth", "hoverData"),
                                                         Input("crime-tickbox", "value")])
def update_linegraph(graph_type, hoverData, crime):
    filtered_df = filter_dataset([], [], [], False, crime)
    print(filtered_df)
    fig = go.Figure()

    if graph_type == 'forecast':

        if hoverData:
            print(hoverData['points'][0]['customdata'][0])
            code = hoverData['points'][0]['customdata'][0]
            if code:
                dataset = False
                try:
                    with open(os.path.join(os.path.dirname(__file__), "static/" + str(code) + ".pickle"), 'rb') as data:
                        df = pickle.load(data)
                except:
                    df = forecast(filtered_df.loc[filtered_df.CODE == code], 'CRIME_RATE')[0]

                col = 'CRIME_RATE'
                print('FORECAST: ', df)
                fig = go.Figure(
                    data=[{'x': df.index, 'y': df[col],
                           'mode': 'lines+markers', 'name': col},
                          {'x': df.index, 'y': df['train'],
                           'mode': 'lines+markers', 'name': 'Lead'},
                          {'x': df.index, 'y': df['Forecast'],
                           'mode': 'lines+markers', 'name': 'Forecast'}],
                    layout=go.Layout({"title": 'Forecast',
                                      "xaxis": {
                                          'rangeslider': {'visible': True},
                                          'rangeselector': {'visible': True,
                                                            'buttons': [{'step': 'all'}, {'step': 'year'},
                                                                        {'step': 'month'},
                                                                        {'step': 'day'}]}},
                                      "showlegend": False,
                                      "height": 500}
                                     )
                )
    return fig

# @app.callback(
#     dash.dependencies.Output('x-time-series', 'figure'),
#     dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData'))
# def update_y_timeseries(hoverData, xaxis_column_name, axis_type):
#     country_name = hoverData['points'][0]['customdata']
#     dff = df[df['Country Name'] == country_name]
#     dff = dff[dff['Indicator Name'] == xaxis_column_name]
#     title = '<b>{}</b><br>{}'.format(country_name, xaxis_column_name)
#     return create_time_series(dff, axis_type, title)


if __name__ == '__main__':
    app.run_server(debug=True)