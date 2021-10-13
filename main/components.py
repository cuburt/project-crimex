import dash_bootstrap_components as dbc
import plotly.graph_objects as go

from dash import html
from dash import dcc
from model import *

PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"

def navbar():
    navbar = dbc.Navbar(
        [
            dbc.Container([
                dbc.Row([
                    html.A(
                        # Use row and col to control vertical alignment of logo / brand
                        dbc.Row(
                            [
                                dbc.Col(html.Img(src=PLOTLY_LOGO, height="30px")),
                                dbc.Col(dbc.NavbarBrand("CRIMEx", className="ml-2")),
                            ],
                            align="center",
                            no_gutters=True,
                        ),
                        href="https://plotly.com",
                    )
                ]),
                dbc.Row([
                    dbc.NavLink('Upload CSV', href='#', style={'align': 'right', 'color': '#eeeeee'}),
                    dbc.NavLink('API Docs', href='#', style={'align': 'right', 'color': '#eeeeee'}),
                    dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
                 ], align="right")

            ])
        ],
        color="dark",
        dark=True,
    )

    return navbar

def header():
    header = dbc.Container([
        dbc.Card(
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H1('Welcome to CRIMEx!'),
                        html.P('Crime Rate Monitoring and Extrapolation')
                    ])
                ]),
            ]), style={'margin-top':'8px', 'margin-bottom':'8px'}
        )
    ], style={'padding':'0px 8px'})
    return header

def body():
    body = dbc.Container([
        dbc.Row(id='notif-space'),
        dbc.Row(
            [
                dbc.Col([
                    dbc.Row(
                        dbc.Col(
                            dbc.Card([
                                dbc.Row(
                                    dbc.Col([
                                        html.Label('Pre-defined filters'),
                                        dcc.Dropdown(
                                            id='filters-dropdown',
                                            options=[
                                                {'label': 'Gender (Male)', 'value': 'male'},
                                                {'label': 'Gender (Female)', 'value': 'female'},
                                                {'label': 'Gender (All)', 'value': 'sex_all'},
                                                {'label': 'Age (All)', 'value': 'age_all'},
                                                {'label': 'Age (13-17)', 'value': 'age_13_17'},
                                                {'label': 'Age (17-22)', 'value': 'age_17_22'},
                                                {'label': 'Age (23-30)', 'value': 'age_23_30'},
                                                {'label': 'Age (31-40)', 'value': 'age_31_40'},
                                                {'label': 'Age (41-60)', 'value': 'age_41_60'},
                                                {'label': 'Age (61-80)', 'value': 'age_61_80'},
                                                {'label': 'Age (81 or Above)', 'value': 'age_81'},
                                            ],
                                            value=['sex_all', 'age_all'],
                                            multi=True
                                        ),
                                    ], md=12
                                    ),
                                    style={'margin-bottom': '20px'}
                                ),
                                dbc.Row(
                                    dbc.Col([
                                        html.Label('Level of Administration'),
                                        dcc.RadioItems(
                                            id='level-radio',
                                            options=[
                                                {'label': 'All         ', 'value': 'All'},
                                                {'label': 'Region      ', 'value': 'Reg'},
                                                {'label': 'Province    ', 'value': 'Prov'},
                                                {'label': 'City        ', 'value': 'City'},
                                                {'label': 'Municipality', 'value': 'Mun'},
                                                {'label': 'Barangay    ', 'value': 'Bgy'}
                                            ],
                                            value='Reg',
                                            labelStyle=dict(display='block')
                                        ),
                                    ], md=12
                                    ),
                                    style={'margin-bottom': '20px'}
                                ),
                                dbc.Row(
                                    dbc.Col([
                                        html.Label('Type of Crime'),
                                        dcc.Checklist(
                                            id='crime-tickbox',
                                            options=[
                                                {'label': 'Murder                   ', 'value': 'MUR'},
                                                {'label': 'Homicide                 ', 'value': 'HOM'},
                                                {'label': 'Physical Injury          ', 'value': 'PHY_INJ'},
                                                {'label': 'Rape                     ', 'value': 'RAPE'},
                                                {'label': 'Robbery                  ', 'value': 'ROB'},
                                                {'label': 'Theft                    ', 'value': 'THEFT'},
                                                {'label': 'Carnapping (MV)          ', 'value': 'MV'},
                                                {'label': 'Carnapping (MC)          ', 'value': 'MC'},
                                                {'label': 'Cattle Rustling          ', 'value': 'CATTLE_RUSTLING'},
                                                {'label': 'RIRT Homicide            ', 'value': 'RIRT_HOM'},
                                                {'label': 'RIRT Phy. Inj.           ', 'value': 'RIRT_PHY_INJ'},
                                                {'label': 'RIRT Damage to Pty.      ', 'value': 'RIRT_DTP'},
                                                {'label': 'Violation of Special Laws', 'value': 'VIOL_OF_SPECIAL_LAWS'},
                                                {'label': 'Other Non-index Crimes   ', 'value': 'OTHER_NON_INDEX_CRIMES'},
                                            ],
                                            value=['MUR', 'HOM', 'PHY_INJ', 'RAPE', 'ROB', 'THEFT', 'MV', 'MC',
                                                   'CATTLE_RUSTLING', 'RIRT_HOM', 'RIRT_PHY_INJ', 'RIRT_DTP',
                                                   'VIOL_OF_SPECIAL_LAWS', 'OTHER_NON_INDEX_CRIMES'],
                                            labelStyle=dict(display='block')
                                        ),
                                    ], md=12
                                    ),
                                    style={'margin-bottom': '10px'}
                                ),
                            ],
                            style={'padding': '18.5px'}
                            ),
                            md=12
                        )
                    ),
                ], md=3, style={'margin':'0px','padding':'0px 4px'}),

                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.Row(
                                    dbc.Col([
                                        dcc.Checklist(
                                            id='year-tickbox',
                                            options=[
                                                {'label': str(year), 'value': year} for year in crime_df['YEAR'].unique()
                                            ],
                                            value=[2018]
                                        ),
                                        dcc.RangeSlider(
                                            id='month-slider',
                                            step=None,
                                            allowCross=False
                                        )
                                    ]),
                                    style={'margin-top': '8px', 'margin-bottom': '8px'}
                                ),
                                dcc.Loading(
                                    id='choropleth-loading',
                                    type='circle',
                                    children=[
                                        html.Div(id='choropleth-loading-output'),
                                        dcc.Graph(id='choropleth', style={'height': '200mm'})
                                    ]
                                )],
                                style = {'padding': '8px', 'margin-bottom':'8px'}
                            )
                        ], md=12)
                    ])
                ], md=9, style={'margin':'0px','padding':'0px 4px'})
            ], style={'padding':'0px 4px'}
        ),
        dbc.Row(
            [
                dbc.Col([
                    dbc.Row(
                        dbc.Col(
                            dbc.Card([
                                html.Label('Radio Items', style={'margin-top': '3.5px'}),
                                dcc.RadioItems(
                                    id='graph-radioButton',
                                    options=[
                                        {'label': 'Crime Rate Forecast', 'value': 'forecast'},
                                        {'label': 'Crime Volume by Month', 'value': 'perMonth'},
                                        {'label': 'Crime Volume by Region', 'value': 'perRegion'},
                                        {'label': 'Crime Volume by Province', 'value': 'perProvince'},
                                        {'label': 'Crime Volume by City', 'value': 'perCity'},
                                        {'label': 'Crime Volume by Municipality', 'value': 'perMunic'},
                                        {'label': 'Crime Volume by Barangay', 'value': 'perBar'},
                                        {'label': 'Volume Distribution', 'value': 'dist'},
                                        {'label': 'Cumulative Distribution', 'value': 'cum_dist'},
                                        {'label': 'Gaussian Process', 'value': 'gpr'},
                                    ],
                                    value='forecast',
                                    labelStyle = dict(display='block')
                                ),
                            ], style = {'padding': '20px', 'margin-bottom':'8px'}
                            ), md=12
                        )
                    ),
                ], md=3, style={'margin': '0px', 'padding': '0px 4px'}),

                dbc.Col([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.Row(
                                    dbc.Col(
                                        dcc.Loading(
                                            id='linegraph-loading',
                                            type='circle',
                                            children=[
                                                html.Div(id='linegraph-loading-output'),
                                                dcc.Graph(id='forecast-time-series',
                                                          style={'height': '130mm'})
                                            ]
                                        )
                                    )
                                )], style = {'padding': '8px', 'margin-bottom':'8px'}
                            )
                        ], md=12)
                    ])
                ], md=9, style={'margin': '0px', 'padding': '0px 4px'})
            ], style={'padding': '0px 4px'}
        ),
        # dbc.Row([
        #     # dbc.Col([
        #     #     create_accordion()
        #     # ], style={'margin':'4px','padding':'0px 4px'})
        # ])
    ])
    return body

def footer():
    footer = dbc.NavbarSimple(
        brand="© 2021 R&D Wing. All rights reserved.",
        color="primary",
        dark=True,
    )
    return footer