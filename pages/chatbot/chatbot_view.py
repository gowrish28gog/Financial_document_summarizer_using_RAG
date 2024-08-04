import dash_bootstrap_components as dbc
from dash import dcc
from dash import html

# import components
from components.navbar import render_navbar
from components.input import render_chat_input


# define layout
chatbot_layout = html.Div(
    html.Div(id="display-conversation"),
    style={
        "overflow-y": "auto",
        "display": "flex",
        "height": "calc(90vh - 132px)",
        "flex-direction": "column-reverse",
    },
)


def render_chatbot():
    return html.Div(
        [
            render_navbar(brand_name="AI Chatbot for Financial Reports (10Q, 10K)"),
            html.Br(),
            dcc.Store(id="store-conversation", data=""),
            dbc.Container(
                fluid=True,
                children=[
                    dbc.Row(
                        [
                            dbc.Col(width=1),
                            dbc.Col(
                                width=10,
                                children=dbc.Card(
                                    [
                                        dbc.CardBody([
                                            chatbot_layout,
                                            html.Div(render_chat_input(), style={'margin-left': '70px', 'margin-right': '70px', 'margin-bottom': '20px'}),
                                            dbc.Spinner(html.Div(id="loading-component")),
                                        ])
                                    ],
                                    style={'border-radius': 25, 'background': '#FFFFFF', 'border': '0px solid'}
                                )
                            ),
                            dbc.Col(width=1),
                        ]
                    )
                ]
            ),
        ],
    )