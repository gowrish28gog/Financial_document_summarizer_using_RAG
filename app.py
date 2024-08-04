import dash_bootstrap_components as dbc
from dash import Dash
import openai

APP_TITLE = "Chatbot App"

app = Dash(__name__,
            title=APP_TITLE,
            update_title='Loading...',
            suppress_callback_exceptions=True,
            external_stylesheets=[dbc.themes.FLATLY])