
import dash_core_components as dcc
import dash_html_components as html

from .player_controls import STEP_COMPLETE
from .dash_app import DashApp
from dash.dependencies import Input, Output


def build_log(dashapp):
    assert isinstance(dashapp, DashApp)

    layout = html.Div([
        html.Button('Refresh', id='log-refresh'),
        html.Div(id='log-dump', style={'background-color': '#F0F0F0'}, children=[
        ]),
        dcc.Interval(id='initialize-log', interval=200, max_intervals=1)
    ])

    @dashapp.dash.callback(Output('log-dump', 'children'), [STEP_COMPLETE, Input('initialize-log', 'n_intervals'), Input('log-refresh', 'n_clicks')])
    def refresh_log(*args):
        try:
            log_file = dashapp.model.log_file
            if log_file:
                with open(log_file, 'r') as stream:
                    log_text = stream.read()
                paragraphs = log_text.split('\n')
                return [html.P(paragraph) for paragraph in paragraphs]
            else:
                return "Log no available. Set scene=True or pass an existing Scene to view() to enable logging."
        except BaseException as exc:
            return 'Could not load log file: %s' % exc

    return layout
