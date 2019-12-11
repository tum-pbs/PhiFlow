import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input
from dash.exceptions import PreventUpdate

from phi.viz.dash.viewsettings import refresh_rate_ms, REFRESH_RATE


def build_status_bar(dashapp):
    layout = html.Div([
        html.Div(id='status-bar', children=['Loading status...'], style={'backgroundColor': '#E0E0FF'}),
        dcc.Interval(id='status-interval', interval=500),
    ])

    @dashapp.dash.callback(Output('status-bar', 'children'), [Input('status-interval', 'n_intervals'), STEP_COMPLETE, PLAYING])
    def update_status_bar(*args):
        return [dashapp.app.status]

    return layout


PLAY_BUTTON = Input('play-button', 'n_clicks')
PLAYING = Input(PLAY_BUTTON.component_id, 'style')
PAUSE_BUTTON = Input('pause-button', 'n_clicks')
STEP_BUTTON = Input('step-button', 'n_clicks')
STEP_COMPLETE = Input('step-complete', 'children')
REFRESH_INTERVAL = Input('playing-refresh-interval', 'n_intervals')


def build_player_controls(dashapp):
    layout = html.Div(style={'height': '30px'}, children=[
        html.Button('Play', id=PLAY_BUTTON.component_id),
        html.Button('Pause', id=PAUSE_BUTTON.component_id),
        html.Button('Step', id=STEP_BUTTON.component_id),
        html.Div(style={'display': 'none'}, id=STEP_COMPLETE.component_id),
        dcc.Interval(id=REFRESH_INTERVAL.component_id, interval=refresh_rate_ms(None)),
    ])

    @dashapp.dash.callback(Output(PLAY_BUTTON.component_id, 'style'), [PLAY_BUTTON])
    def start_simulation(n_clicks):
        if n_clicks and not dashapp.app.running:
            dashapp.app.play()
        else:
            raise PreventUpdate()

    @dashapp.dash.callback(Output(PAUSE_BUTTON.component_id, 'style'), [PAUSE_BUTTON])
    def pause_simulation(n_clicks):
        if n_clicks:
            dashapp.app.pause()
        raise PreventUpdate()

    @dashapp.dash.callback(Output(STEP_BUTTON.component_id, 'style'), [STEP_BUTTON])
    def simulation_step(n_clicks):
        if n_clicks and not dashapp.app.running:
            dashapp.app.run_step()
        raise PreventUpdate()

    @dashapp.dash.callback(Output(STEP_COMPLETE.component_id, 'children'), [STEP_BUTTON, PAUSE_BUTTON])
    def simulation_step(step, pause):
        return ['%s / %s' % (step, pause)]

    @dashapp.dash.callback(Output(REFRESH_INTERVAL.component_id, 'interval'), [REFRESH_RATE, PLAYING, STEP_COMPLETE, REFRESH_INTERVAL])
    def set_refresh_interval(refresh_rate_value, *args):
        if refresh_rate_value is None:
            result = 1000
        elif dashapp.app.running:
            result = refresh_rate_ms(refresh_rate_value)
        else:
            result = refresh_rate_ms(None)
        return result

    return layout
