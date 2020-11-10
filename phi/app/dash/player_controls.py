import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate

from .dash_app import DashApp
from .viewsettings import refresh_rate_ms, REFRESH_RATE


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
STEP_COUNT = State('step-count', 'value')


def build_player_controls(dashapp):
    assert isinstance(dashapp, DashApp)

    layout = html.Div(style={'height': '30px'}, children=[
        html.Button('Play', id=PLAY_BUTTON.component_id),
        html.Button('Pause', id=PAUSE_BUTTON.component_id),
        html.Button('Step', id=STEP_BUTTON.component_id),
        dcc.Textarea(placeholder='#steps', id=STEP_COUNT.component_id, value='', rows=1, style={'width': 70}),
        html.Div(style={'display': 'none'}, id=STEP_COMPLETE.component_id),
        dcc.Interval(id=REFRESH_INTERVAL.component_id, interval=refresh_rate_ms(None)),
    ])

    @dashapp.dash.callback(Output(PLAY_BUTTON.component_id, 'style'), inputs=[PLAY_BUTTON], state=[STEP_COUNT])
    def play(n_clicks, step_count):
        if n_clicks and not dashapp.app.running:
            step_count = parse_step_count(step_count, dashapp, default=None)
            if step_count is None:
                dashapp.app.play()
            else:
                dashapp.app.play(step_count)
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


def parse_step_count(step_count, dashapp, default=1):
    if step_count is None:
        return default
    try:
        step_count = step_count.strip()
        if step_count.startswith('*'):
            step_count = dashapp.app.sequence_stride * int(step_count[1:].strip())
        else:
            step_count = int(step_count)
        return step_count
    except ValueError:
        return default
