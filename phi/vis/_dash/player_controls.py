import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate

from .dash_app import DashApp
from .._vis_base import display_name


REFRESH_INTERVAL = Input('playing-refresh-interval', 'n_intervals')


def build_status_bar(app: DashApp):
    layout = html.Div([
        html.Div(id='status-bar', children=['Loading status...'], style={'backgroundColor': '#E0E0FF'}),
        dcc.Interval(id='status-interval', interval=500),
    ])

    @app.dash.callback(Output('status-bar', 'children'), [Input('status-interval', 'n_intervals'), STEP_COMPLETE, PLAYING])
    def update_status_bar(*_):
        return [app.status_message]

    return layout


PLAY_BUTTON = Input('play-button', 'n_clicks')
PLAYING = Input(PLAY_BUTTON.component_id, 'style')
PAUSE_BUTTON = Input('pause-button', 'n_clicks')
STEP_BUTTON = Input('step-button', 'n_clicks')
STEP_COMPLETE = Input('step-complete', 'children')
STEP_COUNT = State('step-count', 'value')


def all_actions(app: DashApp):
    return tuple(Input(f'action_{action.name}', 'n_clicks') for action in app.model.actions)


def build_player_controls(app: DashApp):

    layout = html.Div(style={'height': '30px'}, children=[
        html.Button('Play', id=PLAY_BUTTON.component_id),
        html.Button('Pause', id=PAUSE_BUTTON.component_id),
        html.Button('Step', id=STEP_BUTTON.component_id),
        dcc.Textarea(placeholder='#steps', id=STEP_COUNT.component_id, value='', rows=1, style={'width': 70}),
        *[html.Button(display_name(action.name), id=f'action_{action.name}') for action in app.model.actions],
        html.Div(style={'display': 'none'}, id=STEP_COMPLETE.component_id),
    ])

    @app.dash.callback(Output(PLAY_BUTTON.component_id, 'style'), inputs=[PLAY_BUTTON], state=[STEP_COUNT])
    def play(n_clicks, step_count):
        if n_clicks and not app.play_status:
            step_count = parse_step_count(step_count, app, default=None)
            app.play(max_steps=step_count)
        else:
            raise PreventUpdate()

    @app.dash.callback(Output(PAUSE_BUTTON.component_id, 'style'), [PAUSE_BUTTON])
    def pause_simulation(n_clicks):
        if n_clicks:
            app.pause()
        raise PreventUpdate()

    @app.dash.callback(Output(STEP_BUTTON.component_id, 'style'), [STEP_BUTTON])
    def simulation_step(n_clicks):
        if n_clicks and not app.play_status:
            app.model.progress()
        raise PreventUpdate()

    @app.dash.callback(Output(STEP_COMPLETE.component_id, 'children'), [STEP_BUTTON, PAUSE_BUTTON])
    def simulation_step(step, pause):
        return ['%s / %s' % (step, pause)]

    for action in app.model.actions:
        @app.dash.callback(Output(f'action_{action.name}', 'disabled'), [Input(f'action_{action.name}', 'n_clicks')])
        def perform_action(n_clicks, action=action):
            if n_clicks is not None:
                app.model.run_action(action.name)
            raise PreventUpdate()

    return layout


def parse_step_count(step_count, app, default: int or None = 1):
    if step_count is None:
        return default
    try:
        step_count = step_count.strip()
        if step_count.startswith('*'):
            step_count = app.model.sequence_stride * int(step_count[1:].strip())
        else:
            step_count = int(step_count)
        return step_count
    except ValueError:
        return default
