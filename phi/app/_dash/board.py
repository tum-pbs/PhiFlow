import logging
import os

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input
from dash.exceptions import PreventUpdate

from .dash_app import DashApp
from .player_controls import STEP_COUNT, parse_step_count


BENCHMARK_BUTTON = Input('benchmark-button', 'n_clicks')
PROFILE_BUTTON = Input('profile-button', 'n_clicks')

NO_BENCHMARK_TEXT = '*No benchmarks available.*'
NO_PROFILES_TEXT = '*No profiles available.*'


def build_benchmark(dashapp):
    assert isinstance(dashapp, DashApp)

    layout = html.Div([
        dcc.Markdown('## Benchmark'),
        html.Div([
            html.Button('Benchmark', id=BENCHMARK_BUTTON.component_id)
        ]),
        dcc.Markdown(children=NO_BENCHMARK_TEXT, id='run-statistics'),
    ])

    @dashapp.dash.callback(Output('run-statistics', 'children'), [BENCHMARK_BUTTON], [STEP_COUNT])
    def run_benchmark(n_clicks, step_count):
        step_count = parse_step_count(step_count, dashapp, default=1)
        if n_clicks is None:
            return NO_BENCHMARK_TEXT
        if dashapp.app.running:
            return '*Pause the app before starting a benchmark.*'
        # --- Run benchmark ---
        step_count, time_elapsed = dashapp.app.benchmark(step_count)
        output = '### Benchmark Results\n'
        if step_count != step_count:
            output += 'The benchmark was stopped prematurely.  \n'
        output += 'Finished %d steps in %.03f seconds.' % (step_count, time_elapsed)
        output += '  \n*Average*: %.04f seconds per step, %.02f steps per second.' % (
            time_elapsed / step_count, step_count / time_elapsed)
        return output

    return layout


def build_tf_profiler(dashapp):
    assert isinstance(dashapp, DashApp)

    layout = html.Div([
        dcc.Markdown('## TensorFlow Profiler'),
        html.Div([
            html.Button('Profile', id=PROFILE_BUTTON.component_id)
        ]),
        dcc.Markdown(children=NO_PROFILES_TEXT, id='profile-output'),
    ])

    @dashapp.dash.callback(Output('profile-output', 'children'), [PROFILE_BUTTON], [STEP_COUNT])
    def run_benchmark(n_clicks, step_count):
        step_count = parse_step_count(step_count, dashapp, default=1)
        if n_clicks is None:
            return NO_PROFILES_TEXT
        if dashapp.app.running:
            return '*Pause the app before starting a profiled run.*'
        # --- Profile ---
        with dashapp.app.session.profiler() as profiler:
            timeline_file = profiler.timeline_file
            step_count, time_elapsed = dashapp.app.benchmark(step_count)
        output = '### Profiling Results\n'
        if step_count != step_count:
            output += 'The profiling run was stopped prematurely.  \n'
        output += 'Finished %d steps in %.03f seconds.' % (step_count, time_elapsed)
        output += '  \n*Average*: %.04f seconds per step, %.02f steps per second.' % (time_elapsed / step_count, step_count / time_elapsed)
        output += '  \nProfile saved. Open  \n*chrome://tracing/*  \n and load file  \n *%s*' % timeline_file
        return output
    return layout


TENSORBOARD_STATUS = Input('tensorboard-status', 'children')


def build_tensorboard_launcher(dashapp):
    assert isinstance(dashapp, DashApp)

    layout = html.Div([
        html.Div(id='tensorboard-div'),
        dcc.Interval(id='tensorboard-init', interval=200, max_intervals=1),
        html.Div(style={'display': 'none'}, id=TENSORBOARD_STATUS.component_id),
    ])

    @dashapp.dash.callback(Output('tensorboard-div', 'children'), [Input('tensorboard-init', 'n_intervals'), TENSORBOARD_STATUS])
    def update(*_):
        if 'tensorboard_url' in dashapp.config:
            return html.A('TensorBoard', href=dashapp.config['tensorboard_url'], id='tensorboard-href')
        else:
            return html.Button('Launch TensorBoard', id='launch-tensorboard')

    @dashapp.dash.callback(Output(TENSORBOARD_STATUS.component_id, TENSORBOARD_STATUS.component_property), [Input('launch-tensorboard', 'n_clicks')])
    def launch_tensorboard(clicks):
        if clicks:
            logging.info('Launching TensorBoard...')
            logdir = dashapp.app.session.summary_directory
            import phi.tf._profiling as profiling
            url = profiling.launch_tensorboard(logdir, port=dashapp.config.get('tensorboard_port', None))
            dashapp.config['tensorboard_url'] = url
            logging.info('TensorBoard launched, URL: %s' % url)
            return 'running'
        else:
            raise PreventUpdate()

    return layout


def build_system_controls(dashapp):
    assert isinstance(dashapp, DashApp)

    layout = html.Div([
        dcc.Markdown('## Application'),
        html.Button('Terminate', id='exit-button')
    ])

    @dashapp.dash.callback(Output('exit-button', 'style'), [Input('exit-button', 'n_clicks')])
    def exit_application(n):
        if n:
            logging.info('DashGUI: Exiting...')
            os._exit(0)  # exit() does not work from Dash threads

    return layout
