import logging
import os
import traceback

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input
from dash.exceptions import PreventUpdate
from plotly import graph_objects

from .dash_app import DashApp
from ._plotly_plots import plot_scalars
from .player_controls import STEP_COUNT, parse_step_count
from .._vis_base import display_name, gui_interrupt, benchmark

BENCHMARK_BUTTON = Input('benchmark-button', 'n_clicks')
PROFILE_BUTTON = Input('profile-button', 'n_clicks')

NO_BENCHMARK_TEXT = '*No benchmarks available.*'
NO_PROFILES_TEXT = '*No profiles available.*'

REFRESH_GRAPHS_BUTTON = Input('refresh-graphs-button', 'n_clicks')


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
        if dashapp.play_status:
            return '*Pause the vis before starting a benchmark.*'
        # --- Run benchmark ---
        step_count, time_elapsed = benchmark(dashapp.model, step_count)
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
        if dashapp.play_status:
            return '*Pause the vis before starting a profiled run.*'
        # --- Profile ---
        with dashapp.model.session.profiler() as profiler:
            timeline_file = profiler.timeline_file
            step_count, time_elapsed = dashapp.model.benchmark(step_count)
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
            logdir = dashapp.model.session.summary_directory
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
        html.Button('Exit / Interrupt', id='exit-button'),
        html.Button('Kill', id='kill-button'),
    ])

    @dashapp.dash.callback(Output('kill-button', 'style'), [Input('kill-button', 'n_clicks')])
    def exit_application(n):
        if n:
            logging.info('DashGUI: Killing process...')
            os._exit(0)  # exit() does not work from Dash threads

    @dashapp.dash.callback(Output('exit-button', 'style'), [Input('exit-button', 'n_clicks')])
    def exit_application(n):
        if n:
            dashapp.exit_interrupt()

    return layout


def build_graph_view(dashapp):
    layout = html.Div(style={'width': '90%', 'margin-left': 'auto', 'margin-right': 'auto'}, children=[
        html.H2("Graphs"),
        html.Div([
            html.Button('Refresh now', id=REFRESH_GRAPHS_BUTTON.component_id),
            dcc.Checklist(id='auto-refresh-checkbox', options=[{'label': 'Auto-refresh', 'value': 'refresh'}], value=['refresh'], style={'display': 'inline-block'}),
            dcc.Checklist(id='subplots-checkbox', options=[{'label': 'Subplots', 'value': 'subplots'}], value=['subplots'], style={'display': 'inline-block'}),
            html.Div(style={'display': 'inline-block', 'width': '200px'}, children=[
                dcc.Slider(id='smooth-slider', min=1, max=10, marks={1: 'Off', 5: '25 steps', 10: '100'}),
            ]),
            dcc.Checklist(id='log-graph-checkbox', options=[{'label': 'Log(x)', 'value': 'x'}, {'label': 'Log(y)', 'value': 'y'}], value=['y'], style={'display': 'inline-block'}),
        ]),
        dcc.Interval(id='graph-update', interval=5000, disabled=False),
        html.Div(id='graph-figure-container', style={'height': 600, 'width': '100%'}, children=[
            dcc.Graph(figure={}, id='board-graph', style={'height': '100%'})
        ])
    ])

    @dashapp.dash.callback(Output('board-graph', 'figure'), [Input('subplots-checkbox', 'value'), Input('smooth-slider', 'value'), Input('log-graph-checkbox', 'value'), REFRESH_GRAPHS_BUTTON, Input('graph-update', 'n_intervals')])
    def update_figure(subplots, smooth, log_scale, _n1, _n2):
        curves = [dashapp.model.get_curve(n) for n in dashapp.model.curve_names]
        labels = [display_name(n) for n in dashapp.model.curve_names]
        try:
            figure = plot_scalars(curves, labels, subplots=bool(subplots), log_scale=log_scale, smooth=(smooth or 1) ** 2)
            return figure
        except BaseException as err:
            traceback.print_exc()
            fig = graph_objects.Figure()
            fig.update_layout(title_text=repr(err))
            return fig

    @dashapp.dash.callback(Output('graph-update', 'disabled'), [Input('auto-refresh-checkbox', 'value')])
    def enable_auto_refresh(selected):
        if selected:
            return False
        else:
            return True

    return layout