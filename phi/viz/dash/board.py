import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate

from phi.viz.dash.dash_app import DashApp
from phi.viz.dash.player_controls import STEP_COUNT, parse_step_count
from phi.viz.dash.viewsettings import refresh_rate_ms, REFRESH_RATE


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
        if dashapp.app.record_data or dashapp.app.record_images:
            output += 'Recording was disabled during benchmark.  \n'
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
        if dashapp.app.record_data or dashapp.app.record_images:
            output += 'Recording was disabled during benchmark.  \n'
        output += 'Finished %d steps in %.03f seconds.' % (step_count, time_elapsed)
        output += '  \n*Average*: %.04f seconds per step, %.02f steps per second.' % (time_elapsed / step_count, step_count / time_elapsed)
        output += '  \nProfile saved. Open  \n*chrome://tracing/*  \n and load file  \n *%s*' % timeline_file
        return output
    return layout
