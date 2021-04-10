import traceback
import warnings

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from .dash_app import DashApp
from .dash_plotting import dash_graph_plot, EMPTY_FIGURE
from .model_controls import all_controls
from .player_controls import STEP_COMPLETE, all_actions, PLAYING, REFRESH_INTERVAL
import webglviewer
from .viewsettings import parse_view_settings, all_view_settings, refresh_rate_ms, REFRESH_RATE
from .webgl_util import default_sky, EMPTY_GRID, webgl_prepare_data
from .._vis_base import select_channel
from ...field import SampledField


def build_viewers(app: DashApp, count: int, height: int, viewer_group: str):
    field_names = app.model.field_names + ('None',) * max(0, 4 - len(app.model.field_names))
    result = []
    ids = [f'{viewer_group}_{i}' for i in range(count)]
    field_selections = tuple(Input(f'{id}-field-select', 'value') for id in ids)
    for id, field_name in zip(ids, field_names):
        result.append(build_viewer(app, height, field_name, id, viewer_group))
    return result, field_selections


def build_viewer(app: DashApp, height: int, initial_field_name: str, id: str, viewer_group: str):
    field_options = [{'label': item, 'value': item} for item in app.model.field_names] + [{'label': '<None>', 'value': 'None'}]

    layout = html.Div(style={'height': '100%'}, children=[
        html.Div(style={'width': '100%', 'height': '5%', 'display': 'inline-block', 'vertical-align': 'middle'}, children=[
            html.Div(style={'width': '90%', 'display': 'inline-block'}, children=[
                dcc.Dropdown(options=field_options, value=initial_field_name, id=id+'-field-select'),
            ]),
            html.Div(style={'width': '10%', 'display': 'inline-block', 'vertical-align': 'top', 'text-align': 'center'}, children=[
                dcc.Checklist(options=[{'label': '3D', 'value': '3D'}], value=[], id=id+'-3d', style={'margin-top': 6}),
            ]),
        ]),
        html.Div(id=id+'-figure-container', children=[], style={'height': '95%', 'width': '100%', 'display': 'inline-block'}),
    ])

    @app.dash.callback(Output(id+'-figure-container', 'children'), [Input(id+'-3d', 'value')])
    def choose_viewer(list3d):
        if list3d:
            return [
                dcc.Interval(id=id + '-webgl-initializer', interval=100, max_intervals=1),
                webglviewer.Webglviewer(id=id+'-webgl', sky=default_sky(), material_type='LIGHT_SMOKE', representation_type='DENSITY')
            ]
        else:
            return dcc.Graph(figure=EMPTY_FIGURE, id=id + '-graph', style={'height': '100%'})

    @app.dash.callback(Output(id+'-graph', 'figure'), (Input(f'{id}-field-select', 'value'), STEP_COMPLETE, REFRESH_INTERVAL, *all_view_settings(app, viewer_group), *all_controls(app), *all_actions(app)))
    def update_figure(field, _0, _1, *settings):
        if field is None or field == 'None':
            return EMPTY_FIGURE
        value = app.get_field(field)
        if not isinstance(value, SampledField):
            return EMPTY_FIGURE
        selection = parse_view_settings(app, *settings)
        value = value[selection['select']]
        value = select_channel(value, selection.get('component', None))
        try:
            return dash_graph_plot(value, app.config, height)
        except BaseException as err:
            warnings.warn(f"Error during plotting: {err}")
            traceback.print_exc()
            return EMPTY_FIGURE

    @app.dash.callback(Output(id+'-webgl', 'data'), (Input(id+'-field-select', 'value'), Input(id+'-webgl-initializer', 'n_intervals'), STEP_COMPLETE, REFRESH_INTERVAL, *all_view_settings(app, viewer_group), *all_controls(app), *all_actions(app)))
    def update_webgl_data(field, _0, _1, _2, *settings):
        if field is None or field == 'None':
            return EMPTY_GRID
        data = app.get_field(field)
        if data is None:
            return EMPTY_GRID
        settings_dict = parse_view_settings(app, *settings)
        return webgl_prepare_data(data, app.config, settings_dict)

    return layout
