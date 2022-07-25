import traceback

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from plotly import graph_objects

from .dash_app import DashApp
from .model_controls import all_controls
from .player_controls import STEP_COMPLETE, all_actions, REFRESH_INTERVAL
from .viewsettings import parse_view_settings, all_view_settings
from .. import plot
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
            dcc.Dropdown(options=field_options, value=initial_field_name, id=id+'-field-select'),
        ]),
        html.Div(id=id+'-figure-container', style={'height': '95%', 'width': '100%', 'display': 'inline-block'}, children=[
            dcc.Graph(figure={}, id=id + '-graph', style={'height': '100%'})
        ]),
    ])

    @app.dash.callback(Output(id+'-graph', 'figure'), (Input(f'{id}-field-select', 'value'), STEP_COMPLETE, REFRESH_INTERVAL, *all_view_settings(app, viewer_group), *all_controls(app), *all_actions(app)))
    def update_figure(field, _0, _1, *settings):
        if field is None or field == 'None':
            fig = graph_objects.Figure()
            fig.update_layout(title_text="None", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            return fig
        selection = parse_view_settings(app, *settings)
        value = app.model.get_field(field, selection['select'])
        try:
            value = select_channel(value, selection.get('component', None))
            return plot(value, lib='plotly', size=(height, height), same_scale=False, colormap=app.config.get('colormap', None)).native()
        except ValueError as err:
            fig = graph_objects.Figure()
            fig.update_layout(title_text=str(value), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            return fig

    return layout
