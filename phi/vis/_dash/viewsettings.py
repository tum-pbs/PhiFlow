from typing import Any, Dict

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

from phi.field import SampledField
from phi.math._shape import parse_dim_order
from phi.vis._dash.dash_app import DashApp
from phi.vis._dash.player_controls import STEP_BUTTON, PAUSE_BUTTON
from phi.vis._dash.player_controls import REFRESH_INTERVAL
from phi.vis._vis_base import display_name

FRONT = 'front'
RIGHT = 'right'
TOP = 'top'

REFRESH_RATE = Input(f'refresh-rate', 'value')


def all_view_settings(app: DashApp, viewer_group: str):
    dims = [Input(f'{viewer_group}_select_{dim}', 'value') for dim in parse_dim_order(app.config.get('select', []))]
    return (Input(f'{viewer_group}_projection-select', 'value'),
            Input(f'{viewer_group}_component-slider', 'value'),
            Input(f'{viewer_group}_refresh-button', 'n_clicks'),
            *dims)


def parse_view_settings(app: DashApp, *args) -> Dict[str, Any]:
    projection, component, _, *selections = args
    return {
        'select': {dim: sel for dim, sel in zip(parse_dim_order(app.config.get('select', [])), selections)},
        'projection': projection,
        'component': [None, 'x', 'y', 'z', 'abs'][component],
    }


def refresh_rate_ms(refresh_value):
    if refresh_value is None:
        return 1000 * 60 * 60 * 24
    else:
        return (2000, 900, 400, 200)[refresh_value]


def build_view_selection(app: DashApp, field_selections: tuple, viewer_group: str):
    dim_sliders = []
    for sel_dim in parse_dim_order(app.config.get('select', [])):
        sel_pane = html.Div(style={'height': '50px', 'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}, children=[
            html.Label(display_name(sel_dim), style={'display': 'inline-block'}),
            html.Div(style={'width': '80%', 'display': 'inline-block'}, children=[
                dcc.Slider(min=0, max=0, step=1, value=0,
                           # marks={} if resolution3d is None else _marks(resolution3d[1]),
                           id=f'{viewer_group}_select_{sel_dim}', updatemode='drag', disabled=False),
                ]),
            ])
        dim_sliders.append(sel_pane)

        @app.dash.callback(Output(f'{viewer_group}_select_{sel_dim}', 'max'), [STEP_BUTTON, REFRESH_INTERVAL, PAUSE_BUTTON, *field_selections])
        def update_dim_max(_s, _r, _p, *field_names, dim=sel_dim):
            shapes = [app.model.get_field_shape(name) for name in field_names if name != 'None']
            sizes = [s.get_size(dim) for s in shapes if dim in s]
            if sizes:
                return max(sizes) - 1
            else:
                raise PreventUpdate()

        @app.dash.callback(Output(f'{viewer_group}_select_{sel_dim}', 'disabled'), [STEP_BUTTON, REFRESH_INTERVAL, PAUSE_BUTTON, *field_selections])
        def update_dim_disabled(_s, _r, _p, *field_names, dim=sel_dim):
            shapes = [app.model.get_field_shape(name) for name in field_names if name != 'None']
            sizes = [s.get_size(dim) for s in shapes if dim in s]
            return max(sizes) <= 1 if sizes else True

        @app.dash.callback(Output(f'{viewer_group}_select_{sel_dim}', 'marks'), [STEP_BUTTON, REFRESH_INTERVAL, PAUSE_BUTTON, *field_selections])
        def update_dim_disabled(_s, _r, _p, *field_names, dim=sel_dim):
            shapes = [app.model.get_field_shape(name) for name in field_names if name != 'None']
            sizes = [s.get_size(dim) for s in shapes if dim in s]
            if sizes:
                return _marks(max(sizes))
            else:
                return {}

    layout = html.Div(style={'width': '100%', 'display': 'inline-block', 'backgroundColor': '#E0E0FF', 'vertical-align': 'middle'}, children=[
        # --- Settings ---
        html.Div(style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top'}, children=[
            html.Div(style={'width': '50%', 'display': 'inline-block'}, children=[
                dcc.Dropdown(options=[{'value': FRONT, 'label': 'Front'}, {'value': RIGHT, 'label': 'Side'}, {'value': TOP, 'label': 'Top'}],
                             value='front', id=f'{viewer_group}_projection-select', disabled=False),
                html.Div(style={'margin-top': 6}, children=[
                    html.Div(style={'width': '90%', 'margin-left': 'auto', 'margin-right': 'auto'}, children=[
                        dcc.Slider(min=0, max=4, step=1, value=0, marks={0: '🡡', 4: '⬤', 1: 'x', 2: 'y', 3: 'z'}, id=f'{viewer_group}_component-slider', updatemode='drag'),
                    ])
                ]),
            ]),
            html.Div(style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'top'}, children=[  # , 'margin-left': 40
                html.Div('Refresh Rate', style={'text-align': 'center'}),
                html.Div(style={'width': '60%', 'margin-left': 'auto', 'margin-right': 'auto', 'height': 50}, children=[
                    dcc.Slider(min=0, max=3, step=1, value=1, marks={0: 'low', 3: 'high'}, id=REFRESH_RATE.component_id, updatemode='drag'),
                ]),
                html.Div(style={'text-align': 'center'}, children=[
                    html.Button('Refresh now', id=f'{viewer_group}_refresh-button'),
                ])
            ])
        ]),
        # --- Batch & Depth ---
        html.Div(style={'width': '70%', 'display': 'inline-block'}, children=dim_sliders),
    ])
    return layout


def _marks(stop, limit=35, step=1) -> dict:
    if stop <= limit * step:
        return {i: str(i) for i in range(0, stop, step)}
    if stop <= 2 * limit * step:
        return _marks(stop, limit, step * 2)
    if stop <= 5 * limit * step:
        return _marks(stop, limit, step * 5)
    else:
        return _marks(stop, limit, step * 10)
