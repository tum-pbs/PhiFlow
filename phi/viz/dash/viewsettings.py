import numpy

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from phi.app import App
from phi.physics.field import Field, CenteredGrid, StaggeredGrid
from phi.viz.plot import FRONT, RIGHT, TOP


VIEWED_BATCH = Input('batch-slider', 'value')
VIEWED_DEPTH = Input('depth-slider', 'value')
PROJECTION_AXIS = Input('projection-select', 'value')
VIEWED_COMPONENT = Input('component-slider', 'value')
REFRESH_BUTTON = Input('refresh-button', 'n_clicks')

VIEW_SETTINGS = (VIEWED_BATCH, VIEWED_DEPTH, PROJECTION_AXIS, VIEWED_COMPONENT, REFRESH_BUTTON)

REFRESH_RATE = Input('refresh-rate-slider', 'value')


def parse_view_settings(config, *args):
    batch = args[0]
    depth = args[1]
    projection = args[2]  # type: str
    component = _COMPONENTS[args[3]]  # type: str
    user_settings = {
        'batch': batch,
        'depth': depth,
        'projection': projection,
        'component': component,
    }
    all_settings = {} if config is None else dict(config)
    all_settings.update(user_settings)
    return all_settings


def refresh_rate_ms(refresh_value):
    if refresh_value is None:
        return 1000 * 60 * 60 * 24
    else:
        return (2000, 900, 400, 200)[refresh_value]


def build_view_selection(dashapp, viewed_batch=0, viewed_depth=0, viewed_component='length'):
    batch_size, resolution3d = detect_slices(dashapp.app)

    layout = html.Div(style={'width': '100%', 'display': 'inline-block', 'backgroundColor': '#E0E0FF', 'vertical-align': 'middle'}, children=[
        # --- Settings ---
        html.Div(style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top'}, children=[
            html.Div(style={'width': '50%', 'display': 'inline-block'}, children=[
                dcc.Dropdown(options=[{'value': FRONT, 'label': 'Front'}, {'value': RIGHT, 'label': 'Side'}, {'value': TOP, 'label': 'Top'}], value='front', id=PROJECTION_AXIS.component_id, disabled=resolution3d is None),
                html.Div(style={'margin-top': 6}, children=[
                    html.Div('Component', style={'text-align': 'center'}),
                    html.Div(style={'width': '90%', 'margin-left': 'auto', 'margin-right': 'auto'}, children=[
                        dcc.Slider(min=0, max=4, step=1, value=_COMPONENTS.index(viewed_component), marks={0: 'v', 4: '|.|', 1: 'x', 2: 'y', 3: 'z'}, id='component-slider', updatemode='drag'),
                    ])
                ]),
            ]),
            html.Div(style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'top'}, children=[  # , 'margin-left': 40
                html.Div('Refresh Rate', style={'text-align': 'center'}),
                html.Div(style={'width': '60%', 'margin-left': 'auto', 'margin-right': 'auto', 'height': 50}, children=[
                    dcc.Slider(min=0, max=3, step=1, value=1, marks={0: 'low', 3: 'high'}, id=REFRESH_RATE.component_id, updatemode='drag'),
                ]),
                html.Div(style={'text-align': 'center'}, children=[
                    html.Button('Refresh now', id=REFRESH_BUTTON.component_id),
                ])
            ])
        ]),
        # --- Batch & Depth ---
        html.Div(style={'width': '70%', 'display': 'inline-block'}, children=[
            html.Div(style={'height': '50px', 'width': '100%', 'display': 'inline-block'}, children=[
                'Batch',
                dcc.Slider(min=0,
                           max=0 if batch_size is None else batch_size - 1,
                           step=1, value=viewed_batch,
                           marks={} if batch_size is None else _marks(batch_size),
                           id=VIEWED_BATCH.component_id,
                           updatemode='drag',
                           disabled=batch_size is None),
            ]),
            html.Div(style={'height': '50px', 'width': '100%', 'display': 'inline-block'}, children=[
                'Depth',
                dcc.Slider(min=0,
                           max=0 if resolution3d is None else resolution3d[1] - 1,
                           step=1,
                           value=viewed_depth,
                           marks={} if resolution3d is None else _marks(resolution3d[1]),
                           id=VIEWED_DEPTH.component_id,
                           updatemode='drag',
                           disabled=resolution3d is None),
            ]),
        ]),
    ])

    return layout


_COMPONENTS = ['vec2', 'x', 'y', 'z', 'length']
_COMPONENT_LABELS = ['v', 'x', 'y', 'z', '|.|']
_COMPONENT_DICT = {name: label for name, label in zip(_COMPONENTS, _COMPONENT_LABELS)}


def detect_slices(app):
    assert isinstance(app, App)
    batch_size = resolution3d = None
    for fieldname in app.fieldnames:
        if batch_size is None or resolution3d is None:
            field = app.get_field(fieldname)
            if batch_size is None:
                batch_size = _batch_size_of_field(field)
            if resolution3d is None:
                resolution3d = _resolution3d_of_field(field)
    return batch_size, resolution3d


def _batch_size_of_field(field):
    if isinstance(field, Field):
        if field._batch_size is not None:
            return field._batch_size
        elif isinstance(field, CenteredGrid) and field.data.shape[0] != 1:
            return field.data.shape[0]
        elif isinstance(field, StaggeredGrid):
            children_batch_sizes = [_batch_size_of_field(f) for f in field.unstack()]
            for bs in children_batch_sizes:
                if bs is not None:
                    return bs
    if isinstance(field, numpy.ndarray):
        if field.ndim > 1 and field.shape[0] != 1:
            return field.shape[0]
    return None


def _resolution3d_of_field(field):
    if isinstance(field, Field) and field.rank >= 3:
        if isinstance(field, (CenteredGrid, StaggeredGrid)):
            return field.resolution[-3:]
    if isinstance(field, numpy.ndarray) and field.ndim >= 5:
        return field.shape[-4:-1]
    return None


def _marks(stop, limit=35, step=1):
    if stop <= limit * step:
        return {i: str(i) for i in range(0, stop, step)}
    if stop <= 2 * limit * step:
        return _marks(stop, limit, step * 2)
    if stop <= 5 * limit * step:
        return _marks(stop, limit, step * 5)
    else:
        return _marks(stop, limit, step * 10)
