
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from phi.viz.plot import FRONT, RIGHT, TOP


VIEWED_BATCH = Input('batch-slider', 'value')
VIEWED_DEPTH = Input('depth-slider', 'value')
PROJECTION_AXIS = Input('projection-select', 'value')
VIEWED_COMPONENT = Input('component-slider', 'value')
REFRESH_BUTTON = Input('refresh-button', 'n_clicks')

VIEW_SETTINGS = (VIEWED_BATCH, VIEWED_DEPTH, PROJECTION_AXIS, VIEWED_COMPONENT, REFRESH_BUTTON)

REFRESH_RATE = Input('refresh-rate-slider', 'value')


def parse_view_settings(*args):
    batch = args[0]
    depth = args[1]
    projection = args[2]  # type: str
    component = _COMPONENTS[args[3]]  # type: str
    return {
        'batch': batch,
        'depth': depth,
        'projection': projection,
        'component': component,
    }


def refresh_rate_ms(refresh_value):
    if refresh_value is None:
        return 1000 * 60 * 60 * 24
    else:
        return (2000, 900, 400, 200)[refresh_value]


def build_view_selection(dashapp, viewed_batch=0, viewed_depth=0, viewed_component='length'):

    layout = html.Div(style={'width': '100%', 'display': 'inline-block', 'backgroundColor': '#E0E0FF', 'vertical-align': 'middle'}, children=[
        # --- Settings ---
        html.Div(style={'width': '30%', 'display': 'inline-block'}, children=[
            html.Div(style={'width': '35%'}, children=[
                dcc.Dropdown(options=[{'value': FRONT, 'label': 'Front'}, {'value': RIGHT, 'label': 'Side'}, {'value': TOP, 'label': 'Top'}], value='front', id=PROJECTION_AXIS.component_id),
            ]),
            html.Div(style={'width': '35%', 'height': '50px', 'display': 'inline-block'}, children=[
                'Component',
                dcc.Slider(min=0, max=4, step=1, value=_COMPONENTS.index(viewed_component), marks={0: 'v', 4: '|.|', 1: 'x', 2: 'y', 3: 'z'}, id='component-slider', updatemode='drag'),
            ]),
            html.Div(style={'width': '30%', 'height': '50px', 'display': 'inline-block', 'margin-left': 40}, children=[
                html.Button('Refresh', id=REFRESH_BUTTON.component_id, style={'margin-bottom': 20}),
                'Refresh Rate',
                dcc.Slider(min=0, max=3, step=1, value=1, marks={0: 'low', 3: 'high'}, id=REFRESH_RATE.component_id, updatemode='drag'),
            ])
        ]),
        # --- Batch & Depth ---
        html.Div(style={'width': '70%', 'display': 'inline-block'}, children=[
            html.Div(style={'height': '50px', 'width': '100%', 'display': 'inline-block'}, children=[
                'Batch',
                dcc.Slider(min=0, max=dashapp.batch_count-1, step=1, value=viewed_batch, marks={0: '0'}, id=VIEWED_BATCH.component_id, updatemode='drag'),
            ]),
            html.Div(style={'height': '50px', 'width': '100%', 'display': 'inline-block'}, children=[
                'Depth',
                dcc.Slider(min=0, max=viewed_batch, step=1, value=viewed_depth, marks={0: '0'}, id=VIEWED_DEPTH.component_id, updatemode='drag'),
            ]),
        ]),
    ])
    return layout


_COMPONENTS = ['vec2', 'x', 'y', 'z', 'length']
_COMPONENT_LABELS = ['v', 'x', 'y', 'z', '|.|']
_COMPONENT_DICT = {name: label for name, label in zip(_COMPONENTS, _COMPONENT_LABELS)}
