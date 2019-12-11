import numpy as np

import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

import webglviewer
from phi.viz.dash.webgl_util import default_sky, load_sky


app = dash.Dash(__name__)

dim = [50, 50, 50]
app.layout = html.Div([
    webglviewer.Webglviewer(
        id='viewer',
        sky=default_sky(),
        material_type="SOLID",
        representation_type="DENSITY",  # SDF, PARTICLE, DENSITY
    ),
    html.Div(id='output'),
    dcc.Interval(id='interval', interval=1000)
])


@app.callback(Output('viewer', 'data'), [Input('interval', 'n_intervals')])
def display_output(n_intervals):
    if n_intervals is None:
        n_intervals = 0
    n_intervals = n_intervals % 5
    grid = np.zeros((dim[0], dim[1], dim[2]), dtype="float32")
    grid[n_intervals:n_intervals + int(dim[0]*0.3), n_intervals:n_intervals + int(dim[0]*0.3), n_intervals:n_intervals + int(dim[0]*0.3)] = 0.5
    return grid


@app.callback(Output('viewer', 'sky'), [Input('interval', 'n_intervals')])
def load_real_sky(n_intervals):
    if n_intervals == 1:
        return load_sky('sky0.hdr')
    else:
        raise PreventUpdate()


app.run_server(debug=True, port=8051)
