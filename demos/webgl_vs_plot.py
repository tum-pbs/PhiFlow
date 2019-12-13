import webglviewer
import dash
from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc
import numpy as np


app = dash.Dash(__name__)
app.config.suppress_callback_exceptions = True


app.layout = html.Div([
    html.Button('2d/3D', id='button'),
    html.Div(id='display'),
    dcc.Interval(id='interval', interval=1000)
])

def cubemap(file='cubemap.jpg'):
    """map = np.array(Image.open(file))
    map = np.concatenate((map, np.ones((map.shape[0], map.shape[1], 1))), axis=-1)
    h, w = map.shape[0] // 3, map.shape[1] // 4
    top, bottom, left, front, right, back = \
        (map[:h, w:2*w], map[2*h:, w:2*w], map[h:2*h, :w], map[h:2*h, w:2*w], map[h:2*h, 2*w:3*w], map[h:2*h, 3*w:])
    images = right, left, top, bottom, front, back
    images = [im.flatten() for im in images]"""
    return np.array([[1,0,0,1],[1,0,0,1],[0,1,0,1],[0,1,0,1],[0,0,1,1],[0,0,1,1]])*255

dim = [50, 50, 50]
data = np.ones((10, dim[0], dim[1], dim[2]), dtype="float32")
for i in range(data.shape[0]):
    data[i, i:i + int(dim[0]*0.3), i:i + int(dim[0]*0.3), i:i + int(dim[0]*0.3)] = -1

webgl = webglviewer.Webglviewer(
        id='viewer',
        sky=cubemap(),
        material_type="DARK_SMOKE",
        representation_type="SDF",
        scale=0.1,
        data=data
    )

@app.callback(Output('display', 'children'), [Input('button', 'n_clicks')])
def switch_display(n):
    if n is None: return []
    if n % 2 == 0:
        return dcc.Graph(id='graph')
    else:
        return html.Div([webgl], style={"width":800, "height":600})

@app.callback(Output('viewer', 'idx'), [Input('interval', 'n_intervals')])
def display_output(n_intervals):
    return n_intervals % 10


if __name__ == '__main__':
    app.run_server(debug=True)