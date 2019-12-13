
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

from phi.viz.dash import viewsettings
from phi.viz.dash.dash_plotting import dash_graph_plot, EMPTY_FIGURE
from phi.viz.dash.player_controls import STEP_COMPLETE, REFRESH_INTERVAL
from phi.viz.dash.webgl_util import EMPTY_GRID, webgl_prepare_data, load_sky
from .viewsettings import parse_view_settings, REFRESH_RATE, refresh_rate_ms


# --- Viewer ---

def build_viewer(dashapp, initial_field_name=None, id='viewer'):
    import webglviewer
    from .webgl_util import default_sky

    field_options = [{'label': item, 'value': item} for item in dashapp.app.fieldnames] + [{'label': '<None>', 'value': 'None'}]
    if initial_field_name is None:
        initial_field_name = dashapp.app.fieldnames[0] if len(dashapp.app.fieldnames) > 0 else 'None'

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

    @dashapp.dash.callback(Output(id+'-figure-container', 'children'), [Input(id+'-3d', 'value')])
    def choose_viewer(list3d):
        if list3d:
            return [
                dcc.Interval(id=id + '-webgl-initializer', interval=100, max_intervals=1),
                webglviewer.Webglviewer(id=id+'-webgl', sky=default_sky(), material_type='LIGHT_SMOKE', representation_type='DENSITY')
            ]
        else:
            return dcc.Graph(figure=EMPTY_FIGURE, id=id + '-graph', style={'height': '100%'})

    @dashapp.dash.callback(Output(id+'-graph', 'figure'), (Input(id+'-field-select', 'value'), STEP_COMPLETE, REFRESH_INTERVAL) + viewsettings.VIEW_SETTINGS)
    def update_figure(field, _0, _1, *settings):
        if field is None or field == 'None':
            return EMPTY_FIGURE
        data = dashapp.app.get_field(field)
        if data is None:
            return EMPTY_FIGURE
        settings_dict = parse_view_settings(*settings)
        return dash_graph_plot(data, settings_dict)

    @dashapp.dash.callback(Output(id+'-webgl', 'data'), (Input(id+'-field-select', 'value'), Input(id+'-webgl-initializer', 'n_intervals'), STEP_COMPLETE, REFRESH_INTERVAL) + viewsettings.VIEW_SETTINGS)
    def update_data(field, _0, _1, _2, *settings):
        if field is None or field == 'None':
            return EMPTY_GRID
        data = dashapp.app.get_field(field)
        if data is None:
            return EMPTY_GRID
        settings_dict = parse_view_settings(*settings)
        return webgl_prepare_data(data, settings_dict)

    return layout
