import numpy as np

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from .dash_app import DashApp
from .._vis_base import display_name, value_range, is_log_control


def all_controls(app: DashApp):
    return tuple(Input(control.name, 'value') for control in app.model.controls)


def build_model_controls(app: DashApp):
    controls = app.model.controls
    if not controls:
        return html.Div()
    model_floats = [control for control in controls if control.control_type == float]
    model_bools = [control for control in controls if control.control_type == bool]
    model_ints = [control for control in controls if control.control_type == int]
    model_texts = [control for control in controls if control.control_type == str]

    # MODEL_ACTIONS.extend([Input(action.name, 'n_clicks') for action in actions])

    layout = html.Div(style={'width': '75%', 'margin-left': 'auto', 'margin-right': 'auto', 'background-color': '#F0F0F0'}, children=[
        html.Div(id='control-div', style={'width': '95%', 'height': '90%', 'margin-left': 'auto', 'margin-right': 'auto', 'margin-top': 15, 'margin-bottom': 'auto'}, children=[
            dcc.Interval(id='initialize-controls', interval=200, max_intervals=1)
        ]),
    ])

    @app.dash.callback(Output('control-div', 'children'), [Input('initialize-controls', 'n_intervals')])
    def build_controls(_):
        model_sliders_float = create_sliders(model_floats)
        model_sliders_int = create_sliders(model_ints)
        model_checkboxes = [dcc.Checklist(options=[{'label': display_name(control.name), 'value': control.name}], value=[control.name] if control.initial else [], id=control.name)
                            for control in model_bools]
        model_textfields = []
        for control in model_texts:
            if not control.value_range:
                text_area = dcc.Textarea(placeholder=control.initial, id=control.name, value=control.initial, rows=1, style={'width': '600px', 'display': 'inline-block'})
                model_textfields.append(html.Div([display_name(control.name) + '  ', text_area]))
            else:
                options = [{'label': o, 'value': o} for o in value_range(control)[1]]
                dropdown = dcc.Dropdown(id=control.name, options=options, value=control.initial, style={'display': 'inline-block', 'width': 200})
                model_textfields.append(html.Div([display_name(control.name) + '  ', dropdown]))
        return [
            dcc.Markdown('### Model'),
            *model_sliders_float,
            *model_sliders_int,
            *model_textfields,
            *model_checkboxes
        ]

    for control in model_floats:
        @app.dash.callback(Output(control.name, 'disabled'), [Input(control.name, 'value')])
        def set_model_value(slider_value, control=control):
            if is_log_control(control):
                value = 10.0 ** slider_value
                if value * 0.99 <= value_range(control)[0]:
                    value = value_range(control)[0]
            else:
                value = slider_value
            app.model.set_control_value(control.name, value)
            return False

    for control in model_ints:
        @app.dash.callback(Output(control.name, 'step'), [Input(control.name, 'value')])
        def set_model_value(value, control=control):
            app.model.set_control_value(control.name, value)
            return 1

    for control in model_bools:
        @app.dash.callback(Output(control.name, 'style'), [Input(control.name, 'value')])
        def set_model_bool(values, control=control):
            app.model.set_control_value(control.name, True if values else False)
            return {}

    for control in model_texts:
        @app.dash.callback(Output(control.name, 'disabled'), [Input(control.name, 'value')])
        def set_model_text(value, control=control):
            if value is not None:
                app.model.set_control_value(control.name, value)
            return False

    return layout


def create_sliders(controls):
    sliders = []
    for control in controls:
        val = control.initial
        lower, upper = value_range(control)
        use_log = is_log_control(control)
        if use_log:
            magn = np.log10(val)
            slider_min = np.log10(lower)
            slider_max = np.log10(upper)
            stepsize_magn = 0.1
            marks = {e: '{:.1e}'.format(np.power(10.0, e)) for e in range(-20, 20) if slider_min <= e <= slider_max}
            slider = dcc.Slider(min=slider_min, max=slider_max, value=magn, id=control.name, step=stepsize_magn, updatemode='drag', marks=marks)
        else:
            if control.control_type == int:
                marks = {v: str(v) for v in range(lower, upper + 1)}
                step = 1
            else:
                marks = {float(v): str(round(v, 4)) for v in np.linspace(lower, upper, 21)}
                step = (upper-lower) / (len(marks)-1)
            slider = dcc.Slider(min=lower, max=upper, value=val, id=control.name, step=step, marks=marks, updatemode='drag')
        slider_container = html.Div(style={'height': '50px', 'width': '100%', 'display': 'inline-block'}, children=[display_name(control.name), slider])
        sliders.append(slider_container)
    return sliders
