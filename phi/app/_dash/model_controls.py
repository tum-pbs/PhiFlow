import numpy as np

import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from .dash_app import DashApp


MODEL_CONTROLS = []
MODEL_ACTIONS = []


def build_model_controls(dashapp):
    assert isinstance(dashapp, DashApp)
    controls = dashapp.app.controls
    actions = dashapp.app.actions

    if len(controls) == len(actions) == 0:
        return html.Div()

    model_floats = [control for control in controls if control.type == 'float']
    model_bools = [control for control in controls if control.type == 'bool']
    model_ints = [control for control in controls if control.type == 'int']
    model_texts = [control for control in controls if control.type == 'text']

    MODEL_CONTROLS.clear()
    MODEL_CONTROLS.extend([Input(control.id, 'n_clicks') for control in actions])
    MODEL_CONTROLS.extend([Input(control.id, 'value') for control in model_floats])
    MODEL_CONTROLS.extend([Input(control.id, 'value') for control in model_ints])
    MODEL_CONTROLS.extend([Input(control.id, 'value') for control in model_bools])
    MODEL_ACTIONS.clear()
    MODEL_ACTIONS.extend([Input(action.id, 'n_clicks') for action in actions])

    layout = html.Div(style={'width': '75%', 'margin-left': 'auto', 'margin-right': 'auto', 'background-color': '#F0F0F0'}, children=[
        html.Div(id='control-div', style={'width': '95%', 'height': '90%', 'margin-left': 'auto', 'margin-right': 'auto', 'margin-top': 15, 'margin-bottom': 'auto'}, children=[
            dcc.Interval(id='initialize-controls', interval=200, max_intervals=1)
        ]),
    ])

    @dashapp.dash.callback(Output('control-div', 'children'), [Input('initialize-controls', 'n_intervals')])
    def build_controls(_):
        model_buttons = [html.Button(control.name, id=control.id) for control in actions]
        model_sliders_float = create_sliders(model_floats)
        model_sliders_int = create_sliders(model_ints)

        model_checkboxes = [dcc.Checklist(options=[{'label': control.name, 'value': control.id}],
                                          value=[control.id] if control.value else [], id=control.id)
                            for control in model_bools]

        model_textfields = []
        for control in model_texts:
            text_area = dcc.Textarea(placeholder=control.value, id=control.id, value=control.value, rows=1,
                                    style={'width': '600px', 'display': 'inline-block'})
            model_textfields.append(html.Div([control.name + '  ', text_area]))
        return [dcc.Markdown('### Model')] + model_sliders_float + model_sliders_int + model_buttons + model_textfields + model_checkboxes

    for action in actions:
        @dashapp.dash.callback(Output(action.id, 'disabled'), [Input(action.id, 'n_clicks')])
        def perform_action(n_clicks, action=action):
            if n_clicks is not None:
                dashapp.app.run_action(action)
            return False

    for control in model_floats:
        @dashapp.dash.callback(Output(control.id, 'disabled'), [Input(control.id, 'value')])
        def set_model_value(slider_value, control=control):
            use_log = False if control.type == 'int' else control.editable_value.use_log_scale
            if use_log:
                value = 10.0 ** slider_value
                if value * 0.99 <= control.range[0]:
                    value = 0.0
            else:
                value = slider_value
            control.value = value
            return False

    for control in model_ints:
        @dashapp.dash.callback(Output(control.id, 'step'), [Input(control.id, 'value')])
        def set_model_value(value, control=control):
            control.value = value
            return 1

    for control in model_bools:
        @dashapp.dash.callback(Output(control.id, 'style'), [Input(control.id, 'value')])
        def set_model_bool(values, control=control):
            control.value = True if values else False
            return {}

    for control in model_texts:
        @dashapp.dash.callback(Output(control.id, 'disabled'), [Input(control.id, 'value')])
        def set_model_text(value, control=control):
            if value is not None:
                control.value = value
            return False

    return layout


def create_sliders(controls):
    sliders = []
    for control in controls:
        val = control.value
        lower, upper = control.range
        use_log = False if control.type == 'int' else control.editable_value.use_log_scale
        if use_log:
            magn = np.log10(val)
            slider_min = np.log10(lower)
            slider_max = np.log10(upper)
            stepsize_magn = 0.1
            marks = {e: '{:.1e}'.format(np.power(10.0, e)) for e in range(-20, 20) if slider_min <= e <= slider_max}
            slider = dcc.Slider(min=slider_min, max=slider_max, value=magn, id=control.id, step=stepsize_magn, updatemode='drag', marks=marks)
        else:
            if control.type == 'int':
                marks = {v: str(v) for v in range(lower, upper + 1)}
                step = 1
            else:
                marks = {float(v): str(round(v, 4)) for v in np.linspace(lower, upper, 21)}
                step = (upper-lower) / (len(marks)-1)
            slider = dcc.Slider(min=lower, max=upper, value=val, id=control.id, step=step, marks=marks, updatemode='drag')
        slider_container = html.Div(style={'height': '50px', 'width': '100%', 'display': 'inline-block'}, children=[control.name, slider])
        sliders.append(slider_container)
    return sliders
