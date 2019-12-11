# coding=utf-8
from __future__ import print_function

import os.path
import logging
import traceback

import numpy as np

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

from ..plot import FRONT, RIGHT, TOP, VECTOR2, LENGTH, PlotlyFigureBuilder
from ..display import ModelDisplay

class DashFieldSequenceGui(ModelDisplay):

    def __init__(self, field_sequence_model,
                 display=None, depth=0, batch=0,
                 figure_builder=None,
                 framerate=1.0,
                 sequence_count=1,
                 production=False,
                 port=None,
                 tb_port=6006):
        ModelDisplay.__init__(self, field_sequence_model)
        self.model = field_sequence_model
        istf = 'tensorflow' in self.model.traits
        hasmodel = 'model' in self.model.traits
        self.model.prepare()

        self.app = dash.Dash()
        self.figures = PlotlyFigureBuilder(batches=[0], depths=[0]) if not figure_builder else figure_builder
        self.max_depth = depth
        self.max_batch = batch
        self.framerate = framerate
        self.fieldshapes = [[self.max_batch + 1, self.max_depth + 1, 0, 0]] * 2
        self.benchmarking = False  # Disable speed control and graph updates if true
        self.sequence_count = sequence_count
        self.production_server = production
        self.target_port = port
        self.tensorboard_port = tb_port

        field_options = [{'label': item, 'value': item} for item in self.model.fieldnames] + [
            {'label': '<None>', 'value': 'None'}]
        fieldnames = list(map(lambda e: e['value'], field_options))
        if display is not None:
            self.selected_fields = [display, display] if isinstance(display, str) else list(display)
            for i in range(len(self.selected_fields)):
                if self.selected_fields[i] not in self.model.fieldnames:
                    self.selected_fields[i] = 'None'
        else:
            self.selected_fields = [fieldnames[0], fieldnames[min(1, len(fieldnames) - 1)]]

        initial_record_values = []
        if self.model.record_data:
            initial_record_values.append('data')
        if self.model.record_images:
            initial_record_values.append('images')

        controls = field_sequence_model.controls
        model_floats = [control for control in controls if control.type == 'float']
        model_bools = [control for control in controls if control.type == 'bool']
        model_ints = [control for control in controls if control.type == 'int']
        model_texts = [control for control in controls if control.type == 'text']

        actions = field_sequence_model.actions
        model_buttons = [html.Button(control.name, id=control.id) for control in actions]
        model_sliders_float = create_sliders(model_floats)
        model_sliders_int = create_sliders(model_ints)

        model_checkboxes = [dcc.Checklist(options=[{'label': control.name, 'value': control.id}],
                                          value=[control.id] if control.value else [], id=control.id)
                            for control in model_bools]

        model_textfields = []
        for control in model_texts:
            textarea = dcc.Textarea(placeholder=control.value, id=control.id, value=control.value, rows=1,
                                    style={'width': '600px', 'display': 'inline-block'})
            model_textfields.append(html.Div([control.name + '  ', textarea]))

        model_inputs = [Input(control.id, 'n_clicks') for control in field_sequence_model.actions]
        model_inputs += [Input(control.id, 'value') for control in model_floats]
        model_inputs += [Input(control.id, 'value') for control in model_ints]
        model_inputs += [Input(control.id, 'value') for control in model_bools]

        self.app.layout = html.Div([
                                    dcc.Markdown('# %s' % field_sequence_model.name)
                                    ] +
            ([
                 dcc.Markdown("""
---

> **_About this application:_**

%s

---""" % field_sequence_model.subtitle),
             ] if field_sequence_model.subtitle is not None and len(field_sequence_model.subtitle) > 0 else []) + [

                                       html.Div([

                                           html.Div(children=[
                                               html.Div([
                                                   'Batch',
                                                   dcc.Slider(min=0, max=self.max_batch, step=1,
                                                              value=self.figures.batches[0], marks={0: '0'},
                                                              id='batch-slider',
                                                              updatemode='drag'),
                                               ], style={'height': '50px', 'width': '100%', 'display': 'inline-block'}),

                                               html.Div([
                                                   'Depth',
                                                   dcc.Slider(min=0, max=self.max_depth, step=1,
                                                              value=self.figures.depths[0], marks={0: '0'},
                                                              id='depth-slider',
                                                              updatemode='drag'),
                                               ], style={'height': '50px', 'width': '100%', 'display': 'inline-block'}),

                                           ], style={'height': '50px', 'width': '60%', 'display': 'inline-block'}),

                                           html.Div([
                                               dcc.Dropdown(options=[{'value': FRONT, 'label': 'Front'},
                                                                     {'value': RIGHT, 'label': 'Side'},
                                                                     {'value': TOP, 'label': 'Top'}], value='front',
                                                            id='view-select'),

                                               dcc.Checklist(options=[{'label': 'Antisymmetry', 'value': 'selected'},
                                                                      {'label': 'Staggered', 'value': 'staggered'}],
                                                             value=[], id='antisymmetry-checkbox'),
                                               html.Button('Refresh', id='button-refresh'),
                                               html.Div([
                                                   'Component',
                                                   dcc.Slider(min=0, max=4, step=1, value={'vec2': 0, 'length': 4, 0:1, 1:2, 2:3}[self.figures.component],
                                                              marks={0: 'v', 4: '|.|', 1: 'x', 2: 'y', 3: 'z'},
                                                              id='component-slider',
                                                              updatemode='drag'),
                                               ], style={'width': '29%', 'height': '50px', 'display': 'inline-block'})

                                           ], style={'width': '25%', 'display': 'inline-block'}),
                                       ], style={'width': '100%', 'display': 'inline-block',
                                                 'backgroundColor': '#E0E0FF'}),

                                       html.Div([
                                           html.Div([
                                               dcc.Dropdown(options=field_options, value=self.selected_fields[0],
                                                            id='channel-select'),
                                               dcc.Graph(
                                                   id='graph',
                                                   figure=self.empty_figure()
                                               )], style={'width': '50%', 'display': 'inline-block'}),
                                           html.Div([
                                               dcc.Dropdown(options=field_options, value=self.selected_fields[1],
                                                            id='channel-select2'),
                                               dcc.Graph(
                                                   id='graph2',
                                                   figure=self.empty_figure()
                                               )], style={'width': '50%', 'display': 'inline-block'}),
                                       ], style={'width': '100%', 'display': 'inline-block'}),

                                       html.Div(id='statusbar', children=field_sequence_model.status,
                                                style={'backgroundColor': '#E0E0FF'}),

                                       html.Div([
                                           html.Button('Play', id='button-start'),
                                           html.Button('Pause', id='button-pause'),
                                           html.Button('Step', id='button-step'),
                                       ], style={'height': '30px'}),

                                   ] + (([dcc.Markdown('### Model')]
                                         + model_buttons
                                         + model_textfields
                                         + model_checkboxes
                                         + model_sliders_float)
                                        if controls or actions else []) + model_sliders_int +

                                   ([
                                        dcc.Markdown('### TensorFlow Model'),
                                        html.Div([
                                            html.Button('Save Model', id='button-save-model'),
                                            html.Button('Load Model from: ', id='button-load-model'),
                                        ]),
                                        dcc.Textarea(placeholder='Checkpoint location to load',
                                                     id='text-model-load-location',
                                                     value=field_sequence_model.directory + '/checkpoint_', rows=1,
                                                     style={'width': '600px', 'display': 'inline-block'}),
                                        dcc.Markdown('', id='model-info'),

                                        html.Div([
                                            html.Button('Launch TensorBoard', id='launch-tensorboard'),
                                            html.Div([html.A('Open TensorBoard', href='', id='tensorboard-href')]),
                                        ]),
                                    ] if hasmodel else []) + [

                                       html.Div([
                                           dcc.Markdown('### Framerate'),
                                           dcc.Checklist(
                                               options=[{'label': 'Enable framerate control', 'value': 'selected'}],
                                               value=['selected'], id='framerate-checkbox'),
                                           html.Div([
                                               dcc.Slider(min=0.5, max=15, step=0.5, value=self.framerate,
                                                          marks={i: '%.1f' % i for i in np.linspace(0.5, 15, 30)},
                                                          id='framerate-slider', updatemode='drag'),

                                           ], style={'height': '32px', 'width': '60%', 'display': 'inline-block'}),
                                           html.Div([
                                               'Substeps: ',
                                               dcc.Textarea(placeholder='1', id='stride',
                                                            value=str(self.model.sequence_stride), rows=1,
                                                            style={'width': '100px', 'display': 'inline-block'})
                                               # dcc.Slider(min=1, max=50, step=1, value=stride_slider_val,
                                               #            marks={i: str(i ** 2) for i in range(1, 51)},
                                               #            id='sequence-stride', updatemode='drag'),
                                           ]),
                                       ]),

                                       html.Div([
                                                    dcc.Markdown('### Sequence'),
                                                    html.Button('Run sequence', id='button-sequence'),
                                                    html.Button('Benchmark sequence', id='button-benchmark'),
                                                ] + ([
                                                         html.Button('Profile sequence', id='button-profile'),
                                                     ] if istf else []) + [

                                                    html.Div([
                                                        'Sequence length: ',
                                                        dcc.Textarea(placeholder='100', id='sequence-count',
                                                                     value=str(sequence_count), rows=1,
                                                                     style={'width': '100px',
                                                                            'display': 'inline-block'})
                                                    ]),

                                                    dcc.Markdown(children=' ', id='run-statistics'),

                                                ]),

                                       html.Div([
                                           html.Div([
                                               dcc.Markdown('### Record  \n' + os.path.abspath(self.model.directory)),
                                               dcc.Checklist(options=[{'value': 'images', 'label': 'Images'},
                                                                      {'value': 'data', 'label': 'Data'}],
                                                             value=initial_record_values, id='record-types',
                                                             style={'width': '300px', 'display': 'inline-block'}),
                                           ]),
                                           html.Div([
                                               'Fields to record: ',
                                               dcc.Checklist(options=field_options, value=self.model.recorded_fields,
                                                             style={'width': '80%', 'display': 'inline-block'},
                                                             id='rec-fields'),
                                           ]),
                                           'Image generation',
                                           dcc.Checklist(
                                               options=[{'label': 'All slices (depth)', 'value': 'all-slices'},
                                                        {'label': 'All batches', 'value': 'all-batches'}], value=[],
                                               style={'width': '300px', 'display': 'inline-block'}, id='rec-slices'),
                                           html.Div([
                                               html.Button('Write current frame', id='button-write-frame'),
                                           ]),
                                           html.Div([
                                               html.Button('Create animation using existing npz files', id='button-animate'),
                                           ]),
                                           html.Div([
                                               'Animation FPS: ',
                                               dcc.Textarea(placeholder='30', id='animation_fps', value=str(self.model.animation_fps), rows=1,
                                                            style={'width': '100px', 'display': 'inline-block'})
                                           ]),
                                       ]),

                                       html.Div(
                                           'This interface belongs to the PhiFlow project, developed by Philipp Holl.',
                                           style={'font-size': '60%'}),

                                       dcc.Interval(id='interval', interval=1000),
                                       dcc.Interval(id='status-interval', interval=200),
                                       html.Div(id='step-complete', style={'display': 'none'})
                                   ], style={'fontFamily': 'arial'})

        @self.app.callback(Output('statusbar', 'children'), [Input('status-interval', 'n_intervals')])
        def update_statusbar(n_intervals):
            return self.model.status

        @self.app.callback(Output('graph', 'figure'),
                           inputs=[Input('channel-select', 'value'),
                                   Input('batch-slider', 'value'),
                                   Input('depth-slider', 'value'),
                                   Input('component-slider', 'value'),
                                   Input('view-select', 'value'),
                                   Input('interval', 'n_intervals'),
                                   Input('step-complete', 'children'),
                                   Input('button-refresh', 'n_clicks'),
                                   Input('antisymmetry-checkbox', 'value')]
                                  + model_inputs)
        def update_graph1(fieldname, batch, depth, component, view, *kwargs):
            self.selected_fields[0] = fieldname
            return self.create_figure(0, view, batch, depth, component)

        @self.app.callback(Output('graph2', 'figure'),
                           inputs=[Input('channel-select2', 'value'),
                                   Input('batch-slider', 'value'),
                                   Input('depth-slider', 'value'),
                                   Input('component-slider', 'value'),
                                   Input('view-select', 'value'),
                                   Input('interval', 'n_intervals'),
                                   Input('step-complete', 'children'),
                                   Input('button-refresh', 'n_clicks'),
                                   Input('antisymmetry-checkbox', 'value')]
                                  + model_inputs)
        def update_graph2(fieldname, batch, depth, component, view, *kwargs):
            self.selected_fields[1] = fieldname
            return self.create_figure(1, view, batch, depth, component)

        @self.app.callback(Output('button-start', 'style'), [Input('button-start', 'n_clicks')])
        def start_simulation(n_clicks):
            if n_clicks and not self.model.running:
                self.play()
            return {}

        @self.app.callback(Output('button-pause', 'style'), [Input('button-pause', 'n_clicks')])
        def pause_simulation(n_clicks):
            if n_clicks:
                self.model.pause()
            return {}

        @self.app.callback(Output('step-complete', 'children'), [Input('button-step', 'n_clicks')])
        def simulation_step(n_clicks):
            if n_clicks is None:
                return ['init']
            if not self.model.running:
                self.model.run_step()
                return [str(n_clicks)]
            else:
                raise PreventUpdate()

        @self.app.callback(Output('framerate-slider', 'disabled'),
                           [Input('framerate-slider', 'value'), Input('framerate-checkbox', 'value')])
        def set_framerate(value, enabled):
            if value is not None and enabled is not None:
                self.framerate = value if enabled else None
            if enabled:
                return False
            else:
                return True

        @self.app.callback(Output('antisymmetry-checkbox', 'style'), [Input('antisymmetry-checkbox', 'value')])
        def set_antisymmetry(checked):
            self.figures.antisymmetry = 'selected' in checked
            self.figures.staggered = 'staggered' in checked
            return {}

        for action in field_sequence_model.actions:
            @self.app.callback(Output(action.id, 'disabled'), [Input(action.id, 'n_clicks')])
            def perform_action(n_clicks, action=action):
                if (n_clicks is not None):
                    self.model.run_action(action)
                return False

        for control in model_floats:
            @self.app.callback(Output(control.id, 'disabled'), [Input(control.id, 'value')])
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
            @self.app.callback(Output(control.id, 'step'), [Input(control.id, 'value')])
            def set_model_value(value, control=control):
                control.value = value
                return 1

        for control in model_bools:
            @self.app.callback(Output(control.id, 'style'), [Input(control.id, 'value')])
            def set_model_bool(values, control=control):
                control.value = True if values else False
                return {}

        for control in model_texts:
            @self.app.callback(Output(control.id, 'disabled'), [Input(control.id, 'value')])
            def set_model_text(value, control=control):
                if value is not None:
                    control.value = value
                return False

        @self.app.callback(Output('depth-slider', 'max'),
                           [Input('depth-slider', 'value'), Input('interval', 'n_intervals')])
        def update_depth_slider(depth, n_intervals):
            if depth is None:
                return self.max_depth
            # Update depth
            self.figures.select_depth(depth)
            # Return maximum
            self.max_depth = max(self.figures.slice_count(self.fieldshapes[0]),
                                 self.figures.slice_count(self.fieldshapes[1])) - 1
            return self.max_depth

        @self.app.callback(Output('depth-slider', 'marks'), [Input('interval', 'n_intervals')])
        def depth_ticks(n_intervals):
            return {i: '{}'.format(i) for i in range(self.max_depth)}

        @self.app.callback(Output('batch-slider', 'max'),
                           inputs=[Input('batch-slider', 'value'), Input('interval', 'n_intervals')])
        def update_batch_slider(batch, n_intervals):
            if batch is None:
                return self.max_batch

            # Task 1: Update depth
            self.figures.select_batch(batch)

            # Task 2: Return maximum
            def batch_of(shape):
                return shape[0]

            self.max_batch = max(batch_of(self.fieldshapes[0]), batch_of(self.fieldshapes[1])) - 1
            return self.max_batch

        @self.app.callback(Output('batch-slider', 'marks'), [Input('interval', 'n_intervals')])
        def batch_ticks(n_intervals):
            return {i: '{}'.format(i) for i in range(self.max_batch)}

        @self.app.callback(Output('interval', 'interval'),
                           inputs=[Input('button-start', 'n_clicks'), Input('button-pause', 'n_clicks'),
                                   Input('button-refresh', 'n_clicks'), Input('status-interval', 'n_intervals')])
        def set_update_timer(n1, n2, n3, n_intervals):
            if self.model.running and not self.benchmarking:
                if self.framerate is None:
                    return 100
                framedelay = 1000 / self.framerate
                return max(framedelay, 100)
            else:
                return 60 * 60 * 24

        @self.app.callback(Output('sequence-count', 'disabled'),
                           [Input('sequence-count', 'value'), Input('stride', 'value')])
        def update_sequence_params(count, stride):
            if count is not None:
                try:
                    self.sequence_count = int(count)
                except ValueError:
                    pass  # Invalid sequence count
            if stride is not None:
                try:
                    self.model.sequence_stride = int(stride)
                except ValueError:
                    pass  # Invalid stride
            return False

        @self.app.callback(Output('record-types', 'style'),
                           [Input('record-types', 'value'),
                            Input('rec-fields', 'value'),
                            Input('rec-slices', 'value'),
                            Input('batch-slider', 'value'),
                            Input('depth-slider', 'value')])
        def update_record_params(types, fields, image_count_selection, batch, depth):
            batch = None if 'all-batches' in image_count_selection else batch
            depth = None if 'all-slices' in image_count_selection else depth
            self.model.config_recording('images' in types, 'data' in types, fields)
            self.model.figures.select_batch(batch)
            self.model.figures.select_depth(depth)
            self.model.invalidate()
            return {}

        @self.app.callback(Output('button-write-frame', 'disabled'), [Input('button-write-frame', 'n_clicks')])
        def write_current_frame(n):
            if n is not None:
                self.model.record_frame()
            return False

        @self.app.callback(Output('animation_fps', 'disabled'),
                           [Input('animation_fps', 'value')])
        def update_animation_fps(fps):
            if fps is not None:
                try:
                    self.model.animation_fps = int(fps)
                except ValueError:
                    pass  # Invalid stride
            return False

        @self.app.callback(Output('button-animate', 'disabled'), [Input('button-animate', 'n_clicks')])
        def animate_data(n):
            if n is not None:
                self.model.animate()
            return False

        @self.app.callback(Output('button-sequence', 'style'), [Input('button-sequence', 'n_clicks')])
        def write_sequence(n1):
            if n1 and not self.model.running:
                self.model.play(max_steps=self.sequence_count * self.model.sequence_stride)
            return {}

        if not istf:
            @self.app.callback(Output('run-statistics', 'children'), [Input('button-benchmark', 'n_clicks')])
            def benchmark_sequence(n1):
                if n1 is None:
                    return [' ']
                if self.model.running:
                    return ['App is running.']
                target_count = self.sequence_count * self.model.sequence_stride
                self.benchmarking = True
                step_count, time_elapsed = self.model.benchmark(target_count)
                self.benchmarking = False
                output = '### Benchmark Results\n'
                if step_count != target_count:
                    output += 'The benchmark was stopped prematurely.  \n'
                if self.model.record_data or self.model.record_images:
                    output += 'Recording was disabled during benchmark.  \n'
                output += 'Finished %d steps in %.03f seconds.' % (step_count, time_elapsed)
                output += '  \n*Average*: %.04f seconds per step, %.02f steps per second.' % (
                time_elapsed / step_count, step_count / time_elapsed)
                return output

        if hasmodel:
            @self.app.callback(Output('tensorboard-href', 'href'), [Input('launch-tensorboard', 'n_clicks')])
            def launch_tb(n1):
                if not n1:
                    return
                logging.info('Launching TensorBoard...')
                logdir = field_sequence_model.session.summary_directory
                import phi.tf.profiling as profiling
                url = profiling.launch_tensorboard(logdir, port=self.tensorboard_port)
                logging.info('TensorBoard launched, URL: %s' % url)
                return url

        if istf:
            @self.app.callback(Output('run-statistics', 'children'),
                               [Input('button-benchmark', 'n_clicks_timestamp'),
                                Input('button-profile', 'n_clicks_timestamp')])
            def benchmark_sequence(bench, prof):
                if not bench and not prof:
                    return [' ']
                if self.model.running:
                    return ['App is running.']
                target_count = self.sequence_count * self.model.sequence_stride
                profile = True if not bench else (False if not prof else prof > bench)
                self.benchmarking = True
                if profile:
                    field_sequence_model.session.tracing = True
                    with field_sequence_model.session.profiler() as profiler:
                        timeline_file = profiler.timeline_file
                        step_count, time_elapsed = self.model.benchmark(target_count)
                else:
                    step_count, time_elapsed = self.model.benchmark(target_count)
                self.benchmarking = False
                if profile: field_sequence_model.session.tracing = False
                output = '### Benchmark Results\n'
                if step_count != target_count:
                    output += 'The benchmark was stopped prematurely.  \n'
                if self.model.record_data or self.model.record_images:
                    output += 'Recording was disabled during benchmark.  \n'
                output += 'Finished %d steps in %.03f seconds.' % (step_count, time_elapsed)
                output += '  \n*Average*: %.04f seconds per step, %.02f steps per second.' % (
                    time_elapsed / step_count, step_count / time_elapsed)
                if profile:
                    output += '  \nProfile saved. Open  \n*chrome://tracing/*  \n and load file  \n *%s*' % timeline_file
                return output

        if hasmodel:
            @self.app.callback(Output('model-info', 'children'),
                               [Input('button-save-model', 'n_clicks_timestamp'),
                                Input('button-load-model', 'n_clicks_timestamp'),
                                Input('text-model-load-location', 'value')])
            def save_model(save, load, load_path):
                if not save and not load:
                    return ''
                save = True if not load else (False if not save else save > load)
                if save:
                    try:
                        path = self.model.save_model()
                        return 'Model at time %d saved to:    %s' % (self.model.steps, path)
                    except Exception as e:
                        traceback.print_exc()
                        return 'Saving model failed: %s  \nSee console for details.' % e
                else:
                    try:
                        self.model.load_model(load_path)
                        return '%d: Model loaded from %s' % (self.model.steps, load_path)
                    except Exception as e:
                        traceback.print_exc()
                        return 'Loading model failed: %s  \nSee console for details.' % e

    def show(self, port=None, use_reloader=False):
        port = self.target_port if port is None else port
        if not self.production_server:
            if port is None: port = 8051
            print('Starting Dash server on http://localhost:%d/' % port)
            self.app.run_server(debug=True, host='0.0.0.0', port=port, use_reloader=use_reloader)
            return self
        else:
            if port is not None: logging.warning('Port request %d ignored because production server used.')
            return self.app.server

    def play(self):
        self.model.play(framerate=self.framerate)

    def create_figure(self, figindex, view, batch, depth, component):
        self.figures.view = view
        self.figures.select_batch(batch)
        self.figures.select_depth(depth)
        if component == 0:
            self.figures.component = VECTOR2
        elif component == 4:
            self.figures.component = LENGTH
        else:
            self.figures.component = component - 1

        fieldname = self.selected_fields[figindex]
        if fieldname is None or fieldname == 'None':
            self.fieldshapes[figindex] = [0, 0]
            return self.empty_figure()

        data = self.model.get_field(fieldname)
        if data is None:
            self.fieldshapes[figindex] = [0, 0]
            return self.empty_figure()
        self.fieldshapes[figindex] = self.figures.slice_dims(data)
        return self.figures.create_figure(data, library='dash')

    def empty_figure(self):
        figure = self.figures.empty_figure(library='dash')
        figure.update({'layout': {'height': 700}})
        return figure




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
            marks = {e: '{:.1e}'.format(np.power(10.0, e)) for e in range(-20, 20) if
                     e >= slider_min and e <= slider_max}
            slider = dcc.Slider(min=slider_min, max=slider_max, value=magn, id=control.id, step=stepsize_magn,
                                updatemode='drag', marks=marks)
        else:
            if control.type == 'int':
                marks = {v: str(v) for v in range(lower, upper + 1)}
                step = 1
            else:
                marks = {float(v): str(round(v, 4)) for v in np.linspace(lower, upper, 21)}
                step = (upper-lower) / (len(marks)-1)
            slider = dcc.Slider(min=lower, max=upper, value=val, id=control.id, step=step, marks=marks, updatemode='drag')
        slider_container = html.Div([control.name, slider],
                                    style={'height': '50px', 'width': '80%', 'display': 'inline-block'})
        sliders.append(slider_container)
    return sliders
