# coding=utf-8
from __future__ import print_function
import logging, os, numbers, six, numpy, threading, inspect, time, sys
from os.path import isfile
from phi.data.fluidformat import Scene
import phi.math.nd
from phi.math import struct
from phi.viz.plot import PlotlyFigureBuilder
from phi.physics.world import world


def synchronized_method(method):
    outer_lock = threading.Lock()
    lock_name = '__' + method.__name__ + '_lock' + '__'

    def sync_method(self, *args, **kws):
        with outer_lock:
            if not hasattr(self, lock_name): setattr(self, lock_name, threading.Lock())
            lock = getattr(self, lock_name)
            with lock:
                return method(self, *args, **kws)

    return sync_method


class TimeDependentField(object):

    def __init__(self, name, generator):
        self.name = name
        self.generator = generator
        self.array = None
        self.invalidation_version = -1

    @synchronized_method
    def get(self, invalidation_version):
        if invalidation_version != self.invalidation_version:
            self.array = self.generator()
            self.invalidation_version = invalidation_version
        return self.array


class FieldSequenceModel(object):

    def __init__(self,
                 name=u'*Φ-Flow* Application',
                 subtitle='',
                 fields=None,
                 stride=1,
                 record_images=False, record_data=False,
                 base_dir='~/phi/data/',
                 recorded_fields=None,
                 summary=None,
                 custom_properties=None,
                 target_scene=None,
                 objects_to_save=None):
        self.name = name
        self.subtitle = subtitle
        self.summary = summary if summary else name
        if fields:
            self.fields = {name: TimeDependentField(name, generator) for (name,generator) in fields.items()}
        else:
            self.fields = {}
        self.message = None
        self.steps = 0
        self._invalidation_counter = 0
        self._controls = []
        self._actions = []
        self._traits = []
        self.prepared = False
        self.current_action = None
        self._pause = False
        self.world = world
        # Setup directory & Logging
        self.objects_to_save = [ self.__class__ ] if objects_to_save is None else list(objects_to_save)
        self.base_dir = os.path.expanduser(base_dir)
        if not target_scene:
            self.new_scene()
            self.uses_existing_scene = False
        else:
            self.scene = target_scene
            self.uses_existing_scene = True
        if not isfile(self.scene.subpath('info.log')):
            logfile = self.scene.subpath('info.log')
        else:
            index = 2
            while True:
                logfile = self.scene.subpath('info_%d.log'%index)
                if not isfile(logfile): break
                else: index += 1
        # Setup logging
        logFormatter = logging.Formatter('%(message)s (%(levelname)s), %(asctime)sn\n')
        rootLogger = logging.getLogger()
        rootLogger.setLevel(logging.WARNING)
        customLogger = logging.Logger('app', logging.DEBUG)
        fileHandler = logging.FileHandler(logfile)
        fileHandler.setFormatter(logFormatter)
        customLogger.addHandler(fileHandler)
        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setFormatter(logFormatter)
        consoleHandler.setLevel(logging.INFO)
        customLogger.addHandler(consoleHandler)
        self.logger = customLogger
        print('Scene directory is %s' % self.scene.path)
        # Recording
        self.record_images = record_images
        self.record_data = record_data
        self.recorded_fields = recorded_fields if recorded_fields is not None else []
        self.rec_all_slices = False
        self.sequence_stride = stride
        self._custom_properties = custom_properties if custom_properties else {}
        self.figures = PlotlyFigureBuilder()
        self.info('Setting up model...')

    def new_scene(self):
        self.scene = Scene.create(self.base_dir, self.scene_summary(),
                                  count=1 if self.world.batch_size is None else self.world.batch_size, mkdir=True)

    @property
    def directory(self):
        return self.scene.path

    @property
    def image_dir(self):
        return self.scene.subpath('images')

    def get_image_dir(self):
        return self.scene.subpath('images', create=True)

    def progress(self):
        self.step()
        self.steps += 1
        self.invalidate()

    def invalidate(self):
        self._invalidation_counter += 1

    def step(self):
        world.step()

    @property
    def fieldnames(self):
        return sorted(self.fields.keys())

    def get_field(self, fieldname):
        if not fieldname in self.fields:
            raise KeyError('Field %s not declared. Available fields are %s' % (fieldname, self.fields.keys()))
        return self.fields[fieldname].get(self._invalidation_counter)

    def add_field(self, name, generator):
        assert not self.prepared, 'Cannot add fields to a prepared model'
        if not callable(generator):
            assert isinstance(generator, numpy.ndarray)
            array = generator
            generator = lambda: array
        self.fields[name] = TimeDependentField(name, generator)

    @property
    def actions(self):
        return self._actions

    def add_action(self, name, methodcall):
        self._actions.append(Action(name, methodcall, name))

    def run_action(self, action):
        message_before = self.message
        action.method()
        self.invalidate()
        message_after = self.message
        if message_before == message_after:
            if self.message is None or self.message == '':
                self.message = display_name(action.name)
            else:
                self.message += ' | ' + display_name(action.name)

    @property
    def traits(self):
        return self._traits

    def add_trait(self, trait):
        assert not self.prepared, 'Cannot add traits to a prepared model'
        self._traits.append(trait)

    @property
    def controls(self):
        return self._controls

    def prepare(self):
        if self.prepared:
            return
        logging.info('Gathering model data...')
        self.prepared = True
        # Controls
        for name in self.__dict__:
            val = getattr(self, name)
            editable_value = None
            if isinstance(val, EditableValue):
                editable_value = val
                setattr(self, name, val.initial_value) # Replace EditableValue with initial value
            elif name.startswith('value_'):
                value_name = display_name(name[6:])
                dtype = type(val)
                if dtype == bool:
                    editable_value = EditableBool(value_name, val)
                elif isinstance(val, numbers.Integral): # Int
                    editable_value = EditableInt(value_name, val)
                elif isinstance(val, numbers.Number): # Float
                    editable_value = EditableFloat(value_name, val)
                elif isinstance(val, six.string_types):
                    editable_value = EditableString(value_name, val)
            if editable_value:
                self._controls.append(Control(self, name, editable_value))
        # Actions
        for method_name in dir(self):
            if method_name.startswith('action_') and callable(getattr(self, method_name)):
                self._actions.append(Action(display_name(method_name[7:]), getattr(self, method_name), method_name))
        # Scene
        self._update_scene_properties()
        source_files_to_save = set()
        for object in self.objects_to_save:
            source_files_to_save.add(inspect.getabsfile(object))
        for source_file in source_files_to_save:
            self.scene.copy_src(source_file)
        return self

    def add_custom_property(self, key, value):
        self._custom_properties[key] = value
        if self.prepared: self._update_scene_properties()

    def add_custom_properties(self, dict):
        self._custom_properties.update(dict)
        if self.prepared: self._update_scene_properties()

    def _update_scene_properties(self):
        if self.uses_existing_scene: return
        app_name = os.path.basename(inspect.getfile(self.__class__))
        app_path = inspect.getabsfile(self.__class__)
        properties = {
            'instigator': 'FieldSequenceModel',
            'traits': self.traits,
            'app': str(app_name),
            'app_path': str(app_path),
            'name': self.name,
            'description': self.subtitle,
            'all_fields': self.fieldnames,
            'actions': [action.name for action in self.actions],
            'controls': [{control.name: control.value} for control in self.controls],
            'summary': self.scene_summary(),
            'time_of_writing': self.steps,
            'world': struct.properties_dict(self.world.state)
        }
        properties.update(self.custom_properties())
        self.scene.properties = properties

    def settings_str(self):
        return ''.join([
            ' ' + str(control) for control in self.controls
        ])

    def custom_properties(self):
        return self._custom_properties

    def info(self, message):
        message = str(message)
        self.message = message
        self.logger.info(message)

    def debug(self, message):
        logging.info(message)

    def scene_summary(self):
        return self.summary

    def show(self, *args, **kwargs):
        from phi.viz.dash_gui import DashFieldSequenceGui
        gui = DashFieldSequenceGui(self, *args, **kwargs)
        return gui.show()

    @property
    def status(self):
        pausing = '/Pausing' if self._pause and self.current_action else ''
        action = self.current_action if self.current_action else 'Idle'
        message = (' - %s'%self.message) if self.message else ''
        return '{}{} ({}){}'.format(action, pausing, self.steps, message)

    def run_step(self, framerate=None, allow_recording=True):
        self.current_action = 'Running'
        starttime = time.time()
        self.progress()
        if allow_recording and self.steps % self.sequence_stride == 0:
            self.record_frame()
        if framerate is not None:
            duration = time.time() - starttime
            rest = 1.0/framerate/self.sequence_stride - duration
            if rest > 0:
                self.current_action = 'Waiting'
                time.sleep(rest)
        self.current_action = None

    def play(self, max_steps=None, callback=None, framerate=None, allow_recording=True, callback_if_aborted=False):
        def target():
            self._pause = False
            step_count = 0
            while not self._pause:
                self.run_step(framerate=framerate, allow_recording=allow_recording)
                step_count += 1
                if max_steps and step_count >= max_steps:
                    break
            if callback is not None:
                if not self._pause or callback_if_aborted:
                    callback()

        thread = threading.Thread(target=target)
        thread.start()
        return self

    def pause(self):
        self._pause = True

    @property
    def running(self):
        return self.current_action is not None

    def record_frame(self):
        self.current_action = 'Recording'
        files = []

        if self.record_images:
            os.path.isdir(self.image_dir) or os.makedirs(self.image_dir)
            arrays = [self.get_field(field) for field in self.recorded_fields]
            for name, array in zip(self.recorded_fields, arrays):
                files += self.figures.save_figures(self.image_dir, name, self.steps, array)

        if self.record_data:
            arrays = [self.get_field(field) for field in self.recorded_fields]
            arrays = [a.staggered if isinstance(a, phi.math.nd.StaggeredGrid) else a for a in arrays]
            files += phi.data.fluidformat.write_sim_frame(self.directory, arrays, self.recorded_fields, self.steps)

        if files:
            self.message = 'Frame written to %s' % files
        self.current_action = None

    def benchmark(self, sequence_count):
        self._pause = False
        step_count = 0
        starttime = time.time()
        for i in range(sequence_count):
            self.run_step(framerate=None, allow_recording=False)
            step_count += 1
            if self._pause: break
        time_elapsed = time.time() - starttime
        return step_count, time_elapsed

    def config_recording(self, images, data, fields):
        self.record_images = images
        self.record_data = data
        self.recorded_fields = fields



class EditableValue(object):

    def __init__(self, name, type, initial_value, category, minmax, is_linear):
        self.name = name
        self.type = type
        self.initial_value = initial_value
        self.category = category
        self.minmax = minmax
        self.is_linear = is_linear

    @property
    def min_value(self):
        return self.minmax[0]

    @property
    def max_value(self):
        return self.minmax[1]


class EditableFloat(EditableValue):

    def __init__(self, name, initial_value, minmax=None, category=None, log_scale=None):
        if minmax is not None:
            assert len(minmax) == 2, 'minmax must be pair (min, max)'

        if log_scale is None:
            if minmax is None:
                log_scale = True
            else:
                log_scale = minmax[1] / float(minmax[0]) > 10

        if not minmax:
            if log_scale:
                magn = numpy.log10(initial_value)
                minmax = (10.0**(magn-3.2), 10.0**(magn+2.2))
            else:
                if initial_value == 0.0:
                    minmax = (-10.0, 10.0)
                elif initial_value > 0:
                    minmax = (0., 4. * initial_value)
                else:
                    minmax = (2. * initial_value, -2. * initial_value)
        else:
            minmax = (float(minmax[0]), float(minmax[1]))
        EditableValue.__init__(self, name, 'float', initial_value, category, minmax, not log_scale)

    @property
    def use_log_scale(self):
        return not self.is_linear


class EditableInt(EditableValue):

    def __init__(self, name, initial_value, minmax=None, category=None):
        if not minmax:
            if initial_value == 0:
                minmax = (-10, 10)
            elif initial_value > 0:
                minmax = (0, 4*initial_value)
            else:
                minmax = (2 * initial_value, -2 * initial_value)
        EditableValue.__init__(self, name, 'int', initial_value, category, minmax, True)


class EditableBool(EditableValue):

    def __init__(self, name, initial_value, category=None):
        EditableValue.__init__(self, name, 'bool', initial_value, category, (False, True), True)


class EditableString(EditableValue):

    def __init__(self, name, initial_value, category=None, rows=20):
        EditableValue.__init__(self, name, 'text', initial_value, category, ('', 'A'*rows), True)

    @property
    def rows(self):
        return len(self.max_value)


class Control(object):

    def __init__(self, model, attribute_name, editable_value):
        self.model = model
        self.attribute_name = attribute_name
        self.editable_value = editable_value

    @property
    def value(self):
        val = getattr(self.model, self.attribute_name)
        if isinstance(val, numpy.float32):
            return float(val)
        if isinstance(val, numpy.float64):
            return float(val)
        return val

    @value.setter
    def value(self, value):
        setattr(self.model, self.attribute_name, value)
        self.model.invalidate()

    @property
    def name(self):
        return self.editable_value.name

    @property
    def type(self):
        return self.editable_value.type

    @property
    def id(self):
        return self.attribute_name

    def __str__(self):
        return self.name + '_' + str(self.value)

    @property
    def range(self):
        return self.editable_value.minmax


class Action(object):

    def __init__(self, name, method, id):
        self.name = name
        self.method = method
        self.method_name = id

    @property
    def id(self):
        return self.method_name


def display_name(python_name):
    n = list(python_name)
    n[0] = n[0].upper()
    for i in range(1,len(n)):
        if n[i] == '_':
            n[i] = ' '
            if len(n) > i+1:
                n[i+1] = n[i+1].upper()
    return ''.join(n)
