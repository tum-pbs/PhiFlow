# coding=utf-8
from __future__ import print_function

import inspect
import logging
import numbers
import os
import sys
import threading
import time
import warnings
from os.path import isfile

import numpy as np
from phi import struct, math
from phi.field import CenteredGrid, Field, StaggeredGrid
from phi.physics._world import StateProxy, world
from .fluidformat import Scene

from .control import Action, Control
from .value import (EditableBool, EditableFloat, EditableInt, EditableString, EditableValue)


def synchronized_method(method):
    outer_lock = threading.Lock()
    lock_name = '__' + method.__name__ + '_lock' + '__'

    def sync_method(self, *args, **kws):
        with outer_lock:
            if not hasattr(self, lock_name):
                setattr(self, lock_name, threading.Lock())
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


class App(object):
    """
    Main class for defining an application that can be displayed in the GUI.

    To display data, call App.add_field().
    All fields need to be registered before the app is prepared or shown.

    To launch the GUI, call show(app). This calls App.prepare() if the app was not prepared.
    """

    def __init__(self,
                 name=None,
                 subtitle='',
                 fields=None,
                 stride=None,
                 record_images=False, record_data=False,
                 base_dir='~/phi/data/',
                 recorded_fields=None,
                 summary=None,
                 custom_properties=None,
                 target_scene=None,
                 objects_to_save=None,
                 framerate=None,
                 dt=1.0):
        self.start_time = time.time()
        self.name = name if name is not None else self.__class__.__name__
        self.subtitle = subtitle
        self.summary = summary if summary else name
        if fields:
            self.fields = {name: TimeDependentField(name, generator) for (name, generator) in fields.items()}
        else:
            self.fields = {}
        self.message = None
        self.steps = 0
        self.time = 0
        self._invalidation_counter = 0
        self._controls = []
        self._actions = []
        self._traits = []
        self.prepared = False
        self.current_action = None
        self._pause = False
        self.detect_fields = 'default'  # False, True, 'default'
        self.world = world
        self._dt = dt
        self.min_dt = self.dt
        self.dt_history = {}  # sparse representation of time when new timestep was set (for the next step)
        # Setup directory & Logging
        self.objects_to_save = [self.__class__] if objects_to_save is None else list(objects_to_save)
        self.base_dir = os.path.expanduser(base_dir)
        if not target_scene:
            self.new_scene()
            self.uses_existing_scene = False
        else:
            self.scene = target_scene
            self.uses_existing_scene = True
        if not isfile(self.scene.subpath('info.log')):
            log_file = self.log_file = self.scene.subpath('info.log')
        else:
            index = 2
            while True:
                log_file = self.scene.subpath('info_%d.log' % index)
                if not isfile(log_file):
                    break
                else:
                    index += 1
        # Setup logging
        logFormatter = logging.Formatter('%(message)s (%(levelname)s), %(asctime)sn\n')
        rootLogger = logging.getLogger()
        rootLogger.setLevel(logging.WARNING)
        customLogger = logging.Logger('app', logging.DEBUG)
        fileHandler = logging.FileHandler(log_file)
        fileHandler.setFormatter(logFormatter)
        customLogger.addHandler(fileHandler)
        consoleHandler = logging.StreamHandler(sys.stdout)
        consoleHandler.setFormatter(logFormatter)
        consoleHandler.setLevel(logging.INFO)
        customLogger.addHandler(consoleHandler)
        self.logger = customLogger
        # Framerate
        self.sequence_stride = stride if stride is not None else 1
        self.framerate = framerate if framerate is not None else stride
        self._custom_properties = custom_properties if custom_properties else {}
        # State
        self.state = None
        self.step_function = None
        # Initial log message
        self.info('App created. Scene directory is %s' % self.scene.path)

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        self._dt = value
        self.min_dt = min(self.min_dt, self.dt)
        self.dt_history[self.time] = self.dt

    def set_state(self, initial_state, step_function=None, show=(), dt=None):
        """
        Specifies the current physics state of the app and optionally the solver step function.
        The current physics state of the app is stored in `app.state`.

        This method replaces `world.add()` calls from Φ-Flow 1.

        :param initial_state: dict mapping names (str) to Fields or Tensors
        :param step_function: function to progress the state. Called as step_function(dt=dt, **current_state)
        :param show: list of names to expose to the user interface
        :param dt: (optional) value of dt to be passed to step_function
        """
        self.state = initial_state
        self.step_function = step_function
        if dt is not None:
            self.dt = dt
        for field_name in show:
            self.add_field(field_name, lambda n=field_name: self.state[n])

    @property
    def frame(self):
        return self.steps

    def new_scene(self, count=None):
        if count is None:
            count = 1 if self.world.batch_size is None else self.world.batch_size
        self.scene = Scene.create(self.base_dir, self.scene_summary(), count=count, mkdir=True)

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
        dt = self.dt  # prevent race conditions
        if self.step_function is None:
            world.step(dt=dt)
        else:
            new_state = self.step_function(dt=dt, **self.state)
            assert isinstance(self.state, dict), 'step_function must return a dict'
            assert new_state.keys() == self.state.keys(), 'step_function must return a state with the same names as the input state.\nInput: %s\nOutput: %s' % (self.state.keys(), new_state.keys())
            self.state = new_state
        self.time += dt

    @property
    def fieldnames(self):
        return sorted(self.fields.keys())

    def get_field(self, fieldname):
        if fieldname not in self.fields:
            raise KeyError('Field %s not declared. Available fields are %s' % (fieldname, self.fields.keys()))
        return self.fields[fieldname].get(self._invalidation_counter)

    def add_field(self, name, value):
        assert not self.prepared, 'Cannot add fields to a prepared model'
        if isinstance(value, StateProxy):
            def current_state():
                return value.state
            generator = current_state
        elif callable(value):
            generator = value
        else:
            assert isinstance(value, (np.ndarray, Field, float, int, math.Tensor)), 'Unsupported type for field "%s": %s' % (name, type(value))

            def get_constant():
                return value
            generator = get_constant
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
        # Controls
        for name in self.__dict__:
            val = getattr(self, name)
            editable_value = None
            if isinstance(val, EditableValue):
                editable_value = val
                setattr(self, name, val.initial_value)  # Replace EditableValue with initial value
            elif name.startswith('value_'):
                value_name = display_name(name[6:])
                dtype = type(val)
                if dtype == bool:
                    editable_value = EditableBool(value_name, val)
                elif isinstance(val, numbers.Integral):  # Int
                    editable_value = EditableInt(value_name, val)
                elif isinstance(val, numbers.Number):  # Float
                    editable_value = EditableFloat(value_name, val)
                elif isinstance(val, str):
                    editable_value = EditableString(value_name, val)
            if editable_value:
                self._controls.append(Control(self, name, editable_value))
        # Actions
        for method_name in dir(self):
            if method_name.startswith('action_') and callable(getattr(self, method_name)):
                self._actions.append(Action(display_name(method_name[7:]), getattr(self, method_name), method_name))
        # Default fields
        if len(self.fields) == 0:
            self._add_default_fields()
        # Scene
        self._update_scene_properties()
        source_files_to_save = set()
        for object in self.objects_to_save:
            try:
                source_files_to_save.add(inspect.getabsfile(object))
            except TypeError:
                pass
        for source_file in source_files_to_save:
            self.scene.copy_src(source_file)
        # End
        self.prepared = True
        return self

    def _add_default_fields(self):
        def add_default_field(trace):
            field = trace.value
            if isinstance(field, (CenteredGrid, StaggeredGrid)):
                def field_generator():
                    world_state = self.world.state
                    return trace.find_in(world_state)
                self.add_field(field.name[0].upper() + field.name[1:], field_generator)
            return None
        struct.map(add_default_field, self.world.state, leaf_condition=lambda x: isinstance(x, (CenteredGrid, StaggeredGrid)), trace=True, content_type=struct.INVALID)

    def add_custom_property(self, key, value):
        self._custom_properties[key] = value
        if self.prepared:
            self._update_scene_properties()

    def add_custom_properties(self, dictionary):
        self._custom_properties.update(dictionary)
        if self.prepared:
            self._update_scene_properties()

    def _update_scene_properties(self):
        if self.uses_existing_scene:
            return
        try:
            app_name = os.path.basename(inspect.getfile(self.__class__))
            app_path = inspect.getabsfile(self.__class__)
        except TypeError:
            app_name = app_path = ''
        properties = {
            'instigator': 'App',
            'traits': self.traits,
            'app': str(app_name),
            'app_path': str(app_path),
            'name': self.name,
            'description': self.subtitle,
            'all_fields': self.fieldnames,
            'actions': [action.name for action in self.actions],
            'controls': [{control.name: control.value} for control in self.controls],
            'summary': self.scene_summary(),
            'steps': self.steps,
            'time': self.time,
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

    def show(self, **config):
        warnings.warn("Use show(model) instead.", DeprecationWarning, stacklevel=2)
        from .display import show
        show(self, **config)

    @property
    def status(self):
        pausing = '/Pausing' if (self._pause and self.current_action) else ''
        action = self.current_action if self.current_action else 'Idle'
        message = f' - {self.message}' if self.message else ''
        return f'{action}{pausing} (t={self.format_time(self.time)} in {self.steps} steps){message}'

    def format_time(self, time):
        commas = int(np.ceil(np.abs(np.log10(self.min_dt))))
        return ("{time:," + f".{commas}f" + "}").format(time=time)

    def run_step(self, framerate=None, allow_recording=True):
        self.current_action = 'Running'
        starttime = time.time()
        try:
            self.progress()
            if framerate is not None:
                duration = time.time() - starttime
                rest = 1.0 / framerate - duration
                if rest > 0:
                    self.current_action = 'Waiting'
                    time.sleep(rest)
        except Exception as e:
            self.info('Error during %s.step() \n %s: %s' % (type(self).__name__, type(e).__name__, e))
            self.logger.exception(e)
        finally:
            self.current_action = None

    def play(self, max_steps=None, callback=None, framerate=None, allow_recording=True, callback_if_aborted=False):
        if framerate is None:
            framerate = self.framerate

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

    def benchmark(self, sequence_count):
        self._pause = False
        step_count = 0
        starttime = time.time()
        for i in range(sequence_count):
            self.run_step(framerate=np.inf, allow_recording=False)
            step_count += 1
            if self._pause:
                break
        time_elapsed = time.time() - starttime
        return step_count, time_elapsed


def display_name(python_name):
    n = list(python_name)
    n[0] = n[0].upper()
    for i in range(1, len(n)):
        if n[i] == '_':
            n[i] = ' '
            if len(n) > i + 1:
                n[i + 1] = n[i + 1].upper()
    return ''.join(n)
