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
from ._fluidformat import Scene

from ._control import Action, Control
from ._value import (EditableBool, EditableFloat, EditableInt, EditableString, EditableValue)


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
    Main class for defining an application that can be displayed in the user interface.
    
    To display data, call App.add_field().
    All fields need to be registered before the app is prepared or shown.
    
    To launch the GUI, call show(app). This calls App.prepare() if the app was not prepared.
    
    See the user interface documentation at https://tum-pbs.github.io/PhiFlow/Web_Interface.html

    Args:

    Returns:

    """

    def __init__(self,
                 name=None,
                 subtitle='',
                 fields=None,
                 stride=None,
                 base_dir='~/phi/data/',
                 summary=None,
                 custom_properties=None,
                 target_scene=None,
                 objects_to_save=None,
                 framerate=None,
                 dt=1.0):
        self.start_time = time.time()
        """ Time of creation (`App` constructor invocation) """
        self.name = name if name is not None else self.__class__.__name__
        """ Human-readable name. """
        self.subtitle = subtitle
        """ Description to be displayed. """
        self.summary = summary if summary else name
        """ The scene directory is derived from the summary. Defaults to `name`. """
        if fields:
            self.fields = {name: TimeDependentField(name, generator) for (name, generator) in fields.items()}
        else:
            self.fields = {}
        self.message = None
        self.steps = 0
        """ Counts the number of times `step()` has been called. May be set by the user. """
        self.time = 0
        """ Time variable for simulations. Can be set by the user. """
        self._invalidation_counter = 0
        self._controls = []
        self._actions = []
        self._traits = []
        self.prepared = False
        """ Wheter `prepare()` has been called. """
        self.current_action = None
        self._pause = False
        self.detect_fields = 'default'  # False, True, 'default'
        self.world = world
        self._dt = dt.initial_value if isinstance(dt, EditableValue) else dt
        if isinstance(dt, EditableValue):
            self._controls.append(Control(self, "dt", dt))
        self.min_dt = self._dt
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
        """ Target frame rate in Hz. Play will not step faster than the framerate. `None` for unlimited frame rate. """
        self._custom_properties = custom_properties if custom_properties else {}
        # State
        self.state = None
        self.step_function = None
        # Initial log message
        self.info('App created. Scene directory is %s' % self.scene.path)

    @property
    def dt(self):
        """
        Current time increment per step.
        Used for `step_function` set by `set_state()` or for `world.step()` in legacy-style simulations. """
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

        Args:
          initial_state: dict mapping names (str) to Fields or Tensors
          step_function: function to progress the state. Called as step_function(dt=dt, **current_state) (Default value = None)
          show: list of names to expose to the user interface (Default value = ())
          dt: optional) value of dt to be passed to step_function (Default value = None)

        Returns:

        """
        self.state = initial_state
        self.step_function = step_function
        if dt is not None:
            self.dt = dt
        if show:
            if not self.prepared:
                for field_name in show:
                    self.add_field(field_name, lambda n=field_name: self.state[n])
                else:
                    warnings.warn('Ignoring show argument because App is already prepared.')

    @property
    def frame(self):
        """ Alias for `steps`. """
        return self.steps

    def new_scene(self, count=None):
        if count is None:
            count = 1 if self.world.batch_size is None else self.world.batch_size
        self.scene = Scene.create(self.base_dir, self.scene_summary(), count=count, mkdir=True)

    @property
    def directory(self):
        """ This directory is automatically created upon `App` creation. Equal to `scene.path`. """
        return self.scene.path

    def _progress(self):
        # actual method called to step.
        self.step()
        self.steps += 1
        self.invalidate()

    def invalidate(self):
        """ Causes the user interface to update. """
        self._invalidation_counter += 1

    def step(self):
        """
        Performs a single step.
        You may override this method to specify what happens when the user presses the buttons `Step` or `Play`.
        
        If a step function has been passed to `App.set_state()`, the state is progressed using that function.
        
        Otherwise, `world.step()` is executed (for phiflow 1 style simulations).
        
        App.steps automatically counts how many steps have been completed.
        If this method is not overridden, `App.time` is additionally increased by `App.dt`.

        Args:

        Returns:

        """
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
        """ Alphabetical list of field names. See `get_field()`. """
        return sorted(self.fields.keys())

    def get_field(self, fieldname):
        """
        Reads the current value of a field.
        Fields can be added using `add_field()`.

        If a generator function was registered as the field data, this method may invoke the function which may take some time to complete.
        """
        if fieldname not in self.fields:
            raise KeyError('Field %s not declared. Available fields are %s' % (fieldname, self.fields.keys()))
        return self.fields[fieldname].get(self._invalidation_counter)

    def add_field(self, name: str, value):
        """
        Expose data to be displayed in the user interface.
        This method must be called before the user interface is launched, i.e. before `show(app)` or `app.prepare()` are invoked.
        
        `value` must be one of the following
        
        * Field
        * tensor
        * function without arguments returning one of the former. This function will be called each time the interface is updated.

        Args:
          name: unique human-readable name
          value: data to display
          name: str: 

        Returns:

        """
        assert not self.prepared, 'Cannot add fields to a prepared model'
        if isinstance(value, StateProxy):
            def current_state():
                return value.state
            generator = current_state
        elif callable(value):
            generator = value
        else:
            assert isinstance(value, (np.ndarray, Field, float, int, list, math.Tensor)), 'Unsupported type for field "%s": %s' % (name, type(value))

            def get_constant():
                return value
            generator = get_constant
        self.fields[name] = TimeDependentField(name, generator)

    @property
    def actions(self):
        """
        List of all custom actions that can be invoked at runtime by the user.
        Actions can be registered using `add_action()` or by defining a method with prefix `action_`.
        """
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
        """
        Prepares the app to be displayed in a user interface.
        
        This method can only be called once.
        If not invoked manually, it is automatically called before the user interface is launched.
        
        Preparation includes:
        
        * Detecting editable values from member variables that start with 'value_'
        * Detecting actions from member functions that start with 'action_'
        * Initializing the scene directory with a JSON file and copying related Python source files
        
        :return: self

        Args:

        Returns:

        """
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

    def info(self, message: str):
        """
        Update the status message.
        The status message is written to the console and the log file.
        Additionally, it may be displayed by the user interface.

        See `debug()`.

        Args:
            message: Message to display
        """
        message = str(message)
        self.message = message
        self.logger.info(message)

    def debug(self, message):
        """
        Prints a message to the log file but does not display it.

        See `info()`.

        Args:
            message: Message to log.
        """
        logging.info(message)

    def scene_summary(self):
        return self.summary

    def show(self, **config):
        warnings.warn("Use show(model) instead.", DeprecationWarning, stacklevel=2)
        from ._display import show
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

    def run_step(self, framerate=None):
        self.current_action = 'Running'
        starttime = time.time()
        try:
            self._progress()
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

    def play(self, max_steps=None, callback=None, framerate=None, callback_if_aborted=False):
        """
        Run a number of steps.

        Args:
            max_steps: (optional) stop when this many steps have been completed (independent of the `steps` variable) or `pause()` is called.
            callback: Function to be run after all steps have been completed.
            framerate: Target frame rate in Hz.
            callback_if_aborted: Whether to invoke `callback` if `pause()` causes this method to abort prematurely.

        Returns:
            self
        """
        if framerate is None:
            framerate = self.framerate

        def target():
            self._pause = False
            step_count = 0
            while not self._pause:
                self.run_step(framerate=framerate)
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
        """ Causes the `play()` method to stop after finishing the current step. """
        self._pause = True

    @property
    def running(self):
        """ Whether `play()` is currently executing. """
        return self.current_action is not None

    def benchmark(self, sequence_count):
        self._pause = False
        step_count = 0
        starttime = time.time()
        for i in range(sequence_count):
            self.run_step(framerate=np.inf)
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
