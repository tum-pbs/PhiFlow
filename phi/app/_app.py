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
from phi.field import Field, Scene
from phi.physics._world import StateProxy, world

from ._control import Action, Control
from ._value import EditableBool, EditableFloat, EditableInt, EditableString, EditableValue


def synchronized_method(method):
    outer_lock = threading.Lock()
    lock_name = "__" + method.__name__ + "_lock" + "__"

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
    """

    def __init__(self,
                 name: str = None,
                 subtitle: str = "",
                 scene: Scene = None,
                 log_performance=True):
        self.start_time = time.time()
        """ Time of creation (`App` constructor invocation) """
        self.name = name if name is not None else self.__class__.__name__
        """ Human-readable name. """
        self.subtitle = subtitle
        """ Description to be displayed. """
        self.scene = scene
        """ Directory to which data and logging information should be written as `Scene` instance. """
        self.uses_existing_scene = scene.exist_properties() if scene is not None else False
        self._field_names = []
        self._fields = {}
        self.message = None
        self.steps = 0
        """ Counts the number of times `step()` has been called. May be set by the user. """
        self.time = 0
        """ Time variable for simulations. Can be set by the user. """
        self._invalidation_counter = 0
        self._controls = []
        self._actions = []
        self.prepared = False
        """ Wheter `prepare()` has been called. """
        self.current_action = None
        self._pause = False
        self.pre_step = []  # callback(app)
        self.post_step = []  # callback(app)
        self.world = world
        self.log_performance = log_performance
        self._elapsed = None
        # Message logging
        log_formatter = logging.Formatter("%(message)s (%(levelname)s), %(asctime)sn\n")
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.WARNING)
        self.logger = logging.Logger("app", logging.DEBUG)
        console_handler = self.console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_formatter)
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)
        if self.scene is not None:
            if not isfile(self.scene.subpath("info.log")):
                log_file = self.log_file = self.scene.subpath("info.log")
            else:
                index = 2
                while True:
                    log_file = self.scene.subpath("info_%d.log" % index)
                    if not isfile(log_file):
                        break
                    else:
                        index += 1
            file_handler = self.file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(log_formatter)
            self.logger.addHandler(file_handler)
        # Data logging
        self._scalars = {}  # name -> (frame, value)
        self._scalar_streams = {}
        # Initial log message
        if self.scene is not None:
            self.info("App created. Scene directory is %s" % self.scene.path)

    @property
    def frame(self):
        """ Alias for `steps`. """
        return self.steps

    @property
    def directory(self):
        """ This directory is automatically created upon `App` creation. Equal to `scene.path`. """
        return self.scene.path

    def _progress(self):
        self._pre_step()
        self.step()
        self._post_step()

    def _pre_step(self):
        for obs in self.pre_step:
            obs(self)
        self._step_start_time = time.perf_counter()

    def _post_step(self):
        self._elapsed = time.perf_counter() - self._step_start_time
        if self.log_performance:
            self.log_scalar('step_time', self._elapsed)
        self.steps += 1
        self.invalidate()
        for obs in self.post_step:
            obs(self)

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
        """
        raise NotImplementedError("step() must be overridden.")

    @property
    def fieldnames(self):
        """ Alphabetical list of field names. See `get_field()`. """
        return self._field_names

    def get_field(self, name):
        """
        Reads the current value of a field.
        Fields can be added using `add_field()`.

        If a generator function was registered as the field data, this method may invoke the function which may take some time to complete.
        """
        if name not in self._fields:
            raise KeyError(f"Field {name} not declared. Available fields are {self._fields.keys()}")
        return self._fields[name].get(self._invalidation_counter)

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
        """
        assert not self.prepared, "Cannot add fields to a prepared model"
        if isinstance(value, StateProxy):

            def current_state():
                return value.state

            generator = current_state
        elif callable(value):
            generator = value
        else:
            assert isinstance(
                value, (np.ndarray, Field, float, int, math.Tensor)
            ), 'Unsupported type for field "%s": %s' % (name, type(value))

            def get_constant():
                return value

            generator = get_constant
        self._field_names.append(name)
        self._fields[name] = TimeDependentField(name, generator)

    def log_scalar(self, name: str, value: float or math.Tensor):
        """
        Adds `value` to the curve `name` at the current step.
        This can be used to log the evolution of scalar quantities or summaries.

        The values are stored in a text file within the scene directory.
        The curves may also be directly viewed in the user interface.

        Args:
            name: Name of the curve. If no such curve exists, a new one is created.
            value: Value to append to the curve, must be a number or `phi.math.Tensor`.
        """
        assert isinstance(name, str)
        value = float(math.mean(value))
        if name not in self._scalars:
            self._scalars[name] = []
            if self.scene is not None:
                path = self.scene.subpath(f"log_{name}.txt")
                self._scalar_streams[name] = open(path, "w")
        self._scalars[name].append((self.frame, value))
        if self.scene is not None:
            self._scalar_streams[name].write(f"{value}\n")
            self._scalar_streams[name].flush()

    def log_scalars(self, **values: float or math.Tensor):
        for name, value in values.items():
            self.log_scalar(name, value)

    def get_scalar_curve(self, name) -> tuple:
        frames = np.array([item[0] for item in self._scalars[name]])
        values = np.array([item[1] for item in self._scalars[name]])
        return frames, values

    def get_logged_scalars(self):
        return self._scalars.keys()

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
            if self.message is None or self.message == "":
                self.message = display_name(action.name)
            else:
                self.message += " | " + display_name(action.name)

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

        Returns:
            `app`
        """
        if self.prepared:
            return
        logging.info("Gathering model data...")
        # Controls
        for name in self.__dict__:
            val = getattr(self, name)
            editable_value = None
            if isinstance(val, EditableValue):
                editable_value = val
                setattr(
                    self, name, val.initial_value
                )  # Replace EditableValue with initial value
            elif name.startswith("value_"):
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
            if method_name.startswith("action_") and callable(
                getattr(self, method_name)
            ):
                self._actions.append(
                    Action(
                        display_name(method_name[7:]),
                        getattr(self, method_name),
                        method_name,
                    )
                )
        # Scene
        if self.scene is not None:
            self._update_scene_properties()
            source_files_to_save = set()
            for object in [self.__class__]:
                try:
                    source_files_to_save.add(inspect.getabsfile(object))
                except TypeError:
                    pass
            for source_file in source_files_to_save:
                self.scene.copy_src(source_file)
        # End
        self.prepared = True
        return self

    def _update_scene_properties(self):
        if self.uses_existing_scene or self.scene is None:
            return
        try:
            app_name = os.path.basename(inspect.getfile(self.__class__))
            app_path = inspect.getabsfile(self.__class__)
        except TypeError:
            app_name = app_path = ""
        properties = {
            "instigator": "App",
            "app": str(app_name),
            "app_path": str(app_path),
            "name": self.name,
            "description": self.subtitle,
            "all_fields": self.fieldnames,
            "actions": [action.name for action in self.actions],
            "controls": [{control.name: control.value} for control in self.controls],
            "steps": self.steps,
            "time": self.time,
            "world": struct.properties_dict(self.world.state),
        }
        self.scene.properties = properties

    def settings_str(self):
        return "".join([" " + str(control) for control in self.controls])

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

    def show(self, **config):
        warnings.warn("Use show(model) instead.", DeprecationWarning, stacklevel=2)
        from ._display import show

        show(self, **config)

    @property
    def status(self):
        pausing = "/Pausing" if (self._pause and self.current_action) else ""
        action = self.current_action if self.current_action else "Idle"
        message = f" - {self.message}" if self.message else ""
        return f"{action}{pausing} ({self.steps} steps){message}"

    def run_step(self, framerate=None):
        self.current_action = "Running"
        starttime = time.time()
        try:
            self._progress()
            if framerate is not None:
                duration = time.time() - starttime
                rest = 1.0 / framerate - duration
                if rest > 0:
                    self.current_action = "Waiting"
                    time.sleep(rest)
        except Exception as e:
            self.info(
                "Error during %s.step() \n %s: %s"
                % (type(self).__name__, type(e).__name__, e)
            )
            self.logger.exception(e)
        finally:
            self.current_action = None

    def pause(self):
        """ Causes the `play()` method to stop after finishing the current step. """
        self._pause = True

    def is_paused(self):
        return self._pause

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
        if n[i] == "_":
            n[i] = " "
            if len(n) > i + 1:
                n[i + 1] = n[i + 1].upper()
    return "".join(n)


def play_async(app: App, max_steps=None, callback=None, framerate=None, callback_if_aborted=False):
    """
    Run a number of steps.

    Args:
        app: `App`
        max_steps: (optional) stop when this many steps have been completed (independent of the `steps` variable) or `pause()` is called.
        callback: Function to be run after all steps have been completed.
        framerate: Target frame rate in Hz.
        callback_if_aborted: Whether to invoke `callback` if `pause()` causes this method to abort prematurely.
    """
    def target():
        app._pause = False
        step_count = 0
        while not app.is_paused():
            app.run_step(framerate=framerate)
            step_count += 1
            if max_steps and step_count >= max_steps:
                break
        if callback is not None:
            if not app.is_paused() or callback_if_aborted:
                callback()

    thread = threading.Thread(target=target)
    thread.start()
