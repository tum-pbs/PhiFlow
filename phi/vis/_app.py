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

from ._log import SceneLog
from ._vis_base import VisModel


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


class App(VisModel):

    def __init__(self,
                 name: str = None,
                 subtitle: str = "",
                 scene: Scene = None):
        """
        Main class for defining an application that can be displayed in the user interface.

        To display data, call App.add_field().
        All fields need to be registered before the vis is prepared or shown.

        To launch the GUI, call show(vis). This calls App.prepare() if the vis was not prepared.

        See the user interface documentation at https://tum-pbs.github.io/PhiFlow/Visualization.html

        Args:
            name: Display name of the vis.
            subtitle: Description.
            scene: (Optional) Directory to which data is stored.
            log_performance: Whether to log the time elapsed during each step as a scalar value.
                The values will be written to the vis's directory and shown in the user interface.
        """
        VisModel.__init__(self, name, subtitle, scene)
        self.current_action = None
        self._pause = False
        self._elapsed = None
        self.time = 0
        """ Time variable for simulations. Can be set by the user. """
        self.prepared = False
        """ Wheter `prepare()` has been called. """
        self._controls = []
        self._actions = []
        self._invalidation_counter = 0
        self._fields = {}
        self._log = SceneLog(scene)
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

    def progress(self):
        with self.progress_lock:
            self._pre_step()
            try:
                self.step()
            except Exception as e:
                self.info(
                    "Error during %s.step() \n %s: %s"
                    % (type(self).__name__, type(e).__name__, e)
                )
                self.logger.exception(e)
            self._post_step()

    def _pre_step(self):
        for obs in self.pre_step:
            obs(self)
        self._step_start_time = time.perf_counter()

    def _post_step(self, notify_observers=True):
        self._elapsed = time.perf_counter() - self._step_start_time
        if self.log_performance:
            self.log_scalar('step_time', self._elapsed)
        self.steps += 1
        self.invalidate()
        if notify_observers:
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
        This method must be called before the user interface is launched, i.e. before `show(vis)` or `vis.prepare()` are invoked.
        
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
        Prepares the vis to be displayed in a user interface.
        
        This method can only be called once.
        If not invoked manually, it is automatically called before the user interface is launched.
        
        Preparation includes:
        
        * Detecting editable values from member variables that start with 'value_'
        * Detecting actions from member functions that start with 'action_'
        * Initializing the scene directory with a JSON file and copying related Python source files

        Returns:
            `vis`
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
                )  # Replace Control with initial value
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
            "vis": str(app_name),
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
