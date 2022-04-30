import itertools
import sys
import time
import warnings
from functools import partial
from threading import Event
from typing import Tuple

from ._log import SceneLog
from ._user_namespace import UserNamespace
from ._vis_base import VisModel, Control, Action
from .. import field
from ..field import Scene, SampledField
from ..math import batch, Tensor
from ..math.backend import PHI_LOGGER


def create_viewer(namespace: UserNamespace,
                  fields: dict,
                  name: str,
                  description: str,
                  scene: Scene or None,
                  asynchronous: bool,
                  controls: tuple,
                  actions: dict,
                  log_performance: bool) -> 'Viewer':
    cls = AsyncViewer if asynchronous else Viewer
    viewer = cls(namespace, fields, name, description, scene, controls, actions, log_performance)
    return viewer


class Viewer(VisModel):
    """
    Shows variables from the user namespace.
    To create a `Viewer`, call `phi.vis.view()` from the top-level Python script or from a notebook.

    Use `Viewer.range()` to control the loop execution from the user interface.

    Also see the user interface documentation at https://tum-pbs.github.io/PhiFlow/Visualization.html
    """

    def __init__(self,
                 namespace: UserNamespace,
                 fields: dict,
                 name: str,
                 description: str,
                 scene: Scene,
                 controls: tuple,
                 actions: dict,
                 log_performance: bool,
                 ):
        VisModel.__init__(self, name, description, scene=scene)
        self.initial_field_values = fields
        self._controls = controls
        self.namespace = namespace
        self.log_performance = log_performance
        self._rec = None
        self._in_loop = False
        self._log = SceneLog(self.scene)
        self.log_file = self._log.log_file
        self._elapsed = None
        self.reset_step = 0
        self._actions = {}
        custom_reset = False
        self.reset_count = 0
        for action, function in actions.items():
            if action.name == 'reset':
                self._actions[action] = partial(self.reset, custom_reset=function)
                custom_reset = True
            else:
                self._actions[action] = function
        if not custom_reset:
            self._actions[Action('reset', Viewer.reset.__doc__)] = self.reset

    def log_scalars(self, **values):
        self._log.log_scalars(self.steps, **values)

    def info(self, message: str):  # may be replaced by a different solution later on
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
        self._log.log(message)

    def __rrshift__(self, other):
        self.info(other)

    @property
    def field_names(self) -> tuple:
        return tuple(self.initial_field_values.keys())

    def get_field(self, name, dim_selection: dict) -> SampledField:
        if name not in self.initial_field_values:
            raise KeyError(name)
        if self._rec:
            value = self._rec[name]
        else:
            value = self.namespace.get_variable(name)
        if callable(value):
            value = value()
        if isinstance(value, (SampledField, Tensor)):
            value = value[dim_selection]
        return value

    @property
    def curve_names(self) -> tuple:
        return self._log.scalar_curve_names

    def get_curve(self, name: str) -> tuple:
        return self._log.get_scalar_curve(name)

    @property
    def controls(self) -> Tuple[Control]:
        return self._controls

    def get_control_value(self, name):
        return self.namespace.get_variable(name)

    def set_control_value(self, name, value):
        self.namespace.set_variable(name, value)

    @property
    def actions(self) -> tuple:
        return tuple(self._actions.keys())

    def run_action(self, name):
        for action, fun in self._actions.items():
            if action.name == name:
                fun()
                return
        raise KeyError(name)

    def range(self, *args, warmup=0, **rec_dim):
        """
        Similarly to `range()`, returns a generator that can be used in a `for` loop.

        ```python
        for step in ModuleViewer().range(100):
            print(f'Running step {step}')
        ```

        However, `Viewer.range()` enables controlling the flow via the user interface.
        Each element returned by the generator waits for `progress` to be invoked once.

        Note that `step` is always equal to `Viewer.steps`.

        This method can be invoked multiple times.
        However, do not call this method while one `range` is still active.

        Args:
            *args: Either no arguments for infinite loop or single `int` argument `stop`.
                Must be empty if `rec_dim` is used.
            **rec_dim: Can be used instead of `*args` to record values along a new batch dimension of this name.
                The recorded values can be accessed as `Viewer.rec.<name>` or `Viewer.rec['<name>']`.
            warmup: Number of uncounted loop iterations to perform before `step()` is invoked for the first time.

        Yields:
            Step count of `Viewer`.
        """
        for _ in range(warmup):
            yield self.steps

        self._in_loop = True
        self._call(self.progress_available)

        if rec_dim:
            assert len(rec_dim) == 1, f"Only one rec_dim allowed but got {rec_dim}"
            assert not args, f"No positional arguments are allowed when a rec_dim is specified. {rec_dim}"
            rec_dim_name = next(iter(rec_dim.keys()))
            size = rec_dim[rec_dim_name]
            assert isinstance(size, int)
            self._rec = Record(rec_dim_name)
            self._rec.append(self.initial_field_values, warn_missing=False)
            args = [size]
            self.growing_dims = [rec_dim_name]

        if len(args) == 0:
            def count():
                i = 0
                while True:
                    yield i
                    i += 1

            step_source = count()
        else:
            step_source = range(*args)

        try:
            for step in step_source:
                self.steps = step - self.reset_step
                try:
                    self._pre_step()
                    t = time.perf_counter()
                    yield step - self.reset_step
                    self._elapsed = time.perf_counter() - t
                    self.steps = step - self.reset_step + 1
                    if rec_dim:
                        self._rec.append({name: self.namespace.get_variable(name) for name in self.field_names})
                    if self.log_performance:
                        self._log.log_scalars(self.steps, step_time=self._elapsed)
                finally:
                    self._post_step()
        finally:
            self._in_loop = False
            self._call(self.progress_unavailable)

    def _pre_step(self):
        self._call(self.pre_step)

    def _post_step(self):
        self._call(self.post_step)

    @property
    def rec(self) -> 'Record':
        """
        Read recorded fields as `viewer.rec.<name>`.
        Accessing `rec` without having started a recording using `Viewer.range()` raises an `AssertionError`.
        """
        assert self._rec, "Enable recording by calling range() with a dimension name, e.g. 'range(frames=10)'."
        return self._rec

    def progress(self):
        raise AssertionError("progress() not supported by synchronous Viewer.")

    @property
    def can_progress(self) -> bool:
        return self._in_loop

    def reset(self, custom_reset=None):
        """
        Restores all viewed fields to the states they were in when the viewer was created.
        Changes variable values in the user namespace.
        """
        if custom_reset:
            custom_reset()
        for name, value in self.initial_field_values.items():
            self.namespace.set_variable(name, value)
        self.reset_step += self.steps
        self.steps = 0
        self.reset_count += 1


class AsyncViewer(Viewer):

    def __init__(self, *args):
        Viewer.__init__(self, *args)
        self.step_exec_event = Event()
        self.step_finished_event = Event()

    def _pre_step(self):
        self.step_exec_event.wait()
        self._call(self.pre_step)

    def _post_step(self):
        self._call(self.post_step)
        self.step_exec_event.clear()
        self.step_finished_event.set()

    def progress(self):  # called by the GUI
        """
        Allows the generator returned by `ModuleViewer.range()` to advance one element.
        In typical scenarios, this will run one loop iteration in the top-level script.
        """
        self.step_finished_event.clear()
        self.step_exec_event.set()
        self.step_finished_event.wait()

    def can_progress(self) -> bool:
        return True


class Record:

    def __init__(self, dim: str or None):
        self.dim = dim
        self.history = {}

    def append(self, variables: dict, warn_missing=True):
        if not self.history:
            self.history = {name: [] for name in variables.keys()}
        for name, val in variables.items():
            self.history[name].append(val)
            if val is None and warn_missing:
                warnings.warn(f"None value encountered for variable '{name}' at step {self.viewer.steps}. This value will not show up in the recording.", RuntimeWarning)

    @property
    def recorded_fields(self):
        return tuple(self.history.keys())

    def get_snapshot(self, name: str, frame: int):
        return self.history[name][frame]

    def recording_size(self, name: str):
        return len(self.history[name])

    def __getattr__(self, item: str):
        assert item in self.history, f"No recording available for '{item}'. The following fields were recorded: {self.recorded_fields}"
        snapshots = [v for v in self.history[item] if v is not None]
        if snapshots:
            return field.stack(snapshots, batch(self.dim))
        else:
            return None

    def __getitem__(self, item):
        assert isinstance(item, str)
        return self.__getattr__(item)

    def __repr__(self):
        return ", ".join([f"{name} ({len(values)})" for name, values in self.history.items()])
