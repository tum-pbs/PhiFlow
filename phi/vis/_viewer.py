import itertools
import warnings
from threading import Event

from . import EditableValue
from ._app import App
from ._user_namespace import UserNamespace
from .. import field
from ..field import Scene


def create_viewer(namespace: UserNamespace,
                  fields: dict,
                  name: str = None,
                  subtitle: str = "",
                  scene: Scene = None,
                  asynchronous: bool = False,
                  controls: tuple or list = None,
                  log_performance: bool = True) -> 'Viewer':
    # controls: `dict` mapping valid Python names to initial values or `EditableValue` instances.
    # The display names for the controls will be generated based on the python names.
    cls = AsyncViewer if asynchronous else Viewer
    viewer = cls(namespace, fields, name, subtitle, scene, log_performance)
    if controls:
        for name, value in controls.items():
            if isinstance(value, EditableValue):
                setattr(viewer, name, value)
            else:
                setattr(viewer, f'value_{name}', value)
    return viewer


class Viewer(App):
    """
    Launches the user interface to display the contents of the calling Python script or notebook.

    Name and subtitle of the App may be specified in the module docstring (string before imports).
    The first line is interpreted as the name, the rest as the subtitle.
    If not specified, a generic name and description is chosen.

    Use ModuleViewer.range() as a for-loop iteratable to control the loop execution from within the GUI.

    Also see the user interface documentation at https://tum-pbs.github.io/PhiFlow/Visualization.html
    """

    def __init__(self,
                 namespace: UserNamespace,
                 fields: dict,
                 name: str,
                 subtitle: str,
                 scene: Scene,
                 log_performance: bool,
                 ):
        App.__init__(self, name, subtitle, scene=scene, log_performance=log_performance)
        self.initial_field_values = fields
        self.namespace = namespace
        self.on_loop_start = []
        self.on_loop_exit = []
        for name in fields.keys():

            def get_field(n=name):
                if self.rec:
                    return self.rec[n]
                else:
                    return self.namespace.get_variable(n)

            self.add_field(name, get_field)
        self.add_action("Reset", lambda: self.restore_initial_field_values())
        self.rec = None

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
            *args: No arguments for infinite loop,
                `(stop: int)` to set number of iterations,
                `(start: int, stop: int)` to additionally set initial value of `step`.
            warmup: Number of uncounted loop iterations to perform before `step()` is invoked for the first time.
            **rec_dim: Can be used instead of `*args` to record values along this batch dimension.
                The recorded values can be accessed as `Viewer.rec.<name>` or `Viewer.rec['<name>']`.

        Returns:
            generator yielding `ModuleViewer.step`
        """
        for _ in range(warmup):
            yield self.steps
            self.invalidate()

        for obs in self.on_loop_start:
            obs(self)

        if rec_dim:
            assert len(rec_dim) == 1, f"Only one rec_dim allowed but got {rec_dim}"
            assert not args, f"No positional arguments are allowed when a rec_dim is specified. {rec_dim}"
            rec_dim_name = next(iter(rec_dim.keys()))
            size = rec_dim[rec_dim_name]
            assert isinstance(size, int)
            self.rec = Record(rec_dim_name)
            self.rec.append(self.initial_field_values, warn_missing=False)
            args = [size]
            self.growing_dims = [rec_dim_name]

        if len(args) == 0:
            step_source = itertools.count(start=1)
        elif len(args) == 1:
            step_source = range(args[0])
        elif len(args) == 2:
            step_source = range(args[0], args[1])
        else:
            raise ValueError(args)

        try:
            for step in step_source:
                self.steps = step
                self._pre_step()
                yield step
                self._post_step(notify_observers=False)
                if rec_dim:
                    self.rec.append({name: self.namespace.get_variable(name) for name in self.fieldnames})
                for obs in self.post_step:
                    obs(self)
        finally:
            for obs in self.on_loop_exit:
                obs(self)

    def step(self):
        """ Has no effect. The real step is a loop iteration. See `Viewer.range()`. """
        pass

    def restore_initial_field_values(self, reset_steps=True):
        for name, value in self.initial_field_values.items():
            self.namespace.set_variable(name, value)
        if reset_steps:
            self.steps = 0


class AsyncViewer(Viewer):

    def __init__(self, *args):
        Viewer.__init__(self, *args)
        self.step_exec_event = Event()
        self.step_finished_event = Event()
        self._interrupt = False

    def _pre_step(self):
        self.step_exec_event.wait()
        if self._interrupt:
            raise InterruptedError()
        App._pre_step(self)

    def _post_step(self, notify_observers=True):
        App._post_step(self)
        if self._interrupt:
            raise InterruptedError()
        self.step_exec_event.clear()
        self.step_finished_event.set()

    def _progress(self):  # called by the GUI
        """
        Allows the generator returned by `ModuleViewer.range()` to advance one element.
        In typical scenarios, this will run one loop iteration in the top-level script.
        """
        self.step_finished_event.clear()
        self.step_exec_event.set()
        self.step_finished_event.wait()

    def interrupt(self):
        self._interrupt = True


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
                warnings.warn(f"None value encountered for variable '{name}' at step {self.viewer.steps}. This value will not show up in the recording.")

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
            return field.batch_stack(*snapshots, dim=self.dim)
        else:
            return None

    def __getitem__(self, item):
        assert isinstance(item, str)
        return self.__getattr__(item)
