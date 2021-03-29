import time
from contextlib import contextmanager
from threading import Event

from ._app import App
from . import EditableValue
from ._user_namespace import UserNamespace
from ..field import Scene


def create_viewer(namespace: UserNamespace,
                  fields: dict,
                  name: str = None,
                  subtitle: str = "",
                  scene: Scene = None,
                  asynchronous: bool = False,
                  controls: tuple or list = None,
                  log_performance: bool = True
                  ):
    cls = AsyncViewer if asynchronous else SyncViewer
    viewer = cls(namespace, fields, name, subtitle, scene, log_performance)
    if controls:
        for name, value in controls.items():
            if isinstance(value, EditableValue):
                setattr(viewer, name, value)
            else:
                setattr(viewer, f'value_{name}', value)
    return viewer


class Viewer(App):

    def __init__(self,
                 namespace: UserNamespace,
                 fields: dict,
                 name: str,
                 subtitle: str,
                 scene: Scene,
                 log_performance: bool):
        App.__init__(self, name, subtitle, scene=scene, log_performance=log_performance)
        self._initial_field_values = fields
        self.namespace = namespace
        self.on_loop_start = []
        self.on_loop_exit = []
        for name, value in fields.items():
            self.add_field(name, lambda n=name: self.namespace.get_variable(n))
        self.add_action("Reset", lambda: self.restore_initial_field_values())

    def range(self, *args, warmup=0):
        raise NotImplementedError(self)

    def restore_initial_field_values(self, reset_steps=True):
        for name, value in self._initial_field_values.items():
            self.namespace.set_variable(name, value)
        if reset_steps:
            self.steps = 0


class SyncViewer(Viewer):

    def step(self):
        pass

    def range(self, *args, warmup=0):
        for _ in range(warmup):
            yield self.steps
            self.invalidate()

        for obs in self.on_loop_start:
            obs(self)

        try:
            if len(args) == 0:
                while True:
                    self._pre_step()
                    yield self.steps
                    self._post_step()
            elif len(args) == 1:
                for _ in range(args[0]):
                    self._pre_step()
                    yield self.steps
                    self._post_step()
            elif len(args) == 2:
                for i in range(args[0], args[1]):
                    self.steps = i
                    self._pre_step()
                    yield self.steps
                    self._post_step()
            else:
                raise ValueError(args)
        finally:
            for obs in self.on_loop_exit:
                obs(self)


class AsyncViewer(Viewer):
    """
    ModuleViewer launches the user interface to display the contents of the calling Python script.

    Name and subtitle of the App may be specified in the module docstring (string before imports).
    The first line is interpreted as the name, the rest as the subtitle.
    If not specified, a generic name and description is chosen.

    Use ModuleViewer.range() as a for-loop iteratable to control the loop execution from within the GUI.

    Also see the user interface documentation at https://tum-pbs.github.io/PhiFlow/Web_Interface.html
    """

    def __init__(self, *args):
        """
        Creates the ModuleViewer `App` and `show()`s it.

        Args:
            fields: (Optional) names of global variables to be displayed.
                If not provided, searches all global variables for Field or Tensor values.
                All fields must exist as global variables before the ModuleViewer is instantiated.
            log_performance: Whether to log the time elapsed during each step as a scalar value.
                The values will be written to the app's directory and shown in the user interface.
            controls: `dict` mapping valid Python names to initial values or `EditableValue` instances.
                The display names for the controls will be generated based on the python names.
        """
        Viewer.__init__(self, *args)
        self.step_exec_event = Event()
        self.step_finished_event = Event()
        self._interrupt = False

    def range(self, *args, warmup=0):
        """
        Similar to `range()`, returns a generator that can be used in a `for` loop.

            for step in ModuleViewer().range(100):
                print(f'Running step {step}')

        However, `ModuleViewer.range()` controlling the flow via the user interface.
        Each element returned by the generator waits for `ModuleViewer.step()` to be invoked once.

        Note that `step` is always equal to `ModuleViewer.step`.

        This method can be invoked multiple times.
        However, do not call this method while one `range` is still active.

        Args:
            *args: No arguments for infinite loop,
                `(stop: int)` to set number of iterations,
                `(start: int, stop: int)` to additionally set initial value of `step`.
            warmup: Number of uncounted loop iterations to perform before `step()` is invoked for the first time.

        Returns:
            generator yielding `ModuleViewer.step`
        """
        for _ in range(warmup):
            yield self.steps
            self.invalidate()

        if len(args) == 0:
            while True:
                with self._perform_step_context():
                    yield self.steps
        elif len(args) == 1:
            for _ in range(args[0]):
                with self._perform_step_context():
                    yield self.steps
        elif len(args) == 2:
            for i in range(args[0], args[1]):
                with self._perform_step_context():
                    self.steps = i
                    yield self.steps
        else:
            raise ValueError(args)

    @contextmanager
    def _perform_step_context(self):
        self.step_exec_event.wait()
        if self._interrupt:
            raise InterruptedError()
        try:
            yield None
        finally:
            if self._interrupt:
                raise InterruptedError()
            self.step_exec_event.clear()
            self.step_finished_event.set()

    def step(self):
        """
        Allows the generator returned by `ModuleViewer.range()` to advance one element.
        In typical scenarios, this will run one loop iteration in the top-level script.
        """
        self.step_finished_event.clear()
        self.step_exec_event.set()
        self.step_finished_event.wait()

    def interrupt(self):
        self._interrupt = True
