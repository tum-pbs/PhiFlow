import inspect
import os
import time
from contextlib import contextmanager
from threading import Thread, Event

from phi.field import Field
from phi.math import Tensor

from ._app import App
from ._display import show
from . import _display, EditableValue


class ModuleViewer(App):
    """
    ModuleViewer launches the user interface to display the contents of the calling Python script.

    Name and subtitle of the App may be specified in the module docstring (string before imports).
    The first line is interpreted as the name, the rest as the subtitle.
    If not specified, a generic name and description is chosen.

    Use ModuleViewer.range() as a for-loop iteratable to control the loop execution from within the GUI.

    Also see the user interface documentation at https://tum-pbs.github.io/PhiFlow/Web_Interface.html
    """
    def __init__(self,
                 fields=None,
                 stride=None,
                 base_dir='~/phi/data/',
                 summary=None,
                 custom_properties=None,
                 target_scene=None,
                 objects_to_save=None,
                 framerate=None,
                 dt=1.0,
                 controls: dict = None,
                 log_performance=True,
                 **show_config):
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
        self._module = module = inspect.getmodule(inspect.stack()[1].frame)
        doc = module.__doc__
        if doc is None:
            name = os.path.basename(module.__file__)[:-3]
            subtitle = module.__doc__ or module.__file__
        else:
            end_of_line = doc.index('\n')
            name = doc[:end_of_line].strip()
            subtitle = doc[end_of_line:].strip() or None
        App.__init__(self, name, subtitle, fields=None, stride=stride, base_dir=base_dir, summary=summary, custom_properties=custom_properties, target_scene=target_scene, objects_to_save=objects_to_save, framerate=framerate, dt=dt)
        self._initial_field_values = {}
        if fields is None:
            for name in dir(module):
                if not name.startswith('_'):
                    val = getattr(module, name)
                    if isinstance(val, Field) or (isinstance(val, Tensor) and val.shape.spatial_rank > 0):
                        self.add_field(name, lambda name=name: getattr(module, name))
                        self._initial_field_values[name] = val
        else:
            for name in fields:
                self.add_field(name, lambda name=name: getattr(module, name))
                self._initial_field_values[name] = getattr(module, name)
        self.step_exec_event = Event()
        self.step_finished_event = Event()
        self._interrupt = False
        self._elapsed = None
        self.log_performance = log_performance

        if controls:
            for name, value in controls.items():
                if isinstance(value, EditableValue):
                    setattr(self, name, value)
                else:
                    setattr(self, f'value_{name}', value)

        self.add_action("Reset", lambda: self.restore_initial_field_values())

        def async_show():
            show(self, **show_config)

        self._display_thread = Thread(target=async_show, name="ModuleViewer_show", daemon=not _display.KEEP_ALIVE)
        self._display_thread.start()

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
        t = time.perf_counter()
        self.step_exec_event.set()
        self.step_finished_event.wait()
        self._elapsed = time.perf_counter() - t
        if self.log_performance:
            self.log_scalar('step_time', self._elapsed)

    def interrupt(self):
        self._interrupt = True

    def restore_initial_field_values(self, reset_steps=True):
        for name, value in self._initial_field_values.items():
            setattr(self._module, name, value)
        if reset_steps:
            self.steps = 0

