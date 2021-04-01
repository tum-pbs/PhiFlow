import sys
import threading
import time
import warnings
from collections import namedtuple
from contextlib import contextmanager
from math import log10
from threading import Thread, Lock
from typing import Tuple

from phi import field
from phi.field import SampledField, Scene


Control = namedtuple('Control', ['name', 'control_type', 'initial', 'value_range', 'kwargs'])


def value_range(control: Control):
    if isinstance(control.value_range, tuple):
        assert len(control.value_range) == 2, f"Tuple must be (min, max) but got length {len(control.value_range)}"
        return control.value_range
    if control.control_type == float:
        log_scale = is_log_control(control)
        if log_scale:
            magn = log10(control.initial)
            val_range = (10.0 ** (magn - 3.2), 10.0 ** (magn + 2.2))
        else:
            if control.initial == 0.0:
                val_range = (-10.0, 10.0)
            elif control.initial > 0:
                val_range = (0., 4. * control.initial)
            else:
                val_range = (2. * control.initial, -2. * control.initial)
    elif control.control_type == int:
        if control.initial == 0:
            val_range = (-10, 10)
        elif control.initial > 0:
            val_range = (0, 4 * control.initial)
        else:
            val_range = (2 * control.initial, -2 * control.initial)
    else:
        raise AssertionError(f"Not a numeric control: {control}")
    return val_range


def is_log_control(control: Control):
    if control.control_type != float:
        return False
    log_scale = control.kwargs.get('log')
    if log_scale is not None:
        return log_scale
    else:
        if control.value_range is None:
            return True
        else:
            return control.value_range[1] / float(control.value_range[0]) > 10


class VisModel:

    def __init__(self, name: str = None, description: str = "", scene: Scene = None):
        self.start_time = time.time()
        """ Time of creation (`App` constructor invocation) """
        self.name = name if name is not None else self.__class__.__name__
        """ Human-readable name. """
        self.description = description
        """ Description to be displayed. """
        self.scene = scene
        """ Directory to which data and logging information should be written as `Scene` instance. """
        self.uses_existing_scene = scene.exist_properties() if scene is not None else False
        self.steps = 0
        """ Counts the number of times `step()` has been called. May be set by the user. """
        self.progress_lock = Lock()
        self.pre_step = []  # callback(vis)
        self.post_step = []  # callback(vis)
        self.progress_available = []  # callback(vis)
        self.progress_unavailable = []  # callback(vis)
        self.growing_dims = ()  # tuple or list, used by GUI to determine whether to scroll to last element
        self.message = None
        self.log_file = None

    def progress(self):
        pass

    @property
    def is_progressing(self) -> bool:
        return self.progress_lock.locked()

    @property
    def can_progress(self) -> bool:
        raise NotImplementedError(self)

    def prepare(self):
        pass

    @property
    def field_names(self) -> tuple:
        raise NotImplementedError(self)

    def get_field(self, field_name) -> SampledField:
        """

        Raises:
            `KeyError` if `field_name` is not a valid field.

        Args:
            field_name:

        Returns:

        """
        raise NotImplementedError(self)

    @property
    def curve_names(self) -> tuple:
        raise NotImplementedError(self)

    def get_curve(self, name: str) -> tuple:
        raise NotImplementedError(self)

    @property
    def controls(self) -> Tuple[Control]:
        raise NotImplementedError(self)

    def get_control_value(self, name):
        raise NotImplementedError(self)

    def set_control_value(self, name, value):
        raise NotImplementedError(self)

    @property
    def action_names(self) -> tuple:
        raise NotImplementedError(self)

    def run_action(self, name):
        raise NotImplementedError(self)

    # Implemented methods

    def _call(self, observers):
        for obs in observers:
            obs(self)


def get_control_by_name(model: VisModel, control_name: str):
    assert isinstance(control_name, str)
    for control in model.controls:
        if control.name == control_name:
            return control
    raise KeyError(f"No control with name '{control_name}'. Available controls: {model.controls}")


def _step_and_wait(model: VisModel, framerate=None):
    t = time.time()
    model.progress()
    if framerate is not None:
        remaining_time = 1.0 / framerate - (time.time() - t)
        if remaining_time > 0:
            time.sleep(remaining_time)


class AsyncPlay:

    def __init__(self, model: VisModel, max_steps, framerate):
        self.model = model
        self.max_steps = max_steps
        self.framerate = framerate
        self.paused = False
        self._finished = False

    def start(self):
        thread = threading.Thread(target=self, name='AsyncPlay')
        thread.start()

    def __call__(self):
        step_count = 0
        while not self.paused:
            _step_and_wait(self.model, framerate=self.framerate)
            step_count += 1
            if self.max_steps and step_count >= self.max_steps:
                break
        self._finished = True

    def pause(self):
        self.paused = True

    def __bool__(self):
        return not self._finished

    def __repr__(self):
        return status_message(self.model, self)


def status_message(model: VisModel, play_status: AsyncPlay or None):
    pausing = "/Pausing" if (play_status and play_status.paused) else ""
    current_action = "Running" if model.is_progressing else "Waiting"
    action = current_action if play_status else "Idle"
    message = f" - {model.message}" if model.message else ""
    return f"{action}{pausing} ({model.steps} steps){message}"


def play_async(model: VisModel, max_steps=None, framerate=None):
    """
    Run a number of steps.

    Args:
        model: Model to progress
        max_steps: (optional) stop when this many steps have been completed (independent of the `steps` variable) or `pause()` is called.
        framerate: Target frame rate in Hz.
    """
    play = AsyncPlay(model, max_steps, framerate)
    play.start()
    return play


def benchmark(model: VisModel, sequence_count):
    # self._pause = False  # TODO allow premature stop
    step_count = 0
    t = time.time()
    for i in range(sequence_count):
        model.progress()
        step_count += 1
        # if self._pause:
        #     break
    time_elapsed = time.time() - t
    return step_count, time_elapsed


class Gui:

    def __init__(self, asynchronous=False):
        """
        Creates a display for the given vis and initializes the configuration.
        This method does not set up the display. It only sets up the Gui object and returns as quickly as possible.
        """
        self.app: VisModel = None
        self.asynchronous = asynchronous
        self.config = {}

    def configure(self, config: dict):
        """
        Updates the GUI configuration.
        This method may only be called while the GUI is not yet visible, i.e. before show() is called.

        Args:
            config: Complete or partial GUI-specific configuration. dictionary mapping from strings to JSON serializable values
        """
        self.config.update(config)

    def get_configuration(self) -> dict:
        """
        Returns the current configuration of the GUI.
        The returned dictionary may only contain serializable values and all keys must be strings.
        The configuration can be passed to another instance of this class using set_configuration().
        """
        return self.config

    def setup(self, app: VisModel):
        """
        Sets up all necessary GUI components.
        
        The GUI can register callbacks with the vis to be informed about vis-state changes induced externally.
        The vis can be assumed to be prepared when this method is called.
        
        This method is called after set_configuration() but before show()

        Args:
          app: vis to be displayed, may not be prepared or be otherwise invalid at this point.
        """
        self.app = app

    def show(self, caller_is_main: bool):
        """
        Displays the previously setup GUI.
        This method is blocking and returns only when the GUI is hidden.

        This method will always be called after setup().

        Args:
            caller_is_main: True if the calling script is the __main__ module.
        """
        pass

    def auto_play(self):
        """
        Called if `autorun=True`.
        If no Gui is specified, `App.run()` is called instead.
        """
        framerate = self.config.get('framerate', None)
        play_async(self.app, framerate=framerate)


def default_gui() -> Gui:
    if GUI_OVERRIDES:
        return GUI_OVERRIDES[-1]
    if 'google.colab' in sys.modules or 'ipykernel' in sys.modules:
        options = ['widgets']
    else:
        options = ['dash', 'console']
    for option in options:
        try:
            return get_gui(option)
        except ImportError as import_error:
            warnings.warn(f"{option} user interface is unavailable because of missing dependency: {import_error}.")
    raise RuntimeError("No user interface available.")


def get_gui(gui: str or Gui) -> Gui:
    if GUI_OVERRIDES:
        return GUI_OVERRIDES[-1]
    if isinstance(gui, str):
        if gui == 'dash':
            from ._dash.dash_gui import DashGui
            return DashGui()
        elif gui == 'console':
            from ._console import ConsoleGui
            return ConsoleGui()
        elif gui == 'matplotlib':
            from ._matplotlib.matplotlib_gui import MatplotlibGui
            return MatplotlibGui()
        elif gui == 'widgets':
            from ._widgets import WidgetsGui
            return WidgetsGui()
        else:
            raise NotImplementedError(f"No display available with name {gui}")
    elif isinstance(gui, Gui):
        return gui
    else:
        raise ValueError(gui)


GUI_OVERRIDES = []


@contextmanager
def force_use_gui(gui: Gui):
    GUI_OVERRIDES.append(gui)
    try:
        yield None
    finally:
        assert GUI_OVERRIDES.pop(-1) is gui


class GuiInterrupt(KeyboardInterrupt):
    pass


def gui_interrupt(*args, **kwargs):
    raise GuiInterrupt()


def show(app: VisModel or None = None, play=True, gui: Gui or str = None, keep_alive=True, **config):
    """
    Launch the registered user interface (web interface by default).
    
    This method may block until the GUI is closed.
    
    This method prepares the vis before showing it. No more fields should be added to the vis after this method is invoked.
    
    Also see the user interface documentation at https://tum-pbs.github.io/PhiFlow/Visualization.html

    Args:
      app: App or None:  (Default value = None)
      play: If true, invokes `App.play()`. The default value is False unless "autorun" is passed as a command line argument.
      app: optional) the application to display. If unspecified, searches the calling script for a subclass of App and instantiates it.
      gui: (optional) class of GUI to use
      keep_alive: Whether the GUI keeps the vis alive. If `False`, the program will exit when the main script is finished.
      **config: additional GUI configuration parameters.
        For a full list of parameters, see the respective GUI documentation at https://tum-pbs.github.io/PhiFlow/Visualization.html
    """
    assert isinstance(app, VisModel), f"show() first argument must be an App instance but got {app}"
    app.prepare()
    # --- Setup Gui ---
    gui = default_gui() if gui is None else get_gui(gui)
    gui.configure(config)
    gui.setup(app)
    if play:
        gui.auto_play()
    if gui.asynchronous:
        display_thread = Thread(target=lambda: gui.show(True), name="AsyncGui", daemon=not keep_alive)
        display_thread.start()
    else:
        gui.show(True)  # may be blocking call


def display_name(python_name):
    n = list(python_name)
    n[0] = n[0].upper()
    for i in range(1, len(n)):
        if n[i] == "_":
            n[i] = " "
            if len(n) > i + 1:
                n[i + 1] = n[i + 1].upper()
    return "".join(n)


def select_channel(value: SampledField, channel: str or None):
    if channel is None:
        return value
    elif channel == 'abs':
        if value.vector.exists:
            return field.vec_abs(value)
        else:
            return value
    else:  # x, y, z
        if channel in value.shape.spatial and 'vector' in value.shape:
            comp_index = value.shape.spatial.index(channel)
            return value.unstack('vector')[comp_index]
        elif 'vector' in value.shape:
            raise ValueError(f"Dimension {value} unavailable.")
        else:
            return value
