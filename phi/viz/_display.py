import sys
import inspect
import os
import warnings
from contextlib import contextmanager
from threading import Thread

from phi.viz._app import App, play_async
from ._user_namespace import default_user_namespace, UserNamespace
from ._viewer import create_viewer, Viewer
from ..field import SampledField, Field, Scene
from ..field._scene import _slugify_filename


class Gui:

    def __init__(self, asynchronous=False):
        """
        Creates a display for the given viz and initializes the configuration.
        This method does not set up the display. It only sets up the Gui object and returns as quickly as possible.
        """
        self.app = None
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

    def setup(self, app: App):
        """
        Sets up all necessary GUI components.
        
        The GUI can register callbacks with the viz to be informed about viz-state changes induced externally.
        The viz can be assumed to be prepared when this method is called.
        
        This method is called after set_configuration() but before show()

        Args:
          app: viz to be displayed, may not be prepared or be otherwise invalid at this point.
        """
        self.app = app

    def show(self, caller_is_main: bool) -> bool:
        """
        Displays the previously setup GUI.
        This method is blocking and returns only when the GUI is hidden.

        This method will always be called after setup().

        Args:
            caller_is_main: True if the calling script is the __main__ module.

        Returns:
            Whether the GUI was displayed
        """
        return False

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


def show(app: App or None = None, play=True, gui: Gui or str = None, keep_alive=True, **config):
    """
    Launch the registered user interface (web interface by default).
    
    This method may block until the GUI is closed.
    
    This method prepares the viz before showing it. No more fields should be added to the viz after this method is invoked.
    
    Also see the user interface documentation at https://tum-pbs.github.io/PhiFlow/Web_Interface.html

    Args:
      app: App or None:  (Default value = None)
      play: If true, invokes `App.play()`. The default value is False unless "autorun" is passed as a command line argument.
      app: optional) the application to display. If unspecified, searches the calling script for a subclass of App and instantiates it.
      gui: (optional) class of GUI to use
      config: additional GUI configuration parameters.
        For a full list of parameters, see https://tum-pbs.github.io/PhiFlow/Web_Interface.html
      keep_alive: Whether the GUI keeps the viz alive. If `False`, the program will exit when the main script is finished.
    """
    assert isinstance(app, App), f"show() first argument must be an App instance but got {app}"
    app.prepare()
    # --- Setup Gui ---
    gui = default_gui() if gui is None else get_gui(gui)
    gui.configure(config)
    gui.setup(app)
    if play:
        gui.auto_play()
    if gui.asynchronous:
        display_thread = Thread(target=lambda: gui.show(True), name="ModuleViewer_show", daemon=not keep_alive)
        display_thread.start()
    else:
        gui.show(True)  # may be blocking call


def _find_subclasses_in_module(base_class, module_name, result_list):
    subclasses = base_class.__subclasses__()
    for subclass in subclasses:
        if subclass not in result_list:
            mod_name = os.path.basename(inspect.getfile(subclass))[:-3]
            if mod_name == module_name:
                result_list.append(subclass)
            _find_subclasses_in_module(subclass, module_name, result_list)
    return result_list


def view(*fields: str or SampledField,
         play: bool = True,
         gui=None,
         name: str = None,
         description: str = None,
         scene: bool or Scene = None,
         controls=None,
         keep_alive=True,
         select: str or tuple or list = 'frames,',
         framerate=None,
         **config) -> Viewer:
    """
    Show `fields` in a graphical user interface.

    `fields` may contain instances of `Field` or variable names of top-level variables (main module or Jupyter notebook).
    During loops, e.g. `view().range()`, the variable status is tracked and the GUI is updated.

    Args:
        *fields: (Optional) Contents to be displayed. Either variable names or values.
            For field instances, all variables referencing the value will be shown.
            If not provided, the user namespace is searched for Field variables.
        play: Whether to immediately start executing loops.
        gui: (Optional) Name of GUI as `str` or GUI class.
        name: Name to display in GUI and use for the output directory if `scene=True`
        framerate: Target frame rate in Hz. Play will not step faster than the framerate. `None` for unlimited frame rate.
        select: Dimension names along which one item to show is selected.
            For each `select` dimension, an associated selection slider will be created.
        **config: Additional GUI configuration arguments.

    Returns:
        `Viewer`
    """
    user_namespace = default_user_namespace()
    variables = _default_field_variables(user_namespace, fields)
    if scene is None:
        scene = not ('google.colab' in sys.modules or 'ipykernel' in sys.modules)
    if scene is False:
        scene = None
    elif scene is True:
        scene = Scene.create(os.path.join("~", "phi", _slugify_filename(name or user_namespace.get_reference())))
    else:
        assert isinstance(scene, Scene)
    name = name or user_namespace.get_title()
    description = description or user_namespace.get_description()
    gui = default_gui() if gui is None else get_gui(gui)
    viewer = create_viewer(user_namespace, variables, name, description, scene, asynchronous=gui.asynchronous, controls=controls)
    show(viewer, play=play, gui=gui, keep_alive=keep_alive, framerate=framerate, select=select, **config)
    return viewer


def _default_field_variables(user_namespace: UserNamespace, fields: tuple):
    names = []
    values = []
    if len(fields) == 0:  # view all Fields
        user_variables = user_namespace.list_variables(only_public=True, only_current_scope=True)
        for name, val in user_variables.items():
            if isinstance(val, Field):
                names.append(name)
                values.append(val)
    elif any(not isinstance(f, str) for f in fields):  # find variable names
        user_variables = user_namespace.list_variables()
        for field in fields:
            if isinstance(field, str):
                split = [n.strip() for n in field.split(',')]
                names.extend(split)
                values.extend([user_namespace.get_variable(n, default=None) for n in split])
            else:
                for name, val in user_variables.items():
                    if val is field:
                        names.append(name)
                        values.append(field)
    assert names, "Nothing to view. Store SampledField instances in top-level variables to have them auto-detected."
    return {n: v for n, v in zip(names, values)}
