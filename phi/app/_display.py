import sys
import inspect
import os
import warnings

from phi.app._app import App


class AppDisplay(object):

    def __init__(self, app: App):
        """
        Creates a display for the given app and initializes the configuration.
        This method does not set up the display. It only sets up the AppDisplay object and returns as quickly as possible.

        Args:
          app: app to be displayed, may not be prepared or be otherwise invalid at this point.
        """
        self.app = app
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

    def setup(self):
        """
        Sets up all necessary GUI components.
        
        The GUI can register callbacks with the app to be informed about app-state changes induced externally.
        The app can be assumed to be prepared when this method is called.
        
        This method is called after set_configuration() but before show()
        
        The return value of this method will be returned by show(app).
        """
        pass

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

    def play(self):
        """
        Called if AUTORUN is enabled.
        If no Display is specified, App.run() is called instead.
        """
        self.app.play()


DEFAULT_DISPLAY_CLASS = None

if 'google.colab' not in sys.modules:
    try:
        from ._dash.dash_gui import DashGui
        DEFAULT_DISPLAY_CLASS = DashGui
    except ImportError as import_error:
        warnings.warn(f"Web interface is disabled because of missing dependency: {import_error}. To install all dependencies, run $ pip install phiflow")
if DEFAULT_DISPLAY_CLASS is None:
    try:
        from ._matplotlib.matplotlib_gui import MatplotlibGui
        DEFAULT_DISPLAY_CLASS = MatplotlibGui
    except ImportError as import_error:
        warnings.warn(f"Matplotlib interface is disabled because of missing dependency: {import_error}. To install all dependencies, run $ pip install phiflow")


def _display_class(gui: str or type):
    if isinstance(gui, str):
        if gui == 'dash':
            from ._dash.dash_gui import DashGui
            return DashGui
        elif gui == 'console':
            from ._console import ConsoleGui
            return ConsoleGui
        elif gui == 'matplotlib':
            from ._matplotlib.matplotlib_gui import MatplotlibGui
            return MatplotlibGui
        elif gui == 'widgets':
            from ._widgets._widgets_gui import WidgetsGui
            return WidgetsGui
        else:
            raise NotImplementedError(f"No display available with name {gui}")
    elif isinstance(gui, type):
        return gui
    else:
        raise ValueError(gui)


KEEP_ALIVE = True  # internal; whether the UI keeps the app alive


def show(app: App or None = None, autorun=False, gui: AppDisplay or str = None, **config):
    """
    Launch the registered user interface (web interface by default).
    
    This method may block until the GUI is closed.
    
    This method prepares the app before showing it. No more fields should be added to the app after this method is invoked.
    
    Also see the user interface documentation at https://tum-pbs.github.io/PhiFlow/Web_Interface.html

    Args:
      autorun: If true, invokes `App.play()`. The default value is False unless "autorun" is passed as a command line argument.
      app: optional) the application to display. If unspecified, searches the calling script for a subclass of App and instantiates it.
      gui: (optional) class of GUI to use
      config: additional GUI configuration parameters.
    For a full list of parameters, see https://tum-pbs.github.io/PhiFlow/Web_Interface.html
      app: App or None:  (Default value = None)
      **config: 

    Returns:
      reference to the GUI, depending on the implementation. For the web interface this may be the web server instance.

    """

    if app is None:
        frame_records = inspect.stack()[1]
        calling_module = inspect.getmodulename(frame_records[1])
        fitting_models = _find_subclasses_in_module(App, calling_module, [])
        assert len(fitting_models) == 1, 'show() called without model but detected %d possible classes: %s' % (len(fitting_models), fitting_models)
        app = fitting_models[0]

    if inspect.isclass(app) and issubclass(app, App):
        app = app()

    app.prepare()

    display = None
    gui = gui or DEFAULT_DISPLAY_CLASS
    gui = _display_class(gui)
    if gui is not None:
        display = gui(app)
        display.configure(config)
        display.setup()
    # --- Autorun ---
    if autorun:
        if display is None:
            app.info('Starting execution because autorun is enabled.')
            app.play()  # asynchronous call
        else:
            display.play()
    # --- Show ---
    if display is None:
        warnings.warn('show() has no effect because no display is available. To use the web interface, run $ pip install phiflow')
        return app
    else:
        return display.show(True)  # may be blocking call


def _find_subclasses_in_module(base_class, module_name, result_list):
    subclasses = base_class.__subclasses__()
    for subclass in subclasses:
        if subclass not in result_list:
            mod_name = os.path.basename(inspect.getfile(subclass))[:-3]
            if mod_name == module_name:
                result_list.append(subclass)
            _find_subclasses_in_module(subclass, module_name, result_list)
    return result_list
