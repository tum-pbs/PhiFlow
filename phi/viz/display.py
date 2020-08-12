import sys
import inspect
import os
import warnings

from phi.app.app import App


class AppDisplay(object):

    def __init__(self, app):
        """
Creates a display for the given app and initializes the configuration.
This method does not set up the display. It only sets up the AppDisplay object and returns as quickly as possible.
        :param app: app to be displayed, may not be prepared or be otherwise invalid at this point.
        """
        self.app = app
        self.config = {}

    def configure(self, config):
        # type: (dict) -> None
        """
Updates the GUI configuration.
This method may only be called while the GUI is not yet visible, i.e. before show() is called.
        :param config: Complete or partial GUI-specific configuration. dictionary mapping from strings to JSON serializable values
        """
        self.config.update(config)

    def get_configuration(self):
        # type: () -> dict
        """
Returns the current configuration of the GUI.
The returned dictionary may only contain serializable values and all keys must be strings.
The configuration can be passed to another instance of this class using set_configuration().
        :rtype: dict
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

    def show(self, caller_is_main):
        # type: (bool) -> bool
        """
Displays the previously setup GUI.
This method is blocking and returns only when the GUI is hidden.

This method will always be called after setup().
        :param caller_is_main: True if the calling script is the __main__ module.
        :return: Whether the GUI was displayed
        :rtype: bool
        """
        return False

    def play(self):
        """
Called if AUTORUN is enabled.
If no Display is specified, App.run() is called instead.
        """
        self.app.play()


DEFAULT_DISPLAY_CLASS = None

if 'headless' not in sys.argv:
    try:
        from .dash.dash_gui import DashGui
        DEFAULT_DISPLAY_CLASS = DashGui
    except ImportError as import_error:
        warnings.warn('GUI is disabled because of missing dependencies: %s. To install all dependencies, run $ pip install phiflow[gui]' % import_error)


AUTORUN = 'autorun' in sys.argv


def show(app=None, **config):

    frame_records = inspect.stack()[1]
    calling_module = inspect.getmodulename(frame_records[1])
    python_file = os.path.basename(sys.argv[0])[:-3]

    if app is None:
        fitting_models = _find_subclasses_in_module(App, calling_module, [])
        assert len(fitting_models) == 1, 'show() called without model but detected %d possible classes: %s' % (len(fitting_models), fitting_models)
        app = fitting_models[0]

    if inspect.isclass(app) and issubclass(app, App):
        app = app()

    called_from_main = inspect.getmodule(app.__class__).__name__ == '__main__' or calling_module == python_file

    app.prepare()

    display = None
    if DEFAULT_DISPLAY_CLASS is not None:
        display = DEFAULT_DISPLAY_CLASS(app)
        display.configure(config)
        display.setup()
    # --- Autorun ---
    if AUTORUN:
        if display is None:
            app.info('Starting execution because autorun is enabled.')
            app.play()  # asynchronous call
        else:
            display.play()
    # --- Show ---
    if display is None:
        warnings.warn('show() has no effect because no display is available. To use the web interface, run $ pip install phiflow[gui]')
        return app
    else:
        return display.show(called_from_main)  # blocking call


def _find_subclasses_in_module(base_class, module_name, result_list):
    subclasses = base_class.__subclasses__()
    for subclass in subclasses:
        if subclass not in result_list:
            mod_name = os.path.basename(inspect.getfile(subclass))[:-3]
            if mod_name == module_name:
                result_list.append(subclass)
            _find_subclasses_in_module(subclass, module_name, result_list)
    return result_list
