import sys, inspect, os
from phi.app.app import App


class ModelDisplay(object):

    def __init__(self, model, *args, **kwargs):
        self.model = model

    def show(self):
        raise NotImplementedError()

    def play(self):
        self.model.play()


DEFAULT_DISPLAY_CLASS = None

if 'headless' not in sys.argv:
    try:
        from .dash.dash_gui import DashFieldSequenceGui
        DEFAULT_DISPLAY_CLASS = DashFieldSequenceGui
    except:
        print('Failed to load dash GUI')


AUTORUN = 'autorun' in sys.argv


def show(model=None, *args, **kwargs):

    if model is None:
        frame_records = inspect.stack()[1]
        calling_module = inspect.getmodulename(frame_records[1])
        fitting_models = _find_subclasses_in_module(App, calling_module, [])
        assert len(fitting_models) == 1, 'show() called without model but detected %d possible classes: %s' % (len(fitting_models), fitting_models)
        model = fitting_models[0]

    if inspect.isclass(model) and issubclass(model, App):
        model = model()

    called_from_main = inspect.getmodule(model.__class__).__name__ == '__main__'

    display = None
    if DEFAULT_DISPLAY_CLASS is not None:
        display = DEFAULT_DISPLAY_CLASS(model, *args, **kwargs)
    # --- Autorun ---
    if AUTORUN:
        model.info('Starting execution because autorun is enabled.')
        display.play()  # asynchronous call
    # --- Show ---
    if display is None:
        return model
    else:
        return display.show()  # blocking call


def _find_subclasses_in_module(base_class, module_name, result_list):
    subclasses = base_class.__subclasses__()
    for subclass in subclasses:
        if subclass not in result_list:
            mod_name = os.path.basename(inspect.getfile(subclass))[:-3]
            if mod_name == module_name:
                result_list.append(subclass)
            _find_subclasses_in_module(subclass, module_name, result_list)
    return result_list
