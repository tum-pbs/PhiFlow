import sys, inspect, os
from phi.model import FieldSequenceModel


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
        from phi.viz.dash_gui import DashFieldSequenceGui
        DEFAULT_DISPLAY_CLASS = DashFieldSequenceGui
    except:
        print('Failed to load dash GUI')


AUTORUN = 'autorun' in sys.argv


def show(model=None, *args, **kwargs):

    if model is None:
        all_models = FieldSequenceModel.__subclasses__()
        frame_records = inspect.stack()[1]
        calling_module = inspect.getmodulename(frame_records[1])
        for m in all_models:
            m_modname = os.path.basename(inspect.getfile(m))[:-3]
            if m_modname == calling_module:
                model = m
        assert model is not None, 'No model found.'

    if inspect.isclass(model) and issubclass(model, FieldSequenceModel):
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

