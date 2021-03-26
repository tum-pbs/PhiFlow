from threading import Event, Thread

from IPython import get_ipython

from ._app import App
from ._display import show
from . import _display, ModuleViewer
from ..field import Field


def notebook_variables(shell):
    all_vars = shell.user_ns
    hidden = shell.user_ns_hidden
    variables = {n: v for n, v in all_vars.items() if not n.startswith('_') and n not in hidden}
    return variables


class NotebookViewer(ModuleViewer):

    def __init__(self,
                 **show_config):
        App.__init__(self, "Notebook")
        self.shell = get_ipython()
        self._initial_field_values = {}
        for name, val in notebook_variables(self.shell).items():
            if isinstance(val, Field):
                self.add_field(name, lambda name=name: self.shell.user_ns[name])
                self._initial_field_values[name] = val
        self.step_exec_event = Event()
        self.step_finished_event = Event()

        if 'gui' not in show_config:
            show_config['gui'] = 'widgets'

        def async_show():
            show(self, **show_config)

        self._display_thread = Thread(target=async_show, name="ModuleViewer_show", daemon=not _display.KEEP_ALIVE)
        self._display_thread.start()
