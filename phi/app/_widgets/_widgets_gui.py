import asyncio
import sys
import time
import warnings

import ipywidgets as widgets
from IPython import get_ipython
from IPython.display import display
from ipywidgets import HBox, VBox
from ipywidgets.widgets.interaction import show_inline_matplotlib_plots
from matplotlib import pyplot as plt

from phi.field.plt import plot
from .. import App
from .._app import display_name
from .._display import Gui
from .._display_util import ordered_field_names
from .._viewer import Viewer
from ...field import SampledField


class WidgetsGui(Gui):

    def __init__(self):
        Gui.__init__(self, asynchronous=False)
        self.shell = get_ipython()
        self.kernel = self.shell.kernel
        self.loop_parent = None  # set when loop is created
        # Status
        self.max_step = -1  # play up until this step if set, determines if playing
        self._interrupted = False
        self.fields = None
        self.field = None
        self._in_loop = None
        # Components will be created during show()
        self.figure_display = None
        self.status = None
        self.buttons = None
        self.field_select = None
        self._graphs_enabled = False

    def setup(self, app: App):
        Gui.setup(self, app)
        self.fields = ordered_field_names(self.app, self.config.get('display'))
        self.field = self.fields[0]
        app.pre_step.append(self.pre_step)
        app.post_step.append(self.post_step)
        if isinstance(app, Viewer):
            app.on_loop_start.append(self.on_loop_start)
            app.on_loop_exit.append(self.on_loop_exit)

        def custom_traceback(exc_tuple=None, filename=None, tb_offset=None, exception_only=False, running_compiled_code=False):
            etype, value, tb = sys.exc_info()
            if etype == GuiInterrupt:
                return
            else:
                normal_traceback(exc_tuple, filename, tb_offset, exception_only, running_compiled_code)

        normal_traceback = self.shell.showtraceback
        self.shell.showtraceback = custom_traceback

    def show(self, caller_is_main: bool) -> bool:
        self.figure_display = widgets.Output()
        # self.status = widgets.Output()
        # self.status.append_stdout('Status')
        play_button = widgets.Button(description="Play")
        play_button.on_click(self.play)
        pause_button = widgets.Button(description="Pause")
        pause_button.on_click(self.pause)
        step_button = widgets.Button(description="Step")
        step_button.on_click(self.step)
        interrupt_button = widgets.Button(description="Break")
        interrupt_button.on_click(self.interrupt)
        self.buttons = HBox([play_button, pause_button, step_button, interrupt_button])
        self.buttons.layout.visibility = 'hidden'
        self.status = widgets.Label(value=self._get_status())
        layout = [self.buttons, self.status]

        self.field_select = widgets.Dropdown(options=self.fields, value=self.fields[0], description='Display:')
        self.field_select.layout.visibility = 'visible' if len(self.app.fieldnames) > 1 else 'hidden'
        self.field_select.observe(lambda change: self.show_field(change['new']) if change['type'] == 'change' and change['name'] == 'value' else None)
        layout.append(self.field_select)

        layout.append(self.figure_display)
        layout = VBox(layout)

        self.update_widgets()
        display(layout)
        return True

    def _get_status(self):
        message = f" - {self.app.message}" if self.app.message else ""
        if self._in_loop is None:  # no loop yet
            return self.app.message or ""
        elif self._in_loop is True:
            playing = self.max_step is None or self.max_step >= self.app.steps
            action = "Playing" if playing else "Idle"
            return f"{action} ({self.app.steps} steps){message}"
        else:
            return f"Finished {self.app.steps} steps."

    def show_field(self, field: str):
        self.field = field
        self.update_widgets()

    def update_widgets(self):
        self.status.value = self._get_status()
        scalars = self.app.get_logged_scalars()
        if not self._graphs_enabled and scalars:
            self._graphs_enabled = True
            self.field_select.options = [*self.fields, 'Scalars']
            self.field_select.layout.visibility = 'visible'
        # Figure
        self.figure_display.clear_output()
        if 'style' in self.config:
            with plt.style.context(self.config['style']):
                self._plot(self.field, self.figure_display)
        else:
            self._plot(self.field, self.figure_display)

    def _plot(self, selection: str, output: widgets.Output):
        with output:
            if selection == 'Scalars':
                plt.figure(figsize=(12, 5))
                for name in self.app.get_logged_scalars():
                    plt.plot(*self.app.get_scalar_curve(name), label=display_name(name))
                plt.legend()
                plt.tight_layout()
                show_inline_matplotlib_plots()
            else:
                field = self.app.get_field(selection)
                if isinstance(field, SampledField):
                    plot(field, figsize=(12, 5))
                    show_inline_matplotlib_plots()
                else:
                    self.figure_display.append_stdout(f"{selection} = {field}")


    def play(self, _):
        self.max_step = None

    def auto_play(self):
        self.max_step = None

    def pause(self, _):
        self.max_step = self.app.steps

    def step(self, _):
        if self.max_step is None:
            return
        else:
            self.max_step += 1

    def interrupt(self, _):
        self._interrupted = True

    def on_loop_start(self, _):
        self._in_loop = True
        self._interrupted = False
        self.buttons.layout.visibility = 'visible'
        self.loop_parent = (self.kernel._parent_ident, self.kernel._parent_header)
        self.kernel.shell_handlers["execute_request"] = lambda *e: self.events.append(e)
        self.events = []

    def pre_step(self, app):
        self._process_kernel_events()
        while self.max_step is not None and self.max_step < app.steps:
            time.sleep(.1)
            self._process_kernel_events()
            if self._interrupted:
                raise GuiInterrupt()
        if self._interrupted:
            raise GuiInterrupt()
        return  # runs loop iteration, then calls post_step

    def _process_kernel_events(self, n=10):
        for _ in range(n):
            self.kernel.set_parent(*self.loop_parent)  # ensure stdout still happens in the same cell
            self.kernel.do_one_iteration()
            self.kernel.set_parent(*self.loop_parent)

    def post_step(self, _):
        self.update_widgets()

    def on_loop_exit(self, _):
        self._in_loop = False
        self.buttons.layout.visibility = 'hidden'
        self.update_widgets()
        self._process_kernel_events()
        self.kernel.shell_handlers["execute_request"] = self.kernel.execute_request
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.call_soon(lambda: _replay_events(self.shell, self.events))
        else:
            warnings.warn("Automatic execution of scheduled cells only works with asyncio based ipython")


def _replay_events(shell, events):
    kernel = shell.kernel
    sys.stdout.flush()
    sys.stderr.flush()
    for stream, ident, parent in events:
        kernel.set_parent(ident, parent)
        if kernel.aborted:  # not available for Colab notebooks
            return  # kernel._send_abort_reply(stream, parent, ident)
        else:
            kernel.execute_request(stream, ident, parent)


class GuiInterrupt(KeyboardInterrupt):
    pass
