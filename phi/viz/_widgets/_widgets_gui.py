import asyncio
import sys
import time
import traceback
import warnings

import ipywidgets as widgets
from IPython import get_ipython
from IPython.display import display
from ipywidgets import HBox, VBox
from ipywidgets.widgets.interaction import show_inline_matplotlib_plots
from matplotlib import pyplot as plt

from phi.viz._matplotlib._matplotlib_plots import plot
from .. import App
from .._app import display_name
from .._display import Gui
from .._display_util import ordered_field_names
from .._viewer import Viewer
from ...field import SampledField
from ...math._shape import parse_dim_order


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
        self._last_plot_update_time = None
        # Components will be created during show()
        self.figure_display = None
        self.status = None
        self.buttons = None
        self.field_select = None
        self.dim_sliders = {}
        self._graphs_enabled = False
        self._last_plot = None

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
        # Icons: https://en.wikipedia.org/wiki/Media_control_symbols️  ⏮ ⏭ ⏺ ⏏
        play_button = widgets.Button(description="️▶ Play")
        play_button.on_click(self.play)
        pause_button = widgets.Button(description="⏸ Pause")
        pause_button.on_click(self.pause)
        step_button = widgets.Button(description="⏯ Step")
        step_button.on_click(self.step)
        interrupt_button = widgets.Button(description="⏹ Break")
        interrupt_button.on_click(self.interrupt)
        self.buttons = HBox([play_button, pause_button, step_button, interrupt_button])
        self.buttons.layout.visibility = 'hidden'
        self.status = widgets.Label(value=self._get_status())
        self.field_select = widgets.Dropdown(options=[*self.fields, 'Scalars'], value=self.fields[0], description='Display:')
        self.field_select.layout.visibility = 'visible' if len(self.app.fieldnames) > 1 else 'hidden'
        self.field_select.observe(lambda change: self.show_field(change['new']) if change['type'] == 'change' and change['name'] == 'value' else None)
        dim_sliders = []
        for sel_dim in parse_dim_order(self.config.get('select', [])):
            slider = widgets.IntSlider(value=0, min=0, max=0, description=sel_dim, continuous_update=False)
            self.dim_sliders[sel_dim] = slider
            dim_sliders.append(slider)
            slider.observe(lambda e: self.update_widgets(), 'value')
        layout = VBox([
            self.buttons,
            self.status,
            HBox([self.field_select, *dim_sliders]),  # sliders in line with select
            self.figure_display
        ] if len(dim_sliders) <= 1 else [
            self.buttons,
            self.status,
            self.field_select,
            HBox(dim_sliders),  # sliders below field select
            self.figure_display
        ])
        # Show initial value and display UI
        self.update_widgets()
        display(layout)
        return True

    def _get_status(self):
        message = f" - {self.app.message}" if self.app.message else ""
        if self._in_loop is None:  # no loop yet
            return self.app.message or ""
        elif self._in_loop is True:
            action = "Playing" if self._is_playing() else "Idle"
            return f"{action} ({self.app.steps} steps){message}"
        else:
            return f"Finished {self.app.steps} steps."

    def _is_playing(self):
        return self.max_step is None or self.max_step >= self.app.steps

    def show_field(self, field: str):
        self.field = field
        self.update_widgets()

    def update_widgets(self, plot=True, scroll_to_last=False):
        self.status.value = self._get_status()
        scalars = self.app.get_logged_scalars()
        if not self._graphs_enabled and scalars:
            self._graphs_enabled = True
            self.field_select.layout.visibility = 'visible'
        for sel_dim, slider in self.dim_sliders.items():
            if self.field == 'Scalars':
                slider.layout.visibility = 'hidden'
            else:
                field = self.app.get_field(self.field)
                if isinstance(field, SampledField) and sel_dim in field.shape:
                    slider.layout.visibility = 'visible'
                    slider.max = field.shape.get_size(sel_dim) - 1
                    if scroll_to_last and sel_dim in self.app.growing_dims:
                        slider.value = field.shape.get_size(sel_dim) - 1
                else:
                    slider.layout.visibility = 'hidden'
        # Figure
        if plot:
            if 'style' in self.config:
                with plt.style.context(self.config['style']):
                    self._plot(self.field, self.figure_display)
            else:
                self._plot(self.field, self.figure_display)

    def _plot(self, field_name: str, output: widgets.Output):
        dim_selection = {name: slider.value for name, slider in self.dim_sliders.items()}
        if self._last_plot == (self.app.steps, self.field, dim_selection):
            return  # plot has not changed
        self._last_plot = (self.app.steps, self.field, dim_selection)
        self.figure_display.clear_output()
        with output:
            try:
                if field_name == 'Scalars':
                    plt.figure(figsize=(12, 5))
                    for name in self.app.get_logged_scalars():
                        plt.plot(*self.app.get_scalar_curve(name), label=display_name(name))
                    plt.legend()
                    plt.tight_layout()
                    show_inline_matplotlib_plots()
                else:
                    field = self.app.get_field(field_name)
                    # self.figure_display.append_stdout(f"Accessing {dim_selection} in {field}")
                    field = field[dim_selection]
                    if isinstance(field, SampledField):
                        plot(field, figsize=(12, 5))
                        show_inline_matplotlib_plots()
                    else:
                        self.figure_display.append_stdout(f"{field_name} = {field}")
            except Exception:
                self.figure_display.append_stdout(traceback.format_exc())
        self._last_plot_update_time = time.time()

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
        while not self._is_playing():
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
        if self._is_playing():  # maybe skip update
            update_interval = self.config.get('update_interval')
            if update_interval is None:
                update_interval = 2.5 if 'google.colab' in sys.modules else 1.2
            elapsed = time.time() - self._last_plot_update_time
            self.update_widgets(plot=elapsed >= update_interval, scroll_to_last=True)
        else:
            self.update_widgets(scroll_to_last=True)

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
