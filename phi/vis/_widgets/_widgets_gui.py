import asyncio
import sys
import time
import traceback
import warnings
from contextlib import contextmanager
from math import log10

import ipywidgets as widgets
from IPython import get_ipython
from IPython.display import display
from ipywidgets import HBox, VBox
from ipywidgets.widgets.interaction import show_inline_matplotlib_plots
from matplotlib import pyplot as plt

from phi.math._shape import parse_dim_order
from phi.field import SampledField
from .._matplotlib._matplotlib_plots import plot
from .._vis_base import Gui, VisModel, display_name, GuiInterrupt, select_channel, value_range, is_log_control, Action


class WidgetsGui(Gui):

    def __init__(self):
        Gui.__init__(self, asynchronous=False)
        self.shell = get_ipython()
        self.kernel = self.shell.kernel
        self.loop_parent = None  # set when loop is created
        # Status
        self.waiting_steps = 0
        self._interrupted = False
        self.fields = None
        self.field = None
        self._in_loop = None
        self._last_plot_update_time = None
        # Components will be created during show()
        self.figure_display = None
        self.status = None
        self.field_select = None
        self.vector_select = None
        self.dim_sliders = {}
        self._graphs_enabled = False

    def setup(self, app: VisModel):
        Gui.setup(self, app)
        self.fields = self.app.field_names
        self.field = self.fields[0]
        app.pre_step.append(self.pre_step)
        app.post_step.append(self.post_step)
        app.progress_available.append(self.on_loop_start)
        app.progress_unavailable.append(self.on_loop_exit)

        def custom_traceback(exc_tuple=None, filename=None, tb_offset=None, exception_only=False, **kwargs):
            etype, value, tb = sys.exc_info()
            if etype == GuiInterrupt:
                return
            else:
                normal_traceback(exc_tuple, filename, tb_offset, exception_only, **kwargs)

        normal_traceback = self.shell.showtraceback
        self.shell.showtraceback = custom_traceback

    def show(self, caller_is_main: bool) -> bool:
        self.figure_display = widgets.Output()
        # Icons: https://en.wikipedia.org/wiki/Media_control_symbols️  ⏮ ⏭ ⏺ ⏏
        self.play_button = widgets.Button(description="▶ Play")
        self.play_button.on_click(self.play)
        self.pause_button = widgets.Button(description="▌▌ Pause")
        self.pause_button.on_click(self.pause)
        self.step_button = widgets.Button(description="Step")
        self.step_button.on_click(self.step)
        self.interrupt_button = widgets.Button(description="⏏ Break")
        self.interrupt_button.on_click(self.interrupt)
        self.progression_buttons = [self.play_button, self.pause_button, self.step_button, self.interrupt_button]
        for button in self.progression_buttons:
            button.layout.visibility = 'hidden'
        self.status = widgets.Label(value=self._get_status())
        self.field_select = widgets.Dropdown(options=[*self.fields, 'Scalars'], value=self.fields[0], description='Display:')
        self.field_select.layout.visibility = 'visible' if len(self.app.field_names) > 1 else 'hidden'
        self.field_select.observe(lambda change: self.show_field(change['new']) if change['type'] == 'change' and change['name'] == 'value' else None)
        dim_sliders = []
        for sel_dim in parse_dim_order(self.config.get('select', [])):
            slider = widgets.IntSlider(value=0, min=0, max=0, description=sel_dim, continuous_update=False)
            self.dim_sliders[sel_dim] = slider
            dim_sliders.append(slider)
            slider.observe(lambda e: None if IGNORE_EVENTS else self.update_widgets(), 'value')
        self.vector_select = widgets.ToggleButtons(
            options=['🡡', 'x', 'y', 'z', '⬤'],
            value='🡡',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltips=['Vectors as arrows', 'x component as heatmap', 'y component as heatmap', 'z component as heatmap', 'vector length as heatmap'],
            #  icons=['check'] * 3
        )
        self.vector_select.style.button_width = '30px'
        self.vector_select.observe(lambda e: None if IGNORE_EVENTS else self.update_widgets(), 'value')
        control_components = []
        for control in self.app.controls:
            val_min, val_max = value_range(control)
            if control.control_type == int:
                control_component = widgets.IntSlider(control.initial, min=val_min, max=val_max, step=1, description=display_name(control.name))
            elif control.control_type == float:
                if is_log_control(control):
                    val_min, val_max = log10(val_min), log10(val_max)
                    control_component = widgets.FloatLogSlider(control.initial, base=10, min=val_min, max=val_max, description=display_name(control.name))
                else:
                    control_component = widgets.FloatSlider(control.initial, min=val_min, max=val_max, description=display_name(control.name))
            elif control.control_type == bool:
                control_component = widgets.Checkbox(control.initial, description=display_name(control.name))
            elif control.control_type == str:
                if not val_max:
                    control_component = widgets.Text(value=control.initial, placeholder=control.initial, description=display_name(control.name))
                else:
                    control_component = widgets.Dropdown(options=control.value_range, value=control.initial, description=display_name(control.name))
            else:
                raise ValueError(f'Illegal control type: {control.control_type}')
            control_component.observe(lambda e, c=control: None if IGNORE_EVENTS else self.app.set_control_value(c.name, e['new']), 'value')
            control_components.append(control_component)
        action_buttons = []
        for action in self.app.actions:
            button = widgets.Button(description=display_name(action.name))
            button.on_click(lambda e, act=action: self.run_action(act))
            action_buttons.append(button)
        layout = VBox([
            HBox(self.progression_buttons + action_buttons),
            self.status,
            HBox(dim_sliders),
            VBox(control_components),
            HBox([self.field_select, self.vector_select], layout=widgets.Layout(height='35px')),
            self.figure_display
        ])
        # Show initial value and display UI
        self.update_widgets()
        display(layout)

    def _get_status(self):
        if self._in_loop is None:  # no loop yet
            return self.app.message or ""
        elif self._in_loop is True:
            return f"Frame {self.app.steps}.  {self.app.message or ''}"
        else:
            return f"Finished {self.app.steps} steps.  {self.app.message or ''}"

    def _should_progress(self):
        return self.waiting_steps is None or self.waiting_steps > 0

    def show_field(self, field: str):
        self.field = field
        self.update_widgets()

    def update_widgets(self, plot=True, scroll_to_last=False):
        self.status.value = self._get_status()
        if not self._graphs_enabled and self.app.curve_names:
            self._graphs_enabled = True
            self.field_select.layout.visibility = 'visible'
        if self.field == 'Scalars':
            self.vector_select.layout.visibility = 'hidden'
            for sel_dim, slider in self.dim_sliders.items():
                slider.layout.visibility = 'hidden'
        else:
            shape = self.app.get_field_shape(self.field)
            self.vector_select.layout.visibility = 'visible' if 'vector' in shape else 'hidden'
            for sel_dim, slider in self.dim_sliders.items():
                if sel_dim in shape:
                    slider.layout.visibility = 'visible'
                    slider.max = shape.get_size(sel_dim) - 1
                    if scroll_to_last and sel_dim in self.app.growing_dims:
                        with ignore_events():
                            slider.value = shape.get_size(sel_dim) - 1
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
        self.figure_display.clear_output()
        with output:
            try:
                if field_name == 'Scalars':
                    plt.figure(figsize=(12, 5))
                    for name in self.app.curve_names:
                        plt.plot(*self.app.get_curve(name), label=display_name(name))
                    plt.legend()
                    plt.tight_layout()
                    show_inline_matplotlib_plots()
                else:
                    value = self.app.get_field(field_name, dim_selection)
                    if isinstance(value, SampledField):
                        try:
                            value = select_channel(value, {'🡡': None, '⬤': 'abs'}.get(self.vector_select.value, self.vector_select.value))
                            plot(value, **self.config.get('plt_args', {}))
                            show_inline_matplotlib_plots()
                        except ValueError as err:
                            self.figure_display.append_stdout(f"{err}")
                    else:
                        self.figure_display.append_stdout(f"{field_name} = {value}")
            except Exception:
                self.figure_display.append_stdout(traceback.format_exc())
        self._last_plot_update_time = time.time()

    def play(self, _):
        self.waiting_steps = None

    def auto_play(self):
        self.waiting_steps = None

    def pause(self, _):
        self.waiting_steps = 0
        self.update_widgets()

    def step(self, _):
        if self.waiting_steps is None:
            return
        else:
            self.waiting_steps += 1

    def run_action(self, action: Action):
        self.app.run_action(action.name)
        self.update_widgets()

    def interrupt(self, _):
        self._interrupted = True

    def on_loop_start(self, _):
        self._in_loop = True
        self._interrupted = False
        for button in self.progression_buttons:
            button.layout.visibility = 'visible'
        self.loop_parent = (self.kernel._parent_ident, self.kernel._parent_header)
        self.kernel.shell_handlers["execute_request"] = lambda *e: self.events.append(e)
        self.events = []
        self.update_widgets()

    def pre_step(self, app):
        self._process_kernel_events()
        while not self._should_progress():
            time.sleep(.1)
            self._process_kernel_events()
            if self._interrupted:
                raise GuiInterrupt()
        if self._interrupted:
            raise GuiInterrupt()
        if self.waiting_steps is not None:
            self.waiting_steps -= 1
        return  # runs loop iteration, then calls post_step

    def _process_kernel_events(self, n=10):
        for _ in range(n):
            self.kernel.set_parent(*self.loop_parent)  # ensure stdout still happens in the same cell
            self.kernel.do_one_iteration()
            self.kernel.set_parent(*self.loop_parent)

    def post_step(self, _):
        if self._should_progress():  # maybe skip update
            update_interval = self.config.get('update_interval')
            if update_interval is None:
                update_interval = 2.5 if 'google.colab' in sys.modules else 1.2
            elapsed = time.time() - self._last_plot_update_time
            self.update_widgets(plot=elapsed >= update_interval, scroll_to_last=True)
        else:
            self.update_widgets(scroll_to_last=True)

    def on_loop_exit(self, _):
        self._in_loop = False
        for button in self.progression_buttons:
            button.layout.visibility = 'hidden'
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


IGNORE_EVENTS = []


@contextmanager
def ignore_events():
    IGNORE_EVENTS.append(object())
    try:
        yield None
    finally:
        IGNORE_EVENTS.pop(-1)
