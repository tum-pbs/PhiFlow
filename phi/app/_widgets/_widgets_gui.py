import numpy as np
from ipywidgets.widgets.interaction import show_inline_matplotlib_plots
from ipywidgets import interactive, HBox, VBox

from .._display import AppDisplay

from matplotlib import pyplot as plt

from IPython.display import display
import ipywidgets as widgets

from phi.field.plt import plot
from .._display_util import ordered_field_names


class WidgetsGui(AppDisplay):

    def show(self, caller_is_main: bool) -> bool:
        figure_display = widgets.Output()
        play_button = widgets.Button(description="Play")
        # button.on_click(lambda b: figure_display.append_stdout("Button clicked."))
        pause_button = widgets.Button(description="Pause")
        # clear_button.on_click(lambda b: figure_display.clear_output())
        step_button = widgets.Button(description="Step")

        if 'style' in self.config:
            plt.style.use(self.config['style'])
        with figure_display:
            shown_fields = ordered_field_names(self.app, self.config.get('display'))
            # plot(self.app.get_field(shown_fields[0]))
            plt.plot([1,4,2,5])
            plt.title(shown_fields[0])
            plt.show()

        # def f(m, b):
        #     plt.figure(2)
        #     x = np.linspace(-10, 10, num=1000)
        #     plt.plot(x, m * x + b)
        #     plt.ylim(-5, 5)
        #     plt.show()
        #
        # interactive_plot = interactive(f, m=(-2.0, 2.0), b=(-3, 3, 0.5))
        # output = interactive_plot.children[-1]
        # output.layout.height = '350px'
        # display(interactive_plot)

        buttons = HBox([play_button, pause_button, step_button])
        layout = VBox([figure_display, buttons])
        display(layout)

        # def replot():
        #     show_inline_matplotlib_plots()
        #     with output:
        #         # clear_output(wait=True)
        #         plt.plot([1, 3, 2, 4])
        #         plt.show()
        #         show_inline_matplotlib_plots()

        #
        # with output:
        #     print("Initial output.")
        return True
