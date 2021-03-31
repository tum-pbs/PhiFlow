import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from phi import math
from .._vis_base import Gui, VisModel
from ...field import StaggeredGrid


class MatplotlibGui(Gui):

    def __init__(self):
        Gui.__init__(self)
        self.axes = None
        self.fig = None

    def setup(self, app: VisModel):
        Gui.setup(self, app)
        self.fig_count = min(self.config.get('figs', 2), len(self.app.fieldnames))

        fig, axes = plt.subplots(1, self.fig_count, figsize=(10, 7))
        self.fig = fig
        if isinstance(axes, np.ndarray):
            self.axes = tuple(axes.flatten())
        else:
            self.axes = (axes,)
        fig.canvas.set_window_title(self.app.name)
        plt.subplots_adjust(top=0.8, bottom=0.2)

        y = 0.02
        button_width = 0.1
        button_height = 0.06
        self.start_button = Button(plt.axes([0.05, y, button_width, button_height]), 'Play')
        self.start_button.on_clicked(self.play)
        self.pause_button = Button(plt.axes([0.05 + button_width + 0.01, y, button_width, button_height]), 'Pause')
        self.pause_button.on_clicked(self.pause)
        self.stop_button = Button(plt.axes([0.05 + 2 * (button_width + 0.01), y, button_width, button_height]), 'Step')
        self.stop_button.on_clicked(self.step)

    def show(self, caller_is_main):
        while True:
            for i in range(self.fig_count):
                self.axes[i].clear()
                v = self.app.get_field(self.shown_fields[i])
                if isinstance(v, StaggeredGrid):
                    v = v.at_centers()
                v = math.vec_squared(v.values).numpy('y, x')
                self.axes[i].imshow(v, origin='lower')
                self.axes[i].set_title(self.shown_fields[i])
            plt.draw()
            plt.pause(0.01)

    def step(self, _event=None):
        self.app.run_step()

    def pause(self, _event=None):
        self.app.pause()
