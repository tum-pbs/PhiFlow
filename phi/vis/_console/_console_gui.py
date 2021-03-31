import shutil
import time

from phi.field import StaggeredGrid
from .._vis_base import Gui, play_async, gui_interrupt
from ._console_plot import heatmap, quiver


class ConsoleGui(Gui):

    def __init__(self):
        Gui.__init__(self, asynchronous=True)
        self.play_status = None

        # def setup(self):
    #     vis = self.vis
    #     self.vis.logger.removeHandler(self.vis.console_handler)
    #     terminal_size = shutil.get_terminal_size(fallback=(80, 20))
    #
    #     class CustomHandler(Handler):
    #
    #         def emit(self, record: LogRecord) -> None:
    #             pass
    #
    #         def handle(self, record: LogRecord) -> None:
    #             line = vis.message + " " * (max(1, terminal_size[0]-len(vis.message)-1))
    #             print(line, end="\r")
    #
    #         def createLock(self) -> None:
    #             pass
    #
    #     self.vis.logger.addHandler(CustomHandler())

    def show(self, caller_is_main: bool):
        print(self.app.name)
        print(self.app.description)
        print()
        print("PhiFlow console interface active. Type 'help' for a list of available commands.")
        while True:
            print("PhiFlow>", end="")
            command = input()
            if command == 'step':
                self.app.progress()
            elif command.startswith('play'):
                if self.play_status:
                    print("Wait for current step to finish." if self.play_status.paused else "Already playing, command ignored.")
                else:
                    if command.strip() == 'play':
                        frames = None
                    else:
                        frames = int(command[len('play '):].strip())
                    self.play_status = play_async(self.app, frames)
            elif command == 'pause':
                if self.play_status:
                    self.play_status.pause()
            elif command == 'exit':
                if self.play_status:
                    self.play_status.pause()
                if self.app.can_progress:
                    self.app.pre_step.append(gui_interrupt)
                    self.app.progress()
                return  # exit this thread
            elif command == 'show':
                self.draw()
            elif command.startswith('show '):
                fields = command[len('show '):].split(',')
                fields = [f.strip() for f in fields]
                self.draw(fields)
            elif command == 'help':
                print("Commands: help, step, play, pause, show, show <comma-separated fieldnames>")
            else:
                print(f"Command {command} not recognized.")
                time.sleep(.1)

    def draw(self, field_names: list = None):
        if field_names is None:
            if len(self.app.field_names) == 0:
                print("Nothing to show.")
                return
            field_names = self.app.field_names[:2] if len(self.app.field_names) > 2 else self.app.field_names
        values = []
        for n in field_names:
            try:
                values.append(self.app.get_field(n))
            except KeyError:
                print(f"The field {n} does not exist. Available fields are {self.app.field_names}")
                return
        cols, rows = shutil.get_terminal_size(fallback=(80, 14))
        plt_width = cols // len(values)
        plt_height = rows - 1
        lines = [""] * plt_height
        for name, v in zip(field_names, values):
            if isinstance(v, StaggeredGrid):
                v = v.at_centers()
            if v.vector.exists:
                plt_lines = quiver(v, plt_width, plt_height, name, threshold=0.1, basic_chars=True)
            else:
                plt_lines = heatmap(v, plt_width, plt_height, name)
            lines = [l+p for l, p in zip(lines, plt_lines)]
        print("\n".join(lines))
