import os
import shutil
import time

from .._vis_base import Gui, play_async, gui_interrupt, select_channel, get_control_by_name, status_message, \
    display_name
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
            if command.strip() == "":
                time.sleep(.1)
            else:
                commands = [s.strip() for s in command.split(';')]
                for command in commands:
                    try:
                        self.process_command(command)
                    except InterruptedError:
                        return
                    except Exception as exc:
                        print(exc)

    def process_command(self, command: str):
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
        elif command == 'status':
            print(status_message(self.app, self.play_status))
        elif command == 'exit':
            if self.play_status:
                self.play_status.pause()
            if self.app.can_progress:
                if self.app.can_progress:
                    self.app.pre_step.append(gui_interrupt)
                    self.app.progress()
            raise InterruptedError()
        elif command == 'kill':
            os._exit(0)
        elif command == 'show':
            self.draw()
        elif command.startswith('show '):
            fields = command[len('show '):].split(',')
            fields = [f.strip() for f in fields]
            self.draw(fields)
        elif command == 'controls':
            if self.app.controls:
                print("Available controls:\n-------------------------------------------")
                for control in self.app.controls:
                    value = self.app.get_control_value(control.name)
                    print(f"{control.name}: {control.control_type.__name__} = {value}  \t(initial value: {control.initial}, \trange {control.value_range})")
                print("-------------------------------------------")
                print("You can change a control value by typing '<control_name> = <value>'")
            else:
                print("No controls available. Create controls in your Python script using '<control_name> = control(value)'.")
        elif '=' in command:
            parts = command.split('=')
            assert len(parts) == 2, "Invalid command syntax. Use '<control_name> = <value>'. Type 'controls' for a list of available controls."
            parts = [p.strip() for p in parts]
            name, value = parts
            control = get_control_by_name(self.app, name)
            value = control.control_type(value)  # raises ValueError
            self.app.set_control_value(name, value)
        elif command == 'help':
            print("General Commands:  \t\tstatus, controls, actions, help\n"
                  "Plotting:  \t\t\t\tshow, show <comma-separated fields>, show <field>.<component>\n"
                  "Control Execution:  \tplay, play <frames>, pause, step, <control> = <value>, exit, kill\n"
                  "See https://tum-pbs.github.io/PhiFlow/ConsoleUI.html for a detailed description of available commands.")
        elif command == 'actions':
            print("Available actions:\n")
            for action in self.app.actions:
                print(f"{display_name(action.name)}  \t(type '{action.name}' to run)")
                if action.description:
                    print(f"{action.description}")
                print()
        elif command in [a.name for a in self.app.actions]:
            self.app.run_action(command)
            print(f"Completed '{display_name(command)}'")
        else:
            print(f"Command {command} not recognized.")

    def draw(self, field_names: list = None):
        if field_names is None:
            if len(self.app.field_names) == 0:
                print("Nothing to show.")
                return
            field_names = self.app.field_names[:2] if len(self.app.field_names) > 2 else self.app.field_names
            channel_sel = [None] * len(field_names)
        else:
            channel_sel = [n[n.index('.')+1:] if '.' in n else None for n in field_names]
            field_names = [n[:n.index('.')] if '.' in n else n for n in field_names]
        values = []
        for n in field_names:
            try:
                values.append(self.app.get_field(n, {}))
            except KeyError:
                print(f"The field {n} does not exist. Available fields are {self.app.field_names}")
                return
        cols, rows = shutil.get_terminal_size(fallback=(80, 14))
        plt_width = cols // len(values)
        plt_height = rows - 1
        lines = [""] * plt_height
        for name, v, ch_sel in zip(field_names, values, channel_sel):
            v = select_channel(v, ch_sel)
            if v.vector.exists:
                plt_lines = quiver(v, plt_width, plt_height, name, threshold=0.1, basic_chars=True)
            else:
                plt_lines = heatmap(v, plt_width, plt_height, name)
            lines = [l+p for l, p in zip(lines, plt_lines)]
        print("\n".join(lines))

    def auto_play(self):
        framerate = self.config.get('framerate', None)
        self.play_status = play_async(self.app, framerate=framerate)
