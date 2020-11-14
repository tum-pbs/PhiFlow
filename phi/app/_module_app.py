import inspect
import os
from threading import Thread, Lock, Event

from phi.field import Field
from phi.math import Tensor

from .app import App
from .display import show


class ModuleViewer(App):
    """
    ModuleViewer shows the contents of a module in the GUI.
    """

    def __init__(self, fields=None, **kwargs):
        module = inspect.getmodule(inspect.stack()[1].frame)
        App.__init__(self, os.path.basename(module.__file__)[:-3], module.__file__, **kwargs)
        if fields is None:
            for name in dir(module):
                val = getattr(module, name)
                if not callable(val) and issubclass(type(val), (Field, Tensor)):
                    self.add_field(name, lambda name=name: getattr(module, name))
        else:
            for name in fields:
                self.add_field(name, lambda name=name: getattr(module, name))
        self.step_exec_event = Event()
        self.step_finished_event = Event()
        Thread(target=lambda: show(self)).start()

    def range(self, *args):
        if len(args) == 0:
            while True:
                self.step_exec_event.wait()
                yield self.steps
                self.step_exec_event.clear()
                self.step_finished_event.set()
        elif len(args) == 1:
            for _ in range(args[0]):
                self.step_exec_event.wait()
                yield self.steps
                self.step_exec_event.clear()
                self.step_finished_event.set()
        elif len(args) == 2:
            for i in range(args[0], args[1]):
                self.step_exec_event.wait()
                self.steps = i
                yield self.steps
                self.step_exec_event.clear()
                self.step_finished_event.set()
        else:
            raise ValueError(args)

    def step(self):
        self.step_finished_event.clear()
        self.step_exec_event.set()
        self.step_finished_event.wait()

