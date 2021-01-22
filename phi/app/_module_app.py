import inspect
import os
from threading import Thread, Lock, Event

from phi.field import Field
from phi.math import Tensor

from .app import App
from .display import show


class ModuleViewer(App):
    def __init__(self,
                 name=None,
                 subtitle='',
                 fields=None,
                 stride=None,
                 base_dir='~/phi/data/',
                 summary=None,
                 custom_properties=None,
                 target_scene=None,
                 objects_to_save=None,
                 framerate=None,
                 dt=1.0,
                 **show_config):
        """
        ModuleViewer shows the contents of the calling Python script in the GUI.

        Name and subtitle of the App may be specified in the module docstring (string before imports).
        The first line is interpreted as the name, the rest as the subtitle.
        If not specified, a generic name and description is chosen.

        Use ModuleViewer.range() as a for-loop iteratable to control the loop execution from within the GUI.

        :param fields: (Optional) names of global variables to be displayed.
          If not provided, searches all global variables for Field or Tensor values.
          All fields must exist as global variables before the ModuleViewer is instantiated.
        """
        module = inspect.getmodule(inspect.stack()[1].frame)
        doc = module.__doc__
        if doc is None:
            name = os.path.basename(module.__file__)[:-3]
            subtitle = module.__doc__ or module.__file__
        else:
            end_of_line = doc.index('\n')
            name = doc[:end_of_line].strip()
            subtitle = doc[end_of_line:].strip() or None
        App.__init__(self, name, subtitle, fields=None, stride=stride, base_dir=base_dir, summary=summary, custom_properties=custom_properties, target_scene=target_scene, objects_to_save=objects_to_save, framerate=framerate, dt=dt)
        if fields is None:
            for name in dir(module):
                val = getattr(module, name)
                if isinstance(val, Field) or (isinstance(val, Tensor) and val.shape.spatial_rank > 0):
                    self.add_field(name, lambda name=name: getattr(module, name))
        else:
            for name in fields:
                self.add_field(name, lambda name=name: getattr(module, name))
        self.step_exec_event = Event()
        self.step_finished_event = Event()
        Thread(target=lambda: show(self, **show_config)).start()

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

