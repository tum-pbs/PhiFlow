# coding=utf-8
import sys

from phi.flow import *
from phi.app import App


class Viewer(App):

    def __init__(self, simpath):
        App.__init__(self, name=u'*Φ-Flow* Viewer', subtitle='Play a recorded simulation',
                     framerate=EditableFloat('Framerate', 10, (1, 30), log_scale=False))
        self.value_directory = simpath
        self.view_scene = None
        self.action_load()
        self.current_frame = EditableInt('Frame', 0, (min(self.view_scene.frames), max(self.view_scene.frames)))
        self.value_looping = True
        for field_name in self.view_scene.fieldnames:
            self.add_field(field_name, lambda f=field_name: self.view_scene.read_array(f, self.current_frame))

    def step(self):
        self.current_frame += 1
        if self.current_frame >= max(self.view_scene.frames):
            self.current_frame = min(self.view_scene.frames) if self.value_looping else max(self.view_scene.frames) - 1
        self.steps = self.current_frame

    def action_load(self):
        self.view_scene = Scene.at(self.value_directory)


SCENE_PATH = sys.argv[1] if len(sys.argv) >= 2 else '~/phi/data/smoke/sim_000000'
SCENE_PATH = os.path.expanduser(SCENE_PATH)
if os.path.isdir(SCENE_PATH):
    show(Viewer(SCENE_PATH), framerate=3)
else:
    print('Scene path %s does not exist.' % SCENE_PATH)
