# coding=utf-8
from phi.data.fluidformat import *
from phi.model import FieldSequenceModel
import sys


class Viewer(FieldSequenceModel):
    def __init__(self, simpath):
        FieldSequenceModel.__init__(self, name=u'*Φ-Flow* Viewer', subtitle='Play a recorded simulation')
        self.value_directory = simpath
        self.action_rewind()

    def update(self):
        if not self.indices:
            self.info('No frames present.')
            self.steps = 0
        else:
            self.steps = self.indices[self.timeindex % len(self.indices)]
            self.info('Loading frame %d...' % self.steps)
            self.fieldvalues = read_sim_frame(self.value_directory, self.fieldnames, self.steps)
            self.info('')

    def step(self):
        self.timeindex += 1
        self.update()

    def action_rewind(self):
        self.timeindex = 0
        self.action_refresh()
        if self.indices:
            self.steps = self.indices[0]
        else:
            self.steps = 0

    def action_refresh(self):
        self.view_scene = Scene.at(self.value_directory)
        self.indices = self.view_scene.get_frames(mode='union')
        for fieldname in self.view_scene.fieldnames:
            def getfield(fieldname=fieldname):
                return self.view_scene.read_array(fieldname, self.steps)
            self.add_field(fieldname, getfield)
        self.update()


scene_path = sys.argv[1] if len(sys.argv) >= 2 else '~/phi/data/simpleplume/sim_000000'
scene_path = os.path.expanduser(scene_path)
if os.path.isdir(scene_path):
    app = Viewer(scene_path).show(framerate=3, production=__name__!='__main__')
else:
    import logging
    logging.fatal('Scene path %s does not exist.' % scene_path)
