from unittest import TestCase
from phi.flow import *


class TestScene(TestCase):

    def test_write_struct(self):
        state = Smoke(Domain([4,4])).initial_state()
        scene = Scene.create('data')
        scene.write(state, frame=0)
        scene.write(np.ones([1,4,4,1]) * 2, frame=1)
        scene.write([np.ones([1,4,4,1])], ['Ones'], frame=2)
