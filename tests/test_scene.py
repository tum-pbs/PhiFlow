from unittest import TestCase
from phi.flow import *
from os.path import isfile


class TestScene(TestCase):

    def test_write_struct(self):
        for scene in Scene.list('data'): scene.remove()

        state = Smoke(Domain([4,4])).initial_state()
        scene = Scene.create('data')
        scene.write(state, frame=0)
        scene.write(np.ones([1,4,4,1]) * 2, frame=1)
        scene.write([np.ones([1,4,4,1])], ['Ones'], frame=2)
        scene.write([{'Two': np.ones([1,4,4,1])*2, 'Three': np.ones([1,4,4,1])*3}], frame=3)

        self.assert_(isfile(scene.subpath('density_000000.npz')))
        self.assert_(isfile(scene.subpath('Ones_000002.npz')))
        self.assert_(isfile(scene.subpath('0_Three_000003.npz')))
        self.assert_(isfile(scene.subpath('0_Two_000003.npz')))
        self.assert_(isfile(scene.subpath('unnamed_000001.npz')))
        self.assert_(isfile(scene.subpath('velocity_staggered_000000.npz')))