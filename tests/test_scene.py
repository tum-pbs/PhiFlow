from unittest import TestCase
from phi.flow import *
from os.path import isfile  # needs to be after
from phi.math import *
from phi import struct


class TestScene(TestCase):

    def test_read_write_struct(self):
        for scene in Scene.list('data'):
            scene.remove()

        state = Smoke(Domain([4, 4]))
        scene = Scene.create('data')

        scene.write(state, frame=0)
        self.assert_(isfile(scene.subpath('density_000000.npz')))
        self.assert_(isfile(scene.subpath('velocity_000000.npz')))
        loaded_state = scene.read(state, frame=0)
        self.assertIsInstance(loaded_state, Smoke)
        self.assertIsInstance(loaded_state.velocity, StaggeredGrid)
        self.assertIsInstance(loaded_state.density, CenteredGrid)
        differences = struct.compare([loaded_state.density, state.density])
        np.testing.assert_equal(loaded_state.density, state.density)
        differences = struct.compare([loaded_state.velocity.data, state.velocity.data])
        np.testing.assert_equal(loaded_state.velocity.data, state.velocity.data)

        scene.write(np.ones([1,4,4,1]) * 2, frame=1)
        self.assert_(isfile(scene.subpath('unnamed_000001.npz')))
        self.assertEqual(scene.read(None, frame=1)[0, 0, 0, 0], 2)

        scene.write([np.ones([1,4,4,1])], ['Ones'], frame=2)
        self.assert_(isfile(scene.subpath('Ones_000002.npz')))

        mystruct = [{'Two': np.ones([1,4,4,1])*2, 'Three': np.ones([1, 4, 4, 1])*3}]
        scene.write(mystruct, frame=3)
        self.assert_(isfile(scene.subpath('0_Three_000003.npz')))
        self.assert_(isfile(scene.subpath('0_Two_000003.npz')))
        loaded_struct = scene.read(mystruct, frame=3)
        self.assertIsInstance(loaded_struct, list)
        np.testing.assert_equal(mystruct[0]['Two'][0, 0, 0, 0], loaded_struct[0]['Two'][0, 0, 0, 0])

        scene.remove()
