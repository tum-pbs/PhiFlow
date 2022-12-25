from unittest import TestCase

from os.path import dirname, abspath, join, basename

import phi
from phi import math
from phi import field
from phi.field import Scene, CenteredGrid, StaggeredGrid
from phi.math import batch, extrapolation, wrap, stack, vec

DIR = join(dirname(dirname(dirname(dirname(abspath(__file__))))), 'test_data')


class TestScene(TestCase):

    def test_create_remove_at_equality_single(self):
        scene = Scene.create(DIR)
        self.assertEqual(basename(scene.path)[:4], "sim_")
        self.assertEqual(1, scene.shape.volume)
        scene_ = Scene.at(scene.path)
        self.assertEqual(scene, scene_)
        repr(scene)
        scene.remove()
        try:
            Scene.at(scene.path)
            self.fail("Scene.at() should fail with IOError if the directory does not exist.")
        except IOError:
            pass

    def test_properties(self):
        scene = Scene.create(DIR)
        self.assertEqual(0, len(scene.properties))
        scene.put_property('a', 1)
        scene.put_properties({'b': 2, 'c': 3}, d=4)
        scene = Scene.at(scene.path)
        self.assertEqual(4, len(scene.properties))
        scene.remove()

    def test_batched_properties(self):
        scenes = Scene.create(DIR, batch(scenes=2))
        batched = wrap([0, 1], batch(scenes=2))
        scenes.put_properties(batched=batched,
                              non_batched=-1.,
                              batched_tensor=batched * vec(x=2, y=3),
                              non_batched_tensor=vec(x=2, y=3))
        s0, s1 = scenes.scenes
        self.assertIsNone(s0._properties)
        self.assertEqual(0, s0.properties['batched'])
        self.assertEqual(1, s1.properties['batched'])
        self.assertEqual(-1, s0.properties['non_batched'])
        self.assertEqual(-1, s1.properties['non_batched'])
        math.assert_close((0, 0), s0.properties['batched_tensor'])
        math.assert_close((2, 3), s1.properties['batched_tensor'])
        math.assert_close((2, 3), s0.properties['non_batched_tensor'])
        math.assert_close((2, 3), s1.properties['non_batched_tensor'])
        scenes = stack([s0, s1], scenes.shape)
        math.assert_close(batched, scenes.properties['batched'])
        math.assert_close(-1, scenes.properties['non_batched'])
        math.assert_close(batched * vec(x=2, y=3), scenes.properties['batched_tensor'])
        math.assert_close(vec(x=2, y=3), scenes.properties['non_batched_tensor'])
        scenes.remove()

    def test_create_remove_at_equality_batch(self):
        scene = Scene.create(DIR, batch=2, config=3)
        self.assertEqual(6, scene.shape.volume)
        self.assertEqual(('batch', 'config'), scene.shape.names)
        scene_ = Scene.at(scene.paths)
        self.assertEqual(scene, scene_)
        repr(scene)
        scene.remove()
        try:
            Scene.at(scene.paths)
            self.fail("Scene.at() should fail with IOError if the directory does not exist.")
        except IOError:
            pass

    def test_list_scenes(self):
        scene = Scene.create(DIR, count=2)
        scenes = Scene.list(DIR, include_other=True)
        scenes_ = Scene.list(DIR, include_other=False)
        self.assertEqual(scenes, scenes_)
        self.assertGreaterEqual(len(scenes), 2)
        scene_ = Scene.list(DIR, dim=batch('batch'))
        self.assertGreaterEqual(scene_.shape.volume, 2)
        scene.remove()

    def test_write_read(self):
        smoke = CenteredGrid(1, extrapolation.BOUNDARY, x=32, y=32)
        vel = StaggeredGrid(2, 0, x=32, y=32)
        # write
        scene = Scene.create(DIR)
        scene.write(smoke=smoke, vel=vel)
        self.assertEqual(1, len(scene.frames))
        self.assertEqual(1, len(scene.complete_frames))
        self.assertEqual(2, len(scene.fieldnames))
        # read single
        smoke_ = scene.read('smoke')
        vel_ = scene.read('vel')
        field.assert_close(smoke, smoke_)
        field.assert_close(vel, vel_)
        self.assertEqual(smoke.extrapolation, smoke_.extrapolation)
        self.assertEqual(vel.extrapolation, vel_.extrapolation)
        # read multiple
        smoke__, vel__ = scene.read(['smoke', 'vel'])  # deprecated
        field.assert_close(smoke, smoke__)
        field.assert_close(vel, vel__)
        smoke__, vel__ = scene.read('smoke', 'vel')
        field.assert_close(smoke, smoke__)
        field.assert_close(vel, vel__)
        # read without Scene
        smoke_ = phi.field.read(join(scene.path, "smoke_000000.npz"))
        field.assert_close(smoke, smoke_)
        scene.remove()

    def test_write_read_batch_matching(self):
        smoke = CenteredGrid(1, extrapolation.BOUNDARY, x=32, y=32) * math.random_uniform(batch(count=2))
        vel = StaggeredGrid(2, 0, x=32, y=32) * math.random_uniform(batch(count=2))
        # write
        scene = Scene.create(DIR, count=2)
        scene.write({'smoke': smoke, 'vel': vel})
        # read batch
        smoke_ = scene.read('smoke')
        vel_ = scene.read('vel')
        field.assert_close(smoke, smoke_)
        field.assert_close(vel, vel_)
        scene.remove()

    def test_write_read_batch_batched_files(self):
        smoke = CenteredGrid(1, extrapolation.BOUNDARY, x=32, y=32) * math.random_uniform(batch(count=2, config=3))
        vel = StaggeredGrid(2, 0, x=32, y=32) * math.random_uniform(batch(count=2, vel=2))
        # write
        scene = Scene.create(DIR, count=2)
        scene.write({'smoke': smoke, 'vel': vel})
        # read batch
        smoke_ = scene.read('smoke')
        vel_ = scene.read('vel')
        field.assert_close(smoke, smoke_)
        field.assert_close(vel, vel_)
        scene.remove()

    def test_write_read_batch_duplicate(self):
        smoke = CenteredGrid(1, extrapolation.BOUNDARY, x=32, y=32) * math.random_uniform(batch(count=2))
        vel = StaggeredGrid(2, 0, x=32, y=32) * math.random_uniform(batch(count=2))
        # write
        scene = Scene.create(DIR,  more=2)
        scene.write({'smoke': smoke, 'vel': vel})
        # read batch
        smoke_ = scene.read('smoke')
        vel_ = scene.read('vel')
        self.assertEqual(4, smoke_.shape.batch.volume)
        self.assertEqual(4, vel_.shape.batch.volume)
        field.assert_close(smoke, smoke_)
        field.assert_close(vel, vel_)
        scene.remove()

    def test_read_legacy_centered_grid(self):
        path = join(dirname(abspath(__file__)))
        density = field.read(join(path, 'dens_001000.npz'))
        assert isinstance(density, CenteredGrid)
        velocity = field.read(join(path, 'velo_001000.npz'))
        assert isinstance(velocity, StaggeredGrid)
