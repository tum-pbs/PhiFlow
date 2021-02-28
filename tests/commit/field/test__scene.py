from unittest import TestCase

from os.path import dirname, abspath, join, basename

from phi import math
from phi import field
from phi.field import Scene
from phi.physics import Domain, CLOSED


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
        scene_ = Scene.list(DIR, dim='batch')
        self.assertGreaterEqual(scene_.shape.volume, 2)
        scene.remove()

    def test_write_read(self):
        DOMAIN = Domain(x=32, y=32, boundaries=CLOSED)
        smoke = DOMAIN.scalar_grid(1)
        vel = DOMAIN.staggered_grid(2)
        # write
        scene = Scene.create(DIR)
        scene.write(smoke=smoke, vel=vel)
        self.assertEqual(1, len(scene.frames))
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
        scene.remove()

    def test_write_read_batch_matching(self):
        DOMAIN = Domain(x=32, y=32, boundaries=CLOSED)
        smoke = DOMAIN.scalar_grid(1) * math.random_uniform(count=2)
        vel = DOMAIN.staggered_grid(2) * math.random_uniform(count=2)
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
        DOMAIN = Domain(x=32, y=32, boundaries=CLOSED)
        smoke = DOMAIN.scalar_grid(1) * math.random_uniform(count=2, config=3)
        vel = DOMAIN.staggered_grid(2) * math.random_uniform(count=2, vel=2)
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
        DOMAIN = Domain(x=32, y=32, boundaries=CLOSED)
        smoke = DOMAIN.scalar_grid(1) * math.random_uniform(count=2)
        vel = DOMAIN.staggered_grid(2) * math.random_uniform(count=2)
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
