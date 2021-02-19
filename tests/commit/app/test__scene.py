from unittest import TestCase

from phi import math
from phi import field
from phi.physics import Domain, CLOSED
from phi.app import Scene


class TestScene(TestCase):

    def test_create_remove_at_equality(self):
        scene = Scene.create('')
        self.assertEqual(scene.path[:4], "sim_")
        self.assertEqual(scene.parent_directory, "")
        self.assertNotEqual(scene.abs_path, "")
        scene_ = Scene.at(scene.parent_directory, scene.id)
        self.assertEqual(scene, scene_)
        scene__ = Scene.at(scene.path)
        self.assertEqual(scene, scene__)
        scene.remove()
        try:
            Scene.at(scene.path)
            self.fail("Scene.at() should fail with IOError if the directory does not exist.")
        except IOError:
            pass

    def test_list_scenes(self):
        scene = Scene.create('')
        scenes = Scene.list('', include_other=True)
        scenes_ = Scene.list('', include_other=False)
        self.assertEqual(scenes, scenes_)
        self.assertGreaterEqual(len(scenes), 1)
        self.assertEqual(scenes[0].parent_directory, "")
        scene.remove()

    def test_write_read(self):
        DOMAIN = Domain(x=32, y=32, boundaries=CLOSED)
        smoke = DOMAIN.scalar_grid(1)
        vel = DOMAIN.staggered_grid(2)
        # write
        scene = Scene.create('')
        scene.write({'smoke': smoke, 'vel': vel})
        # read single
        smoke_ = scene.read('smoke')
        vel_ = scene.read('vel')
        field.assert_close(smoke, smoke_)
        field.assert_close(vel, vel_)
        self.assertEqual(smoke.extrapolation, smoke_.extrapolation)
        self.assertEqual(vel.extrapolation, vel_.extrapolation)
        # read multiple
        smoke__, vel__ = scene.read(['smoke', 'vel'])
        field.assert_close(smoke, smoke__)
        field.assert_close(vel, vel__)
        scene.remove()

    def test_write_read_batch(self):
        DOMAIN = Domain(x=32, y=32, boundaries=CLOSED)
        smoke = DOMAIN.scalar_grid(1) * math.random_uniform(mybatch=2)
        vel = DOMAIN.staggered_grid(2) * math.random_uniform(mybatch=2)
        # write
        scene = Scene.create('', count=2, batch_dim='mybatch')
        scene.write({'smoke': smoke, 'vel': vel})
        # read batch
        smoke_ = scene.read('smoke')
        vel_ = scene.read('vel')
        field.assert_close(smoke, smoke_)
        field.assert_close(vel, vel_)
        scene.remove()
