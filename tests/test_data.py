from unittest import TestCase

import os
import numpy as np

from phi.data.fluidformat import Scene
from phi.data.dataset import Dataset
from phi.data.stream import SOURCE, FRAME, SCENE
from phi.data.reader import BatchReader, SourceStream


def build_test_database(path='data'):
    for scene in Scene.list(path):
        scene.remove()
    val = 1.0
    for _scene_index in range(2):
        scene = Scene.create(path)
        for t in range(4):
            scene.write_sim_frame([np.zeros([1, 4, 4, 1]) + val, np.zeros([1, 5, 5, 2])], ['Density', 'Velocity'], t)
            val += 1


class TestData(TestCase):

    def test_index_cache(self):
        build_test_database()
        reader = BatchReader(Dataset.load('data'), 'Density')
        batch = reader[0:6]
        np.testing.assert_array_equal(batch.shape, [6, 4, 4, 1])

    def test_simple_load(self):
        build_test_database()
        reader = BatchReader(Dataset.load('data'), ['Density', 'Velocity'])
        batch = reader[0]
        self.assertIsInstance(batch, (tuple, list))
        self.assertIsInstance(batch[0], np.ndarray)
        self.assertIsInstance(batch[1], np.ndarray)

    def test_iterator(self):
        build_test_database()
        reader = BatchReader(Dataset.load('data'), 'Density')
        i = 1
        for batch in reader.all_batches(batch_size=2):
            value1, value2 = batch[:, 0, 0, 0]
            self.assertEqual(value1, i)
            self.assertEqual(value2, i+1)
            i += 2

    def test_get_frames(self):
        build_test_database()
        reader = BatchReader(Dataset.load('data'), FRAME)
        frames = reader[0:6]
        np.testing.assert_array_equal(frames, [0, 1, 2, 3, 0, 1])

    def test_get_sources(self):
        build_test_database()
        reader = BatchReader(Dataset.load('data'), (SOURCE, SCENE))
        sources = reader.dataset.sources
        groundtruth0 = np.array(([sources[0]], [sources[0].scene]))
        groundtruth1 = np.array(([sources[1]], [sources[1].scene]))
        np.testing.assert_array_equal(reader[0], groundtruth0)
        np.testing.assert_array_equal(reader[3], groundtruth0)
        np.testing.assert_array_equal(reader[4], groundtruth1)

    def test_empty(self):
        build_test_database()
        reader = BatchReader(Dataset.load('data', name='empty'), ())
        self.assertEqual(reader.dataset.name, 'empty')
        self.assertEqual(len(reader), 0)
        _batch = reader[0]

    def test_write_batch(self):
        for scene in Scene.list('data'):
            scene.remove()
        scene_batch = Scene.create('data', count=2)
        self.assertEqual(scene_batch.batch_size, 2)
        self.assertEqual(len(Scene.list('data')), 2)
        batched_data = np.zeros([2, 4, 4, 1])
        unbatched_data = np.ones([1, 4, 4, 1])
        scene_batch.write(batched_data, frame=0)
        scene_batch.write(unbatched_data, frame=1)
        self.assertTrue(os.path.exists('data/sim_000000/unnamed_000000.npz'))
        self.assertTrue(os.path.exists('data/sim_000000/unnamed_000001.npz'))
        self.assertTrue(os.path.exists('data/sim_000001/unnamed_000000.npz'))
        self.assertTrue(os.path.exists('data/sim_000001/unnamed_000001.npz'))

    def test_calc(self):
        build_test_database()
        reader = BatchReader(Dataset.load('data'), ['Density', SourceStream('Density') + 1, SourceStream('Density') * SourceStream('Density')])
        for batch in reader:
            d, d_1, d_2 = batch
            np.testing.assert_equal(d+1, d_1)
            np.testing.assert_equal(d**2, d_2)
