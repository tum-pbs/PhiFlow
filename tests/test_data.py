from unittest import TestCase
from phi.flow import *


def build_test_database(path='data'):
    for scene in Scene.list(path):
        scene.remove()
    val = 1.0
    for sceneindex in range(2):
        scene = Scene.create(path)
        for t in range(4):
            scene.write_sim_frame([np.zeros([1,4,4,1])+val, np.zeros([1,5,5,2])], ['Density', 'Velocity'], t)
            val += 1


class TestData(TestCase):

    def test_index_cache(self):
        build_test_database()
        reader = BatchReader(Dataset.load('data'), 'Density')
        batch = reader[0:6]
        np.testing.assert_array_equal(batch.shape, [6,4,4,1])

    def test_struct_load(self):
        build_test_database()
        reader = BatchReader(Dataset.load('data'), ('Density', StaggeredGrid('Velocity')))
        batch = reader[0]
        self.assertIsInstance(batch, (tuple, list))
        self.assertIsInstance(batch[1], StaggeredGrid)

    def test_iterator(self):
        build_test_database()
        reader = BatchReader(Dataset.load('data'), 'Density')
        i = 1
        for batch in reader.all_batches(batch_size=2):
            value1, value2 = batch[:,0,0,0]
            self.assertEqual(value1, i)
            self.assertEqual(value2, i+1)
            i += 2

    def test_modify_database(self):
        build_test_database()
        dataset = Dataset('data')
        reader = BatchReader(dataset, FRAME)
        self.assertEqual(len(reader), 0)
        dataset += Dataset.load('data')
        self.assertEqual(len(reader), 8)
        a = reader[0]

    def test_get_frames(self):
        build_test_database()
        reader = BatchReader(Dataset.load('data'), FRAME)
        frames = reader[0:6]
        np.testing.assert_array_equal(frames, [0,1,2,3,0,1])

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
        batch = reader[0]

    # def test_lazy_size_eval(self):