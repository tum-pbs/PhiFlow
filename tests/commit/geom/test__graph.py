from unittest import TestCase

from phi import math
from phi.geom import graph
from phi.math import channel
from phiml.math import wrap, instance, batch, stack, vec


class TestGraph(TestCase):

    def test_slice(self):
        points = wrap([(0, 0), (1.2, 0), (2, 0), (2, 1), (.8, 1), (0, 1)], instance('points'), channel(vector='x,y'))
        dense_edges = math.vec_length(math.pairwise_distances(points, 1.5))
        dense_graph = graph(points, dense_edges, {})
        dense_subgraph = dense_graph[{'points': slice(0, 2)}]
        self.assertEqual(4, math.stored_indices(dense_subgraph.edges).entries.size)
        self.assertEqual((2, 2), dense_subgraph.edges.shape.sizes)
        # --- Sparse ---
        sparse_edges = math.vec_length(math.pairwise_distances(points, 1.5, format='csr'))
        sparse_graph = graph(points, sparse_edges, {})
        sparse_subgraph = sparse_graph[{'points': slice(0, 2)}]
        self.assertEqual(4, math.stored_indices(sparse_subgraph.edges).entries.size)
        self.assertEqual((2, 2), sparse_subgraph.edges.shape.sizes)

    def test_batch_slice(self):
        x = vec(x=[0, 1, 2], y=0)
        deltas = math.pairwise_differences(x, max_distance=.2, format='coo')
        distances = math.vec_length(deltas)
        g = graph(x, distances, {})
        stacked = stack([g, g], batch('b'))
        g0 = stacked.b[0]
        self.assertNotIn('b', g0.nodes.shape)
        self.assertNotIn('b', g0.edges.shape)
