from unittest import TestCase

from phi import math
from phi.geom import Graph
from phi.math import channel
from phiml.math import wrap, instance


class TestBox(TestCase):

    def test_slice(self):
        points = wrap([(0, 0), (1.2, 0), (2, 0), (2, 1), (.8, 1), (0, 1)], instance('points'), channel(vector='x,y'))
        dense_edges = math.vec_length(math.pairwise_distances(points, 1.5))
        dense_graph = Graph(points, dense_edges, {})
        dense_subgraph = dense_graph[{'points': slice(0, 2)}]
        self.assertEqual(4, math.stored_indices(dense_subgraph.edges).entries.size)
        self.assertEqual((2, 2), dense_subgraph.edges.shape.sizes)
        # --- Sparse ---
        sparse_edges = math.vec_length(math.pairwise_distances(points, 1.5, format='csr'))
        sparse_graph = Graph(points, sparse_edges, {})
        sparse_subgraph = sparse_graph[{'points': slice(0, 2)}]
        self.assertEqual(4, math.stored_indices(sparse_subgraph.edges).entries.size)
        self.assertEqual((2, 2), sparse_subgraph.edges.shape.sizes)

