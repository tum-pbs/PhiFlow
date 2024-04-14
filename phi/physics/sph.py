from typing import Dict, Tuple, Any, Union

from phi import math
from phi.field import Field
from phi.math import Tensor, pairwise_distances, vec_length, Shape, non_channel, dual, where, PI
from phi.geom import Geometry, Graph
from phiml.math import channel, stack

_DEFAULT_DESIRED_NEIGHBORS = {
    'quintic-spline': 34,
    'wendland-c2': 22,
}


def neighbor_graph(nodes: Geometry,
                   kernel: str,
                   boundary: Dict[str, Dict[str, slice]] = None,
                   desired_neighbors: float = None,
                   kernel_derivative=True,
                   format='csr',
                   search_method='auto') -> Graph:
    """
    Build a `phi.geom.Graph` based on proximity of `nodes`.

    Args:
        nodes: Particles including obstacle particles as `Geometry` collection.
        kernel: Kernel function to evaluate.
        boundary: Marks ranges of nodes as boundary particles, see `phi.geom.Graph`.
        desired_neighbors: Target average number of neighbors per particle. This determines the support radius (cutoff) used.
        kernel_derivative: Whether to evaluate the kernel derivative
        format: Sparse format in which store neighborhood information. Allowed strings are `'dense', `'csr'`, `'coo'`, `'csc'`.
        search_method: Neighborhood search method, see `phi.math.pairwise_differences`.

    Returns:
        `phi.geom.Graph` with edge values storing the kernel values, i.e. the interaction strength between particles.
    """
    assert isinstance(nodes, Geometry), f"nodes must be a Geometry instance but got {type(nodes)}"
    boundary = {} if boundary is None else boundary
    desired_neighbors = _DEFAULT_DESIRED_NEIGHBORS[kernel] if desired_neighbors is None else desired_neighbors
    # --- neighbor search ---
    support = _optimal_support_radius(nodes.volume, desired_neighbors, nodes.spatial_rank)
    # --- evaluate kernel and derivatives ---
    deltas = math.pairwise_differences(nodes.center, max_distance=support, format=format, method=search_method)
    distances = math.vec_length(deltas, eps=1e-5)
    q = distances / support
    edges = {'kernel': evaluate_kernel(q, nodes.spatial_rank, kernel) / support ** nodes.spatial_rank}
    if kernel_derivative:
        kernel_derivatives = evaluate_kernel(q, nodes.spatial_rank, kernel, derivative=1) * support ** (nodes.spatial_rank + 1) * deltas
        for dim, val in dict(**kernel_derivatives.vector).items():
            edges[dim] = val
    edges = stack(edges, channel('vector'))
    return Graph(nodes, edges, boundary, deltas=deltas, distances=distances, bounding_distance=support)


def _optimal_support_radius(volume: Tensor, desired_neighbors: float, spatial_rank: int) -> Tensor:  # volumeToSupport
    """
    Calculates the optimal kernel support radius so that on average `desired_neighbors` neighbors lie within reach of each particle.

    Args:
        volume: Average particle volume.
        desired_neighbors: Desired average number of neighbor particles to lie within the support.
        spatial_rank: Spatial rank of the simulation.

    Returns:
        Support radius, i.e. neighbor search cutoff.
    """
    if spatial_rank == 1:
        return desired_neighbors * .5 * volume  # N(h) = 2 h / v
    elif spatial_rank == 2:
        return math.sqrt(desired_neighbors / math.PI * volume)  # N(h) = 𝜋 h² / v
    else:
        return (desired_neighbors / math.PI * .75 * volume) ** (1/3)  # N(h) = 4/3 𝜋 h³ / v


def evaluate_kernel(q: Tensor, spatial_rank: int, kernel='wendland-c2', derivative=0):
    """
    Compute the SPH kernel value at a normalized scalar distance `q` or a derivative of the kernel function.

    Args:
        q: Normalized distance `phi.math.Tensor`. All values must lie between 0 and 1.
        spatial_rank: Dimensionality of the simulation.
        kernel: Which kernel to use, one of `'wendland-c2'`, `'quintic-spline'`.
        derivative: Derivative order, `0` for kernel function, `1` for gradient.

    Returns:
        `phi.math.Tensor`
    """
    if kernel == 'quintic-spline':  # cutoff at q = 3 (d=3h)
        alpha_d = {1: 1 / 120, 2: 7 / 478 / PI, 3: 1 / 120 / PI}[spatial_rank]
        if not derivative:
            w1 = (3 - q) ** 5 - 6 * (2 - q) ** 5 + 15 * (1 - q) ** 5
            w2 = (3 - q) ** 5 - 6 * (2 - q) ** 5
            w3 = (3 - q) ** 5
            fac = alpha_d
        elif derivative == 1:
            w1 = (3 - q) ** 4 - 6 * (2 - q) ** 4 + 15 * (1 - q) ** 4
            w2 = (3 - q) ** 4 - 6 * (2 - q) ** 4
            w3 = (3 - q) ** 4
            fac = -5 * alpha_d
        else:
            raise NotImplementedError(f"{derivative}th derivative of {kernel} is not supported")
        return fac * where(q > 2, w3, where(q > 1, w2, w1))
    elif kernel == 'wendland-c2':  # cutoff at q=2 (d=2h)
        alpha_d = {2: 7 / 4 / PI, 3: 21 / 16 / PI}[spatial_rank]
        if not derivative:
            w = (1 - 0.5 * q) ** 4 * (2 * q + 1)
            fac = alpha_d
        elif derivative == 1:
            w = (1 - 0.5 * q) ** 3 * q
            fac = -5 * alpha_d
        else:
            raise NotImplementedError(f"{derivative}th derivative of {kernel} is not supported")
        return fac * w
    else:
        raise ValueError(kernel)


def density(graph: Graph):
    return math.sum(graph.edges['kernel'], dual)


# def diffusion(u: Field):
#     kernel_grad = u.graph.edges.vector[1:]
#     du = math.pairwise_differences(u.values, format=u.graph.edges)
#     dr = u.graph.deltas
#     p = du.vector @ dr.vector / math.vec_squared(dr)
#     term = p * kernel_grad
#     return (u.graph.bounding_distance * alpha * c0 * restDensity / rhoi) * math.sum(term, dual)
