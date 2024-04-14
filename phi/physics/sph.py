"""
Tools for running Smoothed Particle Hydrodynamics (SPH) simulations.

1. Create particles as a `Geometry` collections.
2. Use `neighbor_graph` to find neighbor particles and compute kernel weights.
3. Use custom function or built-in physics operations to integrate the dynamics.
"""
from typing import Dict, Tuple, Any, Union, Sequence

from phi import math
from phi.field import Field
from phi.math import Tensor, pairwise_distances, vec_length, Shape, non_channel, dual, where, PI
from phi.geom import Geometry, Graph
from phiml.math import channel, stack, vec, concat, expand

_DEFAULT_DESIRED_NEIGHBORS = {
    'quintic-spline': 34,
    'wendland-c2': 22,
    'poly6': 30,
}


def neighbor_graph(nodes: Geometry,
                   kernel: str,
                   boundary: Dict[str, Dict[str, slice]] = None,
                   desired_neighbors: float = None,
                   compute: str = 'kernel,grad',
                   format='sparse',
                   search_method='auto') -> Graph:
    """
    Build a `phi.geom.Graph` based on proximity of `nodes` and evaluates the kernel function.

    Args:
        nodes: Particles including obstacle particles as `Geometry` collection.
        kernel: Kernel function to evaluate.
        boundary: Marks ranges of nodes as boundary particles, see `phi.geom.Graph`.
        desired_neighbors: Target average number of neighbors per particle. This determines the support radius (cutoff) used.
        compute: Comma-separated `str` of kernel properties to compute on the graph edges. Can contain `'kernel'`, `'grad'`, `'laplace'`.
            If no kernel property is given, the edge values will be set to the inverse distance between nodes instead.
        format: Sparse format in which store neighborhood information. Allowed strings are `'dense', `'csr'`, `'coo'`, `'csc'`.
        search_method: Neighborhood search method, see `phi.math.pairwise_differences`.

    Returns:
        `phi.geom.Graph` with edge values storing the kernel values, i.e. the interaction strength between particles.
    """
    assert isinstance(nodes, Geometry), f"nodes must be a Geometry instance but got {type(nodes)}"
    boundary = {} if boundary is None else boundary
    desired_neighbors = _DEFAULT_DESIRED_NEIGHBORS[kernel] if desired_neighbors is None else desired_neighbors
    # --- neighbor search ---
    support = _get_support_radius(nodes.volume, desired_neighbors, nodes.spatial_rank)
    # --- evaluate kernel and derivatives ---
    deltas = math.pairwise_differences(nodes.center, max_distance=support, format=format, method=search_method)
    distances = math.vec_length(deltas, eps=1e-5)
    compute = [s.strip() for s in compute.split(',') if s.strip()]
    if compute:
        kernel = evaluate_kernel(deltas, distances, support, nodes.spatial_rank, kernel, types=compute)
        kernel = [v if 'vector' in v.shape else expand(v, channel(vector=k)) for k, v in kernel.items()]
        edges = concat(kernel, 'vector')
    else:
        edges = math.safe_div(1, distances)
    return Graph(nodes, edges, boundary, deltas=deltas, distances=distances, bounding_distance=support)


def _get_support_radius(volume: Tensor, desired_neighbors: float, spatial_rank: int) -> Tensor:  # volumeToSupport
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


def expected_neighbors(volume: Tensor, support_radius, spatial_rank: int):
    """
    Given the average element volume and support radius, returns the average number of neighbors for a region filled with particles.

    Args:
        volume: Average particle volume.
        support_radius: Other elements are considered neighbors if their center lies within a sphere of this radius around a particle.
        spatial_rank: Spatial rank of the simulation.

    Returns:
        Number of expected neighbors.
    """
    if spatial_rank == 1:
        return 2 * support_radius / volume
    elif spatial_rank == 2:
        return PI * support_radius**2 / volume
    else:
        return 4/3 * PI * support_radius**3 / volume


def evaluate_kernel(delta, distance, h, spatial_rank: int, kernel: str, types: Sequence[str] = ('kernel',)) -> Dict[str, Tensor]:
    """
    Compute the SPH kernel value or a derivative of the kernel function.

    For kernels that only depends on the squared distance, such as `poly6`, the `distance` variable is not used.
    Instead, the squared distance is derived from `delta`.

    Args:
        delta: Vectors to neighbors, i.e. position differences.
        distance: Scalar distance to neighbors.
        h: Support radius / smoothing length / maximum distance / cutoff.
        spatial_rank: Dimensionality of the simulation.
        kernel: Which kernel to use, one of `'quintic-spline'`, `'wendland-c2'`, `'poly6'`.
        types: Ordered `tuple` of derivatives to compute, `'kernel'`, `'grad'`, `'laplace'`.

    Returns:
        `phi.math.Tensor`
    """
    # SPH kernels must be divided by h^d for the kernel and h^(d+1) for the gradient
    assert all(d in ['kernel', 'grad', 'laplace'] for d in types), f"Only derivative=0 and 1 are supported but got {types}"
    d = spatial_rank
    result = {}
    # --- Quintic spline ---
    if kernel == 'quintic-spline':  # cutoff at q = 3 (d=3h)
        q = 3 * distance / h
        if 'kernel' in types:
            norm = 6 * 1 / 120 / h if d == 1 else 6 * 7 / 478 / PI / (h*h) if d == 2 else 6 * 1 / 120 / PI / (h*h*h)
            w1 = (3 - q) ** 5 - 6 * (2 - q) ** 5 + 15 * (1 - q) ** 5
            w2 = (3 - q) ** 5 - 6 * (2 - q) ** 5
            w3 = (3 - q) ** 5
            result['kernel'] = norm * where(q > 2, w3, where(q > 1, w2, w1))
        if 'grad' in types:
            norm = 6 * -5 * 3 / 120 / (h*h) if d == 1 else 6 * -5 * 3 * 7 / 478 / PI / (h*h*h) if d == 2 else 6 * -5 * 3 / 120 / PI / h**4
            w1 = (3 - q) ** 4 - 6 * (2 - q) ** 4 + 15 * (1 - q) ** 4
            w2 = (3 - q) ** 4 - 6 * (2 - q) ** 4
            w3 = (3 - q) ** 4
            result['grad'] = norm * where(q > 2, w3, where(q > 1, w2, w1))
        if 'laplace' in types:
            raise NotImplementedError(f"laplace of {kernel} is not yet supported")
    # --- Wendland C2 ---
    elif kernel == 'wendland-c2':  # cutoff at q=2 (d=2h)
        q = distance / h
        if 'kernel' in types:
            norm = 3 / h if d == 1 else 7 / PI / (h*h) if d == 2 else 21 / 2 / PI / (h*h*h)
            result['kernel'] = norm * (1-q) ** 4 * (4*q + 1)
        if 'grad' in types:
            norm = -20 * 3 / (h*h) if d == 1 else -20 * 7 / PI / (h*h*h) if d == 2 else -20 * 21 / 2 / PI / h**4
            result['grad'] = norm * q * (1-q)**3
        if 'laplace' in types:
            raise NotImplementedError(f"laplace of {kernel} is not yet supported")
    # --- poly6 ---
    elif kernel == 'poly6':  # from Müller et al., Particle-based fluid simulation for interactive applications
        diff = h**2 - math.vec_squared(delta)
        if 'kernel' in types:
            norm = 35 / 16 / h**7 if d == 1 else 4 / PI / h**8 if d == 2 else 315 / 64 / PI / h**9
            result['kernel'] = norm * diff ** 3
        if 'grad' in types:
            norm = -6 * 35 / 16 / h**7 if d == 1 else -6 * 4 / PI / h**8 if d == 2 else -6 * 315 / 64 / PI / h**9
            result['grad'] = delta * norm * diff ** 2
        if 'laplace' in types:
            raise NotImplementedError(f"laplace of {kernel} is not yet supported")
    else:
        raise ValueError(kernel)
    return {t: result[t] for t in types}  # re-order output to match input


def density(graph: Graph) -> Tensor:
    """
    Sum the kernel function over all neighbors within the support radius.

    Args:
        graph: `Graph` with `kernel` values stored in the edges.

    Returns:
        Relative density, i.e. not yet scaled by particle mass.
    """
    return math.sum(graph.edges['kernel'], dual)


# def diffusion(u: Field):
#     kernel_grad = u.graph.edges.vector[1:]
#     du = math.pairwise_differences(u.values, format=u.graph.edges)
#     dr = u.graph.deltas
#     p = du.vector @ dr.vector / math.vec_squared(dr)
#     term = p * kernel_grad
#     return (u.graph.bounding_distance * alpha * c0 * restDensity / rhoi) * math.sum(term, dual)
