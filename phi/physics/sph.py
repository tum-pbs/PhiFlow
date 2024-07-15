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
from phi.geom import Geometry, Graph, Box, Sphere
from phiml.math import channel, stack, vec, concat, expand, clip

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
                   search_method='auto',
                   domain: Box = None,
                   periodic: Union[bool, Tensor] = False) -> Graph:
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
        domain: (Optional) Specify a fixed domain size in which the centers of all nodes must be located.
            This is required for periodic domains.
        periodic: Which domain boundaries should be treated as periodic, i.e. particles on opposite sides are neighbors.
            Can be specified as a `bool` for all sides or as a vector-valued boolean `Tensor` to specify periodicity by direction.

    Returns:
        `phi.geom.Graph` with edge values storing the kernel values, i.e. the interaction strength between particles.
    """
    assert isinstance(nodes, Geometry), f"nodes must be a Geometry instance but got {type(nodes)}"
    boundary = {} if boundary is None else boundary
    desired_neighbors = _DEFAULT_DESIRED_NEIGHBORS[kernel] if desired_neighbors is None else desired_neighbors
    # --- neighbor search ---
    domain = (domain.lower, domain.upper) if domain is not None else None
    support = _get_support_radius(nodes.volume, desired_neighbors, nodes.spatial_rank)
    deltas = math.pairwise_differences(nodes.center, max_distance=support, format=format, method=search_method, domain=domain, periodic=periodic, avg_neighbors=desired_neighbors)
    distances = math.vec_length(deltas, eps=1e-5)
    # --- evaluate kernel and derivatives ---
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
    return Sphere.radius_from_volume(volume * desired_neighbors, spatial_rank)


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
    return Sphere.volume_from_radius(support_radius, spatial_rank) / volume


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
    if kernel == 'quintic-spline':
        const = 3**5 / 40 if d == 1 else 3**7 * 7 / 478 / PI if d == 2 else 3**7 / 40 / PI
        q = distance / h
        if 'kernel' in types:
            k = clip(1-q)**5 - 6 * clip(2/3-q)**5 + 15 * clip(1/3-q)**5
            result['kernel'] = const / h**d * k
        if 'grad' in types:
            dk = -5 * clip(1-q)**4 + 30 * clip(2/3-q)**4 - 75 * clip(1/3-q)**4
            result['grad'] = const / h**(d+1) * dk * math.safe_div(delta, distance)
        if 'laplace' in types:
            d2k = 20 * clip(1-q)**3 - 120 * clip(2/3-q)**3 + 300 * clip(1/3-q)**3
            result['laplace'] = const / h**(d+2) * d2k
    # --- Wendland C2 ---
    elif kernel == 'wendland-c2':
        const = 3 / 2 if d == 1 else 7 / PI if d == 2 else 21 / 2 / PI
        q = distance / h
        if 'kernel' in types:
            k = (1-q) ** 4 * (4*q + 1)
            result['kernel'] = const / h**d * k
        if 'grad' in types:
            dk = -20 * q * (1-q)**3
            result['grad'] = const / h**(d+1) * dk * math.safe_div(delta, distance)
        if 'laplace' in types:
            d2k = 20 * (4*q - 1) * (1-q)**2
            result['laplace'] = const / h**(d+2) * d2k
    # --- poly6 from Müller et al., Particle-based fluid simulation for interactive applications ---
    elif kernel == 'poly6':
        const = 35 / 32 if d == 1 else 4 / PI if d == 2 else 315 / 64 / PI
        norm = const / h**(d+6)
        r2 = math.vec_squared(delta)
        diff = h**2 - r2
        if 'kernel' in types:
            result['kernel'] = norm * diff ** 3
        if 'grad' in types:
            result['grad'] = -6 * norm * diff**2 * delta
        if 'laplace' in types:
            result['laplace'] = -6 * norm * (5*r2**2 - 6*r2*h**2 + h**4)
    else:
        raise ValueError(kernel)
    return {t: result[t] for t in types}  # re-order output to match input


# def density(graph: Graph) -> Tensor:
#     """
#     Sum the kernel function over all neighbors within the support radius.
#
#     Args:
#         graph: `Graph` with `kernel` values stored in the edges.
#
#     Returns:
#         Relative density, i.e. not yet scaled by particle mass.
#     """
#     return math.sum(graph.edges['kernel'], dual)


# def diffusion(u: Field):
#     kernel_grad = u.graph.edges.vector[1:]
#     du = math.pairwise_differences(u.values, format=u.graph.edges)
#     dr = u.graph.deltas
#     p = du.vector @ dr.vector / math.vec_squared(dr)
#     term = p * kernel_grad
#     return (u.graph.bounding_distance * alpha * c0 * restDensity / rhoi) * math.sum(term, dual)
