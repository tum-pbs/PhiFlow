from typing import Dict, Tuple, Any, Union

from ._geom import Geometry
from .. import math
from ..math import Tensor, pairwise_distances, vec_length, Shape, non_channel, dual, where, PI


class DynamicGraph(Geometry):

    def __init__(self, nodes: Geometry, boundary: Dict[str, Tuple[Dict[str, slice], Dict[str, slice]]], kernel: str, format='dense'):
        self._nodes = nodes
        self._kernel = kernel
        self._format = format
        self._boundary = boundary
        self._deltas = None
        self._distances = None

    @property
    def nodes(self):
        return self._nodes

    @property
    def kernel(self):
        return self._kernel

    @property
    def format(self):
        return self._format

    @property
    def center(self) -> Tensor:
        return self._nodes.center

    @property
    def boundary_elements(self) -> Dict[str, Tuple[Dict[str, slice], Dict[str, slice]]]:
        return self._boundary_elements

    @property
    def boundary_faces(self) -> Dict[str, Tuple[Dict[str, slice], Dict[str, slice]]]:
        return {key: {'~' + dim: s for dim, s in slices.items()} for key, slices in self._boundary_elements.items()}

    @property
    def distances(self) -> Tensor:
        if self._distances is None:
            self._distances = vec_length(self.deltas)
        return self._distances

    @property
    def element_size(self):
        return 2 * math.max(self._nodes.bounding_half_extent(), 'vector')

    @property
    def deltas(self) -> Tensor:
        """
        Returns the pairwise position deltas between all elements as `Tensor`, possibly sparse depending on `format´.
        The result has shape (elements, ~elements, vector).
        """
        if self._deltas is None:
            max_distance = get_kernel_cutoff(self._kernel, self.element_size)
            self._deltas = pairwise_distances(self.center, max_distance, format=self._format)
        return self._deltas

    def __with_attrs__(self, **attrs):  # Make sure cached distances are invalidated
        construct_kwargs = {'nodes': self._nodes, 'boundary': self._boundary, 'kernel': self._kernel, 'format': self._format}
        construct_kwargs.update({k[1:] if k.startswith('_') else k: v for k, v in attrs.items()})
        return DynamicGraph(**construct_kwargs)

    def __variable_attrs__(self) -> Tuple[str, ...]:
        return '_nodes',

    def at(self, center: Tensor) -> 'Geometry':
        return DynamicGraph(self._nodes.at(center), self._boundary, self._kernel, self._format)

    @property
    def volume(self) -> Tensor:
        return self._nodes.volume

    @property
    def shape(self) -> Shape:
        return self._nodes.shape

    @property
    def face_centers(self) -> Tensor:
        return self._nodes.center + .5 * self.deltas

    @property
    def face_areas(self) -> Tensor:
        dual_dims = dual(**non_channel(self._nodes.shape).non_batch.untyped_dict)
        return math.zeros(non_channel(self._nodes), dual_dims)

    @property
    def face_normals(self) -> Tensor:
        return self.deltas / self.distances

    @property
    def face_shape(self) -> Shape:
        return non_channel(self._nodes) & dual(**non_channel(self._nodes.shape).non_batch.untyped_dict)

    @property
    def shape_type(self) -> Tensor:
        return self._nodes.shape_type

    def lies_inside(self, location: Tensor) -> Tensor:
        raise NotImplementedError(f"lies_inside not defined for DynamicGraph")

    def approximate_signed_distance(self, location: Union[Tensor, tuple]) -> Tensor:
        raise NotImplementedError(f"approximate_signed_distance not defined for DynamicGraph")

    def push(self, positions: Tensor, outward: bool = True, shift_amount: float = 0) -> Tensor:
        raise NotImplementedError(f"push not defined for DynamicGraph")

    def sample_uniform(self, *shape: math.Shape) -> Tensor:
        raise NotImplementedError(f"sample_uniform not defined for DynamicGraph")

    def bounding_radius(self) -> Tensor:
        return self._nodes.bounding_radius()

    def bounding_half_extent(self) -> Tensor:
        return self._nodes.bounding_half_extent()

    def rotated(self, angle: Union[float, Tensor]) -> 'Geometry':
        raise NotImplementedError(f"rotated not defined for DynamicGraph")

    def scaled(self, factor: Union[float, Tensor]) -> 'Geometry':
        return DynamicGraph(self._nodes.scaled(factor), self._boundary, self._kernel, self._format)  # this may add more connections

    def __hash__(self):
        return hash((self._nodes, self._boundary, self._kernel, self._format))

    def __getitem__(self, item):
        return DynamicGraph(self._nodes[item], self._boundary, self._kernel, self._format)


def get_kernel_cutoff(kernel: str, element_size):
    """
    Returns the cut-off distance for a kernel given the element size.

    Args:
        kernel: Kernel name as `str`, one of `'wendland-c2'`, `'quintic-spline'`.
        element_size: Physical size of elements as `float` or `Tensor`.
            Pass `1` to get the cut-off in normalized space.

    Returns:
        Cut-off distance as float or float `Tensor`
    """
    if kernel == 'quintic-spline':
        return 3. * element_size
    elif kernel == 'wendland-c2':
        return 2. * element_size
    else:
        raise ValueError(kernel)


def evaluate_kernel(q: Tensor, spatial_rank: int, kernel='wendland-c2', derivative=0, enforce_cutoff=True):
    """
    Compute the SPH kernel value at a normalized scalar distance `q` or a derivative of the kernel function.

    Args:
        q: Normalized distance `phi.math.Tensor`.
        spatial_rank: Dimensionality of the simulation.
        kernel: Which kernel to use, one of `'wendland-c2'`, `'quintic-spline'`.
        derivative: Derivative order, `0` for kernel function, `1` for gradient.
        enforce_cutoff: If `True`, returns 0 outside the kernel's defined range, else the result is undefined.

    Returns:
        `phi.math.Tensor`
    """
    if kernel == 'quintic-spline':  # cutoff at q = 3 (d=3h)
        alpha_d = {1: 1/120, 2: 7/478/PI, 3: 1/120/PI}[spatial_rank]
        if not derivative:
            w1 = (3-q)**5 - 6 * (2-q)**5 + 15 * (1-q)**5
            w2 = (3-q)**5 - 6 * (2-q)**5
            w3 = (3-q)**5
            fac = alpha_d
        elif derivative == 1:
            w1 = (3-q)**4 - 6 * (2-q)**4 + 15 * (1-q)**4
            w2 = (3-q)**4 - 6 * (2-q)**4
            w3 = (3-q)**4
            fac = -5 * alpha_d
        else:
            raise NotImplementedError(f"{derivative}th derivative of {kernel} is not supported")
        return fac * _cutoff(where(q > 2, w3, where(q > 1, w2, w1)), q, kernel, enforce_cutoff)
    elif kernel == 'wendland-c2':  # cutoff at q=2 (d=2h)
        alpha_d = {2: 7/4/PI, 3: 21/16/PI}[spatial_rank]
        if not derivative:
            w = (1 - 0.5*q)**4 * (2*q + 1)
            fac = alpha_d
        elif derivative == 1:
            w = (1 - 0.5*q)**3 * q
            fac = -5 * alpha_d
        else:
            raise NotImplementedError(f"{derivative}th derivative of {kernel} is not supported")
        return fac * _cutoff(w, q, kernel, enforce_cutoff)
    else:
        raise ValueError(kernel)


def _cutoff(value, q, kernel: str, enforce_cutoff: bool):
    if not enforce_cutoff:
        return value
    else:
        cutoff = get_kernel_cutoff(kernel, 1)
        return where(q <= cutoff, value, 0)
