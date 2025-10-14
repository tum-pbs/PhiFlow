from dataclasses import dataclass
from functools import cached_property
from typing import Union, Tuple, Dict, Any, Callable, Optional

from phiml import math, wrap
from phiml.dataclasses import sliceable
from phiml.math import Shape, Tensor, channel, instance
from phiml.math.magic import slicing_dict
from ._box import Box
from ._geom import Geometry


@sliceable(keepdims='vector')
@dataclass(frozen=True, eq=False)
class SDF(Geometry):
    """
    Function-based signed distance field.
    Negative values lie inside the geometry, the 0-level represents the surface.
    """
    sdf: Callable  # SDF function. First argument is a `phiml.math.Tensor` with a `vector` channel dim.
    bounds: Box  # Grid limits. The bounds fully enclose all virtual cells.
    vol: Tensor = None
    grad_fn: Optional[Callable] = None  # Returns (sdf, grad) when called with location

    variable_attrs: Tuple[str, ...] = 'bounds',
    value_attrs: Tuple[str, ...] = ()
    
    def __call__(self, location, *aux_args, **aux_kwargs):
        native_loc = not isinstance(location, Tensor)
        if native_loc:
            location = wrap(location, instance('points'), self.shape['vector'])
        sdf_val: Tensor = self.sdf(location, *aux_args, **aux_kwargs)
        return sdf_val.native() if native_loc else sdf_val
    
    @cached_property
    def out_shape(self):
        dims = channel(self.bounds)
        assert 'vector' in dims, f"If out_shape is not specified, either bounds, center or bounding_radius must be given."
        return self.sdf(math.zeros(dims['vector'])).shape
    
    @cached_property
    def grad(self) -> Callable:
        return math.gradient(self.sdf, wrt=0, get_output=True) if self.grad_fn is None else self.grad_fn
    
    @property
    def center(self):
        return self.bounds.center
    
    @property
    def volume(self):
        return self.vol

    @property
    def size(self):
        return self.bounds.size

    @property
    def shape(self) -> Shape:
        return self.out_shape & self.bounds.shape

    @property
    def faces(self) -> 'Geometry':
        raise NotImplementedError(f"SDF does not support faces")

    @property
    def face_centers(self) -> Tensor:
        raise NotImplementedError(f"SDF does not support faces")

    @property
    def face_areas(self) -> Tensor:
        raise NotImplementedError(f"SDF does not support faces")

    @property
    def face_normals(self) -> Tensor:
        raise NotImplementedError(f"SDF does not support faces")

    @property
    def boundary_elements(self) -> Dict[Any, Dict[str, slice]]:
        return {}

    @property
    def boundary_faces(self) -> Dict[Any, Dict[str, slice]]:
        return {}

    @property
    def face_shape(self) -> Shape:
        return math.EMPTY_SHAPE

    @property
    def corners(self) -> Tensor:
        raise NotImplementedError(f"SDF does not support corners")

    def lies_inside(self, location: Tensor) -> Tensor:
        sdf = self.sdf(location)
        return sdf <= 0

    def approximate_closest_surface(self, location: Tensor, refine_iter=0) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        sgn_dist, outward = self.grad(location)
        closest = location - sgn_dist * outward
        if not refine_iter:
            _, normal = self.grad(closest)
        else:
            for i in range(refine_iter):
                sgn_dist, outward = self.grad(closest)
                closest -= sgn_dist * outward
            normal = outward
        offset = None
        face_index = None
        return sgn_dist, closest - location, normal, offset, face_index

    def sdf_and_gradient(self, location: Tensor, refine_iter=0) -> Tuple[Tensor, Tensor]:
        if not refine_iter:
            sgn_dist, outward = self.grad(location)
        else:
            sgn_dist, delta, *_ = self.approximate_closest_surface(location)
            outward = math.vec_normalize(math.sign(-sgn_dist) * delta)
        return sgn_dist, outward

    def approximate_signed_distance(self, location: Tensor) -> Tensor:
        return self.sdf(location)

    def sample_uniform(self, *shape: math.Shape) -> Tensor:
        raise NotImplementedError

    def bounding_radius(self) -> Tensor:
        return self.bounds.bounding_radius()

    def bounding_half_extent(self) -> Tensor:
        return self.bounds.half_size  # this could be too small if the center is not in the middle of the bounds

    def bounding_box(self) -> 'Box':
        return self.bounds

    def shifted(self, delta: Tensor) -> 'Geometry':
        raise NotImplementedError("SDF does not yet support shifting")

    def at(self, center: Tensor) -> 'Geometry':
        raise NotImplementedError("SDF does not yet support shifting")

    def rotated(self, angle: Union[float, Tensor]) -> 'Geometry':
        raise NotImplementedError("SDF does not yet support rotation")

    def scaled(self, factor: Union[float, Tensor]) -> 'Geometry':
        raise NotImplementedError("SDF does not yet support scaling")

    def __getitem__(self, item):
        item = slicing_dict(self, item)
        if not item:
            return self
        raise NotImplementedError("SDF cannot be sliced.")

    @staticmethod
    def __stack__(values: tuple, dim: Shape, **kwargs) -> 'Geometry':
        from ._geom_ops import GeometryStack
        return GeometryStack(math.layout(values, dim))


def numpy_sdf(sdf: Callable, bounds: Box) -> SDF:
    """
    Define a `SDF` (signed distance function) from a NumPy function.

    Args:
        sdf: Function mapping a location `numpy.ndarray` of shape `(points, vector)` to the corresponding SDF value `(points,)`.
        bounds: Bounds inside which the function is defined.

    Returns:
        `SDF`
    """
    def native_sdf_function(pos: Tensor) -> Tensor:
        nat_pos = math.reshaped_native(pos, [..., 'vector'])
        nat_sdf = pos.default_backend.numpy_call(sdf, nat_pos.shape[:1], math.DType(float, 32), nat_pos)
        with pos.default_backend:
            return math.reshaped_tensor(nat_sdf, [pos.shape - 'vector'])
    result = SDF(native_sdf_function, bounds)
    result.__dict__['out_shape'] = math.EMPTY_SHAPE
    return result
