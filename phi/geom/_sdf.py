from typing import Union, Tuple, Dict, Any, Callable

from phiml import math
from phiml.math import Shape, Tensor, spatial, channel, instance
from . import UniformGrid
from ._box import BaseBox
from ._geom import Geometry


class SDF(Geometry):
    """
    Function-based signed distance field.
    Negative values lie inside the geometry, the 0-level represents the surface.
    """
    def __init__(self, sdf: Callable, out_shape=None, bounds: BaseBox = None, center: Tensor = None, volume: Tensor = None, bounding_radius: Tensor = None):
        """
        Args:
            sdf: SDF function. First argument is a `phiml.math.Tensor` with a `vector` channel dim.
            bounds: Grid limits. The bounds fully enclose all virtual cells.
            center: (Optional) Geometry center point. Will be computed otherwise.
            volume: (Optional) Geometry volume. Will be computed otherwise.
            bounding_radius: (Optional) Geometry bounding radius around center. Will be computed otherwise.
        """
        self._sdf = sdf
        if out_shape is not None:
            self._out_shape = out_shape
        else:
            dims = channel([bounds, center, bounding_radius])
            assert 'vector' in dims, f"If out_shape is not specified, either bounds, center or bounding_radius must be given."
            self._out_shape = sdf(math.zeros(dims['vector'])).shape
        self._bounds = bounds
        self._grad = math.gradient(sdf, wrt=0, get_output=True)
        if center is not None:
            self._center = center
        else:
            self._center = bounds.center
        if volume is not None:
            self._volume = volume
        else:
            self._volume = None
        if bounding_radius is not None:
            self._bounding_radius = bounding_radius
        else:
            self._bounding_radius = self._bounds.bounding_radius()

    def __call__(self, location, *aux_args, **aux_kwargs):
        native_loc = not isinstance(location, Tensor)
        if native_loc:
            location = math.wrap(location, instance('points'), self.shape['vector'])
        sdf_val: Tensor = self._sdf(location, *aux_args, **aux_kwargs)
        return sdf_val.native() if native_loc else sdf_val

    @property
    def values(self):
        """Signed distance grid."""
        return self._sdf

    @property
    def bounds(self) -> BaseBox:
        return self._bounds

    @property
    def size(self):
        return self._bounds.size

    @property
    def resolution(self):
        return spatial(self._sdf)

    @property
    def points(self):
        return UniformGrid(spatial(self._sdf), self._bounds).center

    @property
    def center(self) -> Tensor:
        return self._center

    @property
    def shape(self) -> Shape:
        return self._out_shape & self._bounds.shape

    @property
    def volume(self) -> Tensor:
        return self._volume

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
        raise NotImplementedError(f"SDF does not support boundaries")

    @property
    def corners(self) -> Tensor:
        raise NotImplementedError(f"SDF does not support corners")

    def lies_inside(self, location: Tensor) -> Tensor:
        sdf = self._sdf(location)
        return sdf <= 0

    def approximate_closest_surface(self, location: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        sgn_dist, neg_direction = self._grad(location)
        delta = sgn_dist * -neg_direction
        _, normal = self._grad(location + delta)
        offset = None
        face_index = None
        return sgn_dist, delta, normal, offset, face_index

    def approximate_signed_distance(self, location: Tensor) -> Tensor:
        return self._sdf(location)

    def sample_uniform(self, *shape: math.Shape) -> Tensor:
        raise NotImplementedError

    def bounding_radius(self) -> Tensor:
        return self._bounding_radius

    def bounding_half_extent(self) -> Tensor:
        return self._bounds.half_size  # this could be too small if the center is not in the middle of the bounds

    def bounding_box(self) -> 'BaseBox':
        return self._bounds

    def shifted(self, delta: Tensor) -> 'Geometry':
        raise NotImplementedError("SDF does not yet support shifting")

    def at(self, center: Tensor) -> 'Geometry':
        raise NotImplementedError("SDF does not yet support shifting")

    def rotated(self, angle: Union[float, Tensor]) -> 'Geometry':
        raise NotImplementedError("SDF does not yet support rotation")

    def scaled(self, factor: Union[float, Tensor]) -> 'Geometry':
        off_center = self._center - self._bounds.center
        volume = self._volume * factor ** self.spatial_rank
        bounds = self._bounds.scaled(factor).shifted(off_center * (factor - 1)).corner_representation()
        return SDF(self._sdf, bounds, self._center, volume, self._bounding_radius * factor)

    def __getitem__(self, item):
        if not item:
            return self
        raise NotImplementedError
