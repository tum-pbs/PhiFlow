from typing import Tuple, Dict, Any, Optional, Union

import numpy as np

from ._box import BaseBox, Box, Cuboid
from ._geom import Geometry, GeometryException
from .. import math
from ..math import Shape, Tensor, Extrapolation, stack, vec
from phiml.math._shape import shape_stack, dual, spatial, EMPTY_SHAPE, channel
from ..math.magic import slicing_dict


def _get_bounds(bounds: Union[Box, float, None], resolution: Shape):
    if bounds is None:
        return Box(math.const_vec(0, resolution), math.wrap(resolution, channel(vector=resolution.names)))
    if isinstance(bounds, BaseBox):
        assert set(bounds.vector.item_names) == set(resolution.names), f"bounds dimensions {bounds.vector.item_names} must match resolution {resolution}"
        return bounds.corner_representation()
    if isinstance(bounds, (int, float)):
        return Box(math.const_vec(0, resolution), math.const_vec(bounds, resolution))
    raise ValueError(f"bounds must be a Box, float or None but got {type(bounds).__name__}")


class UniformGrid(BaseBox):
    """
    An instance of UniformGrid represents all cells of a regular grid as a batch of boxes.
    """

    def __init__(self, resolution: Shape = None, bounds: BaseBox = None, **resolution_):
        assert resolution is None or resolution.is_uniform, f"spatial dimensions must form a uniform grid but got {resolution}"
        resolution = (resolution or EMPTY_SHAPE).spatial & spatial(**resolution_)
        bounds = _get_bounds(bounds, resolution)
        assert set(bounds.vector.item_names) == set(resolution.names)
        self._resolution = resolution.only(bounds.vector.item_names, reorder=True)  # reorder only
        self._bounds = bounds
        self._shape = self._resolution & bounds.shape.non_spatial

    @property
    def resolution(self):
        return self._resolution

    @property
    def bounds(self):
        return self._bounds

    @property
    def spatial_rank(self) -> int:
        return self._resolution.spatial_rank

    @property
    def center(self):
        local_coords = math.meshgrid(**{dim.name: math.linspace(0.5 / dim.size, 1 - 0.5 / dim.size, dim) for dim in self.resolution})
        points = self.bounds.local_to_global(local_coords)
        return points

    @property
    def boundary_elements(self) -> Dict[Any, Dict[str, slice]]:
        return {}

    @property
    def boundary_faces(self) -> Dict[Any, Dict[str, slice]]:
        result = {}
        for dim in self.vector.item_names:
            result[dim+'-'] = {'~vector': dim, dim: slice(1)}
            result[dim+'+'] = {'~vector': dim, dim: slice(-1, None)}
        return result

    @property
    def face_centers(self) -> Tensor:
        centers = [self.stagger(dim, True, True).center for dim in self.vector.item_names]
        return stack(centers, dual(vector=self.vector.item_names))

    @property
    def faces(self) -> Geometry:
        slices = [self.stagger(d, True, True) for d in self.resolution.names]
        return stack(slices, dual(vector=self.vector.item_names))

    @property
    def face_normals(self) -> Tensor:
        normals = [vec(**{d: float(d == dim) for d in self.vector.item_names}) for dim in self.vector.item_names]
        return stack(normals, dual(vector=self.vector.item_names))

    @property
    def face_areas(self) -> Tensor:
        areas = [math.prod(self.dx.vector[[d for d in self.vector.item_names if d != dim]], 'vector') for dim in self.vector.item_names]
        return stack(areas, dual(vector=self.vector.item_names))

    @property
    def face_shape(self) -> Shape:
        shapes = [self._shape.spatial.with_dim_size(dim, self._shape.get_size(dim) + 1) for dim in self.vector.item_names]
        return shape_stack(dual(vector=self.vector.item_names), *shapes)

    def interior(self) -> 'Geometry':
        raise GeometryException("Regular grid does not have an interior")

    @property
    def grid_size(self):
        return self._bounds.size

    @property
    def size(self):
        return self.bounds.size / math.wrap(self.resolution.sizes)

    @property
    def dx(self):
        return self.bounds.size / self.resolution

    @property
    def lower(self):
        return self.center - self.half_size

    @property
    def upper(self):
        return self.center + self.half_size

    @property
    def half_size(self):
        return self.bounds.size / self.resolution.sizes / 2

    @property
    def rotation_matrix(self) -> Optional[Tensor]:
        return None

    def __getitem__(self, item):
        item = slicing_dict(self, item)
        resolution = self._resolution.after_gather(item)
        bounds = self._bounds[{d: s for d, s in item.items() if d != 'vector'}]
        if 'vector' in item:
            resolution = resolution.only(item['vector'], reorder=True)
            bounds = bounds.vector[item['vector']]
        bounds = bounds.vector[resolution.name_list]
        dx = self.size
        for dim, selection in item.items():
            if dim in resolution:
                if isinstance(selection, slice):
                    start = selection.start or 0
                    if start < 0:
                        start += self.resolution.get_size(dim)
                    stop = selection.stop or self.resolution.get_size(dim)
                    if stop < 0:
                        stop += self.resolution.get_size(dim)
                    assert selection.step is None or selection.step == 1
                else:  # int slices are not contained in resolution anymore
                    raise ValueError(f"Illegal selection: {item}")
                dim_mask = math.wrap(self.resolution.mask(dim))
                lower = bounds.lower + start * dim_mask * dx
                upper = bounds.upper + (stop - self.resolution.get_size(dim)) * dim_mask * dx
                bounds = Box(lower, upper)
        return UniformGrid(resolution, bounds)

    def __pack_dims__(self, dims: Tuple[str, ...], packed_dim: Shape, pos: Optional[int], **kwargs) -> 'Cuboid':
        return math.pack_dims(self.center_representation(size_variable=False), dims, packed_dim, pos, **kwargs)

    @staticmethod
    def __stack__(values: tuple, dim: Shape, **kwargs) -> 'Geometry':
        from ._geom_ops import GeometryStack
        return GeometryStack(math.layout(values, dim))

    def __replace_dims__(self, dims: Tuple[str, ...], new_dims: Shape, **kwargs) -> 'UniformGrid':
        resolution = math.rename_dims(self._resolution, dims, new_dims).spatial
        bounds = math.rename_dims(self._bounds, dims, new_dims, **kwargs)[resolution.name_list]
        return UniformGrid(resolution, bounds)

    def list_cells(self, dim_name):
        center = math.pack_dims(self.center, self._shape.spatial.names, dim_name)
        return Cuboid(center, self.half_size, size_variable=False)

    def stagger(self, dim: str, lower: bool, upper: bool):
        dim_mask = np.array(self.resolution.mask(dim))
        unit = self.bounds.size / self.resolution * dim_mask
        bounds = Box(self.bounds.lower + unit * (-0.5 if lower else 0.5), self.bounds.upper + unit * (0.5 if upper else -0.5))
        ext_res = self.resolution.sizes + dim_mask * (int(lower) + int(upper) - 1)
        return UniformGrid(self.resolution.with_sizes(ext_res), bounds)

    def staggered_cells(self, boundaries: Extrapolation) -> Dict[str, 'UniformGrid']:
        grids = {}
        for dim in self.vector.item_names:
            grids[dim] = self.stagger(dim, *boundaries.valid_outer_faces(dim))
        return grids

    def padded(self, widths: dict):
        resolution, bounds = self.resolution, self.bounds
        for dim, (lower, upper) in widths.items():
            masked_dx = self.dx * math.dim_mask(self.resolution, dim)
            resolution = resolution.with_dim_size(dim, self.resolution.get_size(dim) + lower + upper)
            bounds = Box(bounds.lower - masked_dx * lower, bounds.upper + masked_dx * upper)
        return UniformGrid(resolution, bounds)

    @property
    def shape(self):
        return self._shape

    def shifted(self, delta: Tensor, **delta_by_dim) -> BaseBox:
        # delta += math.padded_stack()
        if delta.shape.spatial_rank == 0:
            return UniformGrid(self.resolution, self.bounds.shifted(delta))
        else:
            center = self.center + delta
            return Cuboid(center, self.half_size, size_variable=False)

    def rotated(self, angle) -> Geometry:
        raise NotImplementedError("Grids cannot be rotated. Use center_representation() to convert it to Cuboids first.")

    def shallow_equals(self, other):
        return self == other

    def __repr__(self):
        return f"{self._resolution}, bounds={self._bounds}"

    def __variable_attrs__(self):
        return ()

    def __value_attrs__(self):
        return ()

    def __eq__(self, other):
        if not isinstance(other, UniformGrid):
            return False
        return self._resolution == other._resolution and self._bounds == other._bounds

    def __hash__(self):
        return hash(self._resolution) + hash(self._bounds)

    @property
    def _center(self):
        return self.center

    @property
    def _half_size(self):
        return self.half_size

    @property
    def normal(self) -> Tensor:
        raise GeometryException("UniformGrid does not have normals")

    def bounding_half_extent(self) -> Tensor:
        return self.half_size
