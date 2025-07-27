from dataclasses import dataclass
from functools import cached_property
from typing import Tuple, Dict, Any, Optional, Union

import numpy as np

from phiml.dataclasses import sliceable
from phiml.math import rename_dims, wrap
from ._box import Box, Cuboid, bounding_box
from ._functions import vec_length
from ._geom import Geometry, GeometryException
from .. import math
from ..math import Shape, Tensor, Extrapolation, stack, vec
from phiml.math._shape import shape_stack, dual, spatial, EMPTY_SHAPE, channel, batch, shape, non_spatial
from ..math.magic import slicing_dict


def _get_bounds(bounds: Union[Box, float, None], resolution: Shape):
    if bounds is None:
        return Box(math.const_vec(0, resolution), math.wrap(resolution, channel(vector=resolution.names)))
    if isinstance(bounds, Box):
        assert set(bounds.vector.labels) == set(resolution.names), f"bounds dimensions {bounds.vector.labels} must match resolution {resolution}"
        return bounds.corner_representation()
    if isinstance(bounds, (int, float)):
        return Box(math.const_vec(0, resolution), math.const_vec(bounds, resolution))
    raise ValueError(f"bounds must be a Box, float or None but got {type(bounds).__name__}")


class UniformGridType(type):
    
    def __call__(cls, resolution: Shape = None, bounds: Box = None, **resolution_):
        assert resolution is None or resolution.is_uniform, f"spatial dimensions must form a uniform grid but got {resolution}"
        resolution = (resolution or EMPTY_SHAPE).spatial & spatial(**resolution_)
        bounds = _get_bounds(bounds, resolution)
        resolution = resolution.only(bounds.vector.labels, reorder=True)  # reorder only
        return type.__call__(cls, resolution, bounds)


@sliceable(keepdims='vector')
@dataclass(frozen=True, eq=False)
class UniformGrid(Geometry, metaclass=UniformGridType):
    """
    An instance of UniformGrid represents all cells of a regular grid as a batch of boxes.
    """
    resolution: Shape
    bounds: Box
    
    def __post_init__(self):
        assert set(self.bounds.vector.labels) == set(self.resolution.names)

    @property
    def spatial_rank(self) -> int:
        return self.resolution.spatial_rank
    
    @cached_property
    def shape(self):
        return self.resolution & non_spatial(self.bounds)

    @cached_property
    def center(self):
        local_coords = math.meshgrid(**{dim.name: math.linspace(0.5 / dim.size, 1 - 0.5 / dim.size, dim) for dim in self.resolution})
        points = self.bounds.local_to_global(local_coords)
        return points

    def position_of(self, voxel_index: Tensor):
        voxel_index = rename_dims(voxel_index, channel, 'vector')
        return self.bounds.lower + (voxel_index+.5) / self.resolution * self.bounds.size

    def voxel_at(self, location: Tensor, clamp=True):
        float_idx = (location - self.bounds.lower) / self.bounds.size * self.resolution
        index = math.to_int32(float_idx)
        if clamp:
            index = math.clip(index, 0, wrap(self.resolution, channel('vector'))-1)
        return index

    @property
    def boundary_elements(self) -> Dict[Any, Dict[str, slice]]:
        return {}

    @property
    def boundary_faces(self) -> Dict[Any, Dict[str, slice]]:
        result = {}
        for dim in self.vector.labels:
            result[dim+'-'] = {'~vector': dim, dim: slice(1)}
            result[dim+'+'] = {'~vector': dim, dim: slice(-1, None)}
        return result

    @property
    def face_centers(self) -> Tensor:
        centers = [self.stagger(dim, True, True).center for dim in self.vector.labels]
        return stack(centers, dual(vector=self.vector.labels))

    @property
    def faces(self) -> Geometry:
        slices = [self.stagger(d, True, True) for d in self.resolution.names]
        return stack(slices, dual(vector=self.vector.labels))

    @property
    def face_normals(self) -> Tensor:
        normals = [vec(**{d: float(d == dim) for d in self.vector.labels}) for dim in self.vector.labels]
        return stack(normals, dual(vector=self.vector.labels))

    @property
    def face_areas(self) -> Tensor:
        areas = [math.prod(self.dx.vector[[d for d in self.vector.labels if d != dim]], 'vector') for dim in self.vector.labels]
        return stack(areas, dual(vector=self.vector.labels))

    @cached_property
    def face_shape(self) -> Shape:
        staggered_shapes = [self.shape.spatial.with_dim_size(dim, self.shape.get_size(dim) + 1) for dim in self.vector.labels]
        return shape_stack(dual(vector=self.vector.labels), *staggered_shapes)

    def interior(self) -> 'Geometry':
        raise GeometryException("Regular grid does not have an interior")

    @property
    def grid_size(self):
        return self.bounds.size

    @cached_property
    def dx(self):
        return self.bounds.size / self.resolution

    @property
    def size(self):
        return self.dx

    @property
    def volume(self) -> Tensor:
        return math.prod(self.dx, 'vector')

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
    def _rot_or_none(self) -> Optional[Tensor]:
        return None

    def corner_representation(self) -> 'Box':
        return Box(self.lower, self.upper)

    box = corner_representation

    def center_representation(self, size_variable=True) -> 'Cuboid':
        return Cuboid(self.center, self.half_size, size_variable=size_variable)

    cuboid = center_representation

    def with_scaled_resolution(self, scale: float):
        return UniformGrid(self.resolution.with_sizes([s*scale for s in self.resolution.sizes]), self.bounds)

    def __getitem__(self, item):
        item = slicing_dict(self, item)
        resolution = self.resolution.after_gather(item)
        bounds = self.bounds[{d: s for d, s in item.items() if d != 'vector'}]
        if 'vector' in item:
            resolution = resolution.only(item['vector'], reorder=True)
            bounds = bounds.vector[item['vector']]
        bounds = bounds.vector[resolution.name_list]
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
                lower = bounds.lower + start * dim_mask * self.dx
                upper = bounds.upper + (stop - self.resolution.get_size(dim)) * dim_mask * self.dx
                bounds = Box(lower, upper)
        return UniformGrid(resolution, bounds)

    def __pack_dims__(self, dims: Tuple[str, ...], packed_dim: Shape, pos: Optional[int], **kwargs) -> 'Box':
        return math.pack_dims(self.center_representation(size_variable=False), dims, packed_dim, pos, **kwargs)

    @staticmethod
    def __stack__(values: tuple, dim: Shape, **kwargs) -> 'Geometry':
        from ._geom_ops import GeometryStack
        return GeometryStack(math.layout(values, dim))

    def __replace_dims__(self, dims: Tuple[str, ...], new_dims: Shape, **kwargs) -> 'UniformGrid':
        resolution = math.rename_dims(self.resolution, dims, new_dims).spatial
        bounds = math.rename_dims(self.bounds, dims, new_dims, **kwargs)[resolution.name_list]
        return UniformGrid(resolution, bounds)

    def list_cells(self, dim_name):
        center = math.pack_dims(self.center, self.shape.spatial.names, dim_name)
        return Cuboid(center, self.half_size, size_variable=False)

    def stagger(self, dim: str, lower: bool, upper: bool):
        dim_mask = np.array(self.resolution.mask(dim))
        unit = self.bounds.size / self.resolution * dim_mask
        bounds = Box(self.bounds.lower + unit * (-0.5 if lower else 0.5), self.bounds.upper + unit * (0.5 if upper else -0.5))
        ext_res = self.resolution.sizes + dim_mask * (int(lower) + int(upper) - 1)
        return UniformGrid(self.resolution.with_sizes(ext_res), bounds)

    def staggered_cells(self, boundaries: Extrapolation) -> Dict[str, 'UniformGrid']:
        grids = {}
        for dim in self.vector.labels:
            grids[dim] = self.stagger(dim, *boundaries.valid_outer_faces(dim))
        return grids

    def padded(self, widths: dict):
        resolution, bounds = self.resolution, self.bounds
        for dim, (lower, upper) in widths.items():
            masked_dx = self.dx * math.dim_mask(self.resolution, dim)
            resolution = resolution.with_dim_size(dim, self.resolution.get_size(dim) + lower + upper)
            bounds = Box(bounds.lower - masked_dx * lower, bounds.upper + masked_dx * upper)
        return UniformGrid(resolution, bounds)

    def shifted(self, delta: Tensor, **delta_by_dim):
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
        return f"{self.resolution}, bounds={self.bounds}"

    def __eq__(self, other):
        if not isinstance(other, UniformGrid):
            return False
        return self.resolution == other.resolution and self.bounds == other.bounds

    def __hash__(self):
        return hash(self.resolution) + hash(self.bounds)

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

    def bounding_radius(self) -> Tensor:
        return vec_length(self.half_size)


def enclosing_grid(*geometries: Union[Geometry, Tensor], voxel_count: int, rel_margin=0., abs_margin=0., margin_cells=0) -> UniformGrid:
    """
    Constructs a `UniformGrid` which fully encloses the `geometries`.
    The grid voxels are chosen to have approximately the same size along each axis.

    Args:
        *geometries: `Geometry` objects `Tensor` of points which should lie within the grid.
        voxel_count: Approximate number of total voxels.
        rel_margin: Relative margin, i.e. empty space on each side as a fraction of the bounding box size of `geometries`.
        abs_margin: Absolute margin, i.e. empty space on each side.
        margin_cells: Number of cell layers to fit outside the bounding box around `geometries`. This is cumulative with `rel_margin` and `abs_margin`.

    Returns:
        `UniformGrid`
    """
    bounds = stack([g.bounding_box() if isinstance(g, Geometry) else bounding_box(g) for g in geometries], batch('_geometries'))
    bounds = bounds.largest(shape).scaled(1+rel_margin)
    bounds = Box(bounds.lower - abs_margin, bounds.upper + abs_margin)
    if not margin_cells:
        voxel_vol = bounds.volume / voxel_count
        voxel_size = voxel_vol ** (1/bounds.spatial_rank)
        resolution = math.to_int32(math.round(bounds.size / voxel_size))
        resolution = spatial(**resolution.vector)
    else:
        inner_res, outer_res = solve_resolution_with_margin_cells(*bounds.size.vector, voxel_count, margin_cells)
        dx = bounds.size / inner_res
        bounds = Box(bounds.lower - dx*margin_cells, bounds.upper + dx*margin_cells)
        resolution = spatial(**{d: r for d, r in zip(bounds.size.vector.labels, outer_res)})
    return UniformGrid(resolution, bounds)


def solve_resolution_with_margin_cells(W, H, L, n, l: int = 1):
    coeffs = [W * H * L, 2 * l * (W*H + W*L + H*L), 4 * l**2 * (W+H+L), 8 * l**3 - n]
    roots = np.roots(coeffs)
    real_positive_roots = [r.real for r in roots if np.isreal(r) and r.real > 0]  # Filter out only the real, positive roots
    if not real_positive_roots:
        raise ValueError(f"No grid resolution fulfills margin_cells={l} given {n} total cells.")
    inner_res = np.round(real_positive_roots[0] * np.asarray([W, H, L]))
    return inner_res, inner_res + 2*l
