from numbers import Number
from typing import TypeVar, Callable

from phi import math
from phi.geom import Box, Geometry, GridCell
from . import HardGeometryMask
from ._field import SampledField, Field, sample, reduce_sample
from ..geom._stack import GeometryStack
from ..math import Shape
from ..math._shape import CHANNEL_DIM
from ..math._tensors import TensorStack, Tensor


class Grid(SampledField):
    """
    Base class for `CenteredGrid` and `StaggeredGrid`.

    Use `grid()` to create a grid.
    Alternatively, the `phi.physics.Domain` class provides convenience methods for grid creation.
    """

    def __init__(self, elements: Geometry, values: Tensor, extrapolation: math.Extrapolation, resolution: Shape, bounds: Box):
        SampledField.__init__(self, elements, values, extrapolation)
        assert values.shape.spatial_rank == elements.spatial_rank, f"Spatial dimensions of values ({values.shape}) do not match elements {elements}"
        assert values.shape.spatial_rank == bounds.spatial_rank, f"Spatial dimensions of values ({values.shape}) do not match elements {elements}"
        self._bounds = bounds
        self._resolution = resolution

    def closest_values(self, points: Geometry):
        """
        Sample the closest grid point values of this field at the world-space locations (in physical units) given by `points`.
        Points must have a single channel dimension named `vector`.
        It may additionally contain any number of batch and spatial dimensions, all treated as batch dimensions.

        Args:
            points: world-space locations

        Returns:
            Closest grid point values as a `Tensor`.
            For each dimension, the grid points immediately left and right of the sample points are evaluated.
            For each point in `points`, a *2^d* cube of points is determined where *d* is the number of spatial dimensions of this field.
            These values are stacked along the new dimensions `'closest_<dim>'` where `<dim>` refers to the name of a spatial dimension.
        """
        raise NotImplementedError(self)

    def _sample(self, geometry: Geometry) -> math.Tensor:
        raise NotImplementedError(self)

    def with_(self, elements: Geometry or None = None, values: Tensor = None, extrapolation: math.Extrapolation = None, **other_attributes) -> 'Grid':
        raise NotImplementedError(self)

    def __getitem__(self, item: dict) -> 'Grid':
        raise NotImplementedError(self)

    @property
    def shape(self):
        return self._resolution & self._values.shape.non_spatial

    @property
    def bounds(self) -> Box:
        return self._bounds

    @property
    def box(self) -> Box:
        return self._bounds

    @property
    def resolution(self) -> Shape:
        return self._resolution

    @property
    def dx(self) -> Tensor:
        return self.box.size / self.resolution

    def __repr__(self):
        return f"{self.__class__.__name__}[{self.shape.non_spatial & self.resolution}, size={self.box.size}, extrapolation={self._extrapolation}]"


GridType = TypeVar('GridType', bound=Grid)


class CenteredGrid(Grid):
    """
    N-dimensional grid with values sampled at the cell centers.
    A centered grid is defined through its data tensor, its bounds describing the physical size, and its extrapolation.
    
    Centered grids support arbitrary batch, spatial and channel dimensions.

    Use `grid()` with `type=CenteredGrid` (default) to create a centered grid.
    Alternatively, the `phi.physics.Domain` class provides convenience methods for grid creation.
    
    See the `phi.field` module documentation at https://tum-pbs.github.io/PhiFlow/Fields.html
    """

    def __init__(self, values: math.Tensor, bounds: Box, extrapolation: math.Extrapolation):
        """
        Args:
            values: `phi.math.Tensor` containing all dimensions of this grid.
            bounds: Physical size and location of the grid.
            extrapolation: The grid extrapolation determines the value outside the `values` tensor.
        """
        Grid.__init__(self, GridCell(values.shape.spatial, bounds), values, extrapolation, values.shape.spatial, bounds)

    def with_(self,
              elements: Geometry or None = None,
              values: Tensor = None,
              extrapolation: math.Extrapolation = None,
              **other_attributes) -> 'CenteredGrid':
        assert elements is None
        assert not other_attributes, f"Invalid attributes for type {type(self)}: {other_attributes}"
        values = values if values is not None else self.values
        extrapolation = extrapolation if extrapolation is not None else self._extrapolation
        return CenteredGrid(values, self.bounds, extrapolation)

    def __getitem__(self, item: dict):
        values = self._values[{dim: slice(sel, sel + 1) if isinstance(sel, int) and dim in self.shape.spatial else sel for dim, sel in item.items()}]
        extrapolation = self._extrapolation[item]
        bounds = self.elements[item].bounds
        return CenteredGrid(values, bounds, extrapolation)

    def _sample(self, geometry: Geometry) -> Tensor:
        if isinstance(geometry, GeometryStack):
            sampled = [self.sample(g) for g in geometry.geometries]
            return math.batch_stack(sampled, geometry.stack_dim_name)
        if isinstance(geometry, GridCell):
            if self.elements == geometry:
                return self.values
            elif math.close(self.dx, geometry.size):
                fast_resampled = self._shift_resample(geometry.resolution, geometry.bounds)
                if fast_resampled is not NotImplemented:
                    return fast_resampled
        points = geometry.center
        local_points = self.box.global_to_local(points) * self.resolution - 0.5
        return math.grid_sample(self.values, local_points, self.extrapolation)

    def _shift_resample(self, resolution: Shape, bounds: Box, threshold=1e-5, max_padding=20):
        assert math.all_available(bounds.lower, bounds.upper), "Shift resampling requires 'bounds' to be available."
        lower = math.to_int(math.ceil(math.maximum(0, self.box.lower - bounds.lower) / self.dx - threshold))
        upper = math.to_int(math.ceil(math.maximum(0, bounds.upper - self.box.upper) / self.dx - threshold))
        total_padding = (math.sum(lower) + math.sum(upper)).numpy()
        if total_padding > max_padding:
            return NotImplemented
        elif total_padding > 0:
            from phi.field import pad
            padded = pad(self, {dim: (int(lower[i]), int(upper[i])) for i, dim in enumerate(self.shape.spatial.names)})
            grid_box, grid_resolution, grid_values = padded.box, padded.resolution, padded.values
        else:
            grid_box, grid_resolution, grid_values = self.box, self.resolution, self.values
        origin_in_local = grid_box.global_to_local(bounds.lower) * grid_resolution
        data = math.sample_subgrid(grid_values, origin_in_local, resolution)
        return data

    def closest_values(self, points: Geometry):
        assert 'vector' not in points.shape
        local_points = self.box.global_to_local(points.center) * self.resolution - 0.5
        return math.closest_grid_values(self.values, local_points, self.extrapolation)


class StaggeredGrid(Grid):
    """
    N-dimensional grid whose vector components are sampled at the respective face centers.
    A staggered grid is defined through its values tensor, its bounds describing the physical size, and its extrapolation.
    
    Staggered grids support arbitrary batch and spatial dimensions but only one channel dimension for the staggered vector components.

    Use `grid()` with `type=StaggeredGrid` to create a staggered grid.
    Alternatively, the `phi.physics.Domain` class provides convenience methods for grid creation.
    
    See the `phi.field` module documentation at https://tum-pbs.github.io/PhiFlow/Fields.html
    """

    def __init__(self, values: TensorStack, bounds: Box, extrapolation: math.Extrapolation):
        """
        Args:
            values: `phi.math.Tensor` containing all dimensions of this grid.
                Must contain a `vector` dimension with each slice consisting of one more element along the dimension they describe.
                Use `phi.math.channel_stack()` to manually create this non-uniform tensor.
            bounds: Physical size and location of the grid.
            extrapolation: The grid extrapolation determines the value outside the `values` tensor.
        """
        values = _validate_staggered_values(values)
        any_dim = values.shape.spatial.names[0]
        x = values.vector[any_dim]
        resolution = x.shape.spatial.with_size(any_dim, x.shape.get_size(any_dim) - 1)
        grids = [GridCell(resolution, bounds).extend_symmetric(dim, 1) for dim in values.shape.spatial.names]
        elements = GeometryStack(grids, 'vector_', CHANNEL_DIM)
        Grid.__init__(self, elements, values, extrapolation, resolution, bounds)

    def with_(self,
              elements: Geometry or None = None,
              values: TensorStack = None,
              extrapolation: math.Extrapolation = None,
              **other_attributes) -> 'StaggeredGrid':
        assert elements is None
        assert not other_attributes, f"Invalid attributes for type {type(self)}: {other_attributes}"
        values = _validate_staggered_values(values) if values is not None else self.values
        return StaggeredGrid(values, self.bounds, extrapolation if extrapolation is not None else self._extrapolation)

    @property
    def cells(self):
        return GridCell(self.resolution, self.bounds)

    def _sample(self, geometry: Geometry) -> Tensor:
        channels = [sample(component, geometry) for component in self.vector.unstack()]
        return math.channel_stack(channels, 'vector')

    def closest_values(self, points: Geometry):
        assert 'vector' not in points.shape
        if 'vector_' in points.shape:
            points = points.unstack('vector_')
            channels = [component.closest_values(p) for p, component in zip(points, self.vector.unstack())]
        else:
            channels = [component.closest_values(points) for component in self.vector.unstack()]
        return math.channel_stack(channels, 'vector')

    def at_centers(self) -> CenteredGrid:
        return grid(self, resolution=self.resolution, bounds=self.bounds, extrapolation=self.extrapolation)

    def __getitem__(self, item: dict):
        values = self._values[{dim: sel for dim, sel in item.items() if dim not in self.shape.spatial}]
        for dim, sel in item.items():
            if dim in self.shape.spatial:
                sel = slice(sel, sel + 1) if isinstance(sel, int) else sel
                values = []
                for vdim, val in zip(self.shape.spatial.names, self.values.unstack('vector')):
                    if vdim == dim:
                        values.append(val[{dim: slice(sel.start, sel.stop + 1)}])
                    else:
                        values.append(val[{dim: sel}])
                values = math.channel_stack(values, 'vector')
        extrapolation = self._extrapolation[item]
        bounds = GridCell(self._resolution, self._bounds)[item].bounds
        if 'vector' in item:
            if isinstance(item['vector'], int):
                dim = self.shape.spatial.names[item['vector']]
                comp_cells = GridCell(self.resolution, bounds).extend_symmetric(dim, 1)
                return CenteredGrid(values, comp_cells.bounds, extrapolation)
            else:
                assert isinstance(item['vector'], slice) and not item['vector'].start and not item['vector'].stop
        return StaggeredGrid(values, bounds, extrapolation)

    def staggered_tensor(self):
        return stack_staggered_components(self.values)

    def _op2(self, other, operator):
        if isinstance(other, StaggeredGrid) and self.bounds == other.bounds and self.shape.spatial == other.shape.spatial:
            values = operator(self._values, other.values)
            extrapolation_ = operator(self._extrapolation, other.extrapolation)
            return self.with_(values=values, extrapolation=extrapolation_)
        else:
            return SampledField._op2(self, other, operator)


def unstack_staggered_tensor(data: Tensor) -> TensorStack:
    sliced = []
    for dim, component in zip(data.shape.spatial.names, data.unstack('vector')):
        sliced.append(component[{d: slice(None, -1) for d in data.shape.spatial.without(dim).names}])
    return math.channel_stack(sliced, 'vector')


def stack_staggered_components(data: Tensor) -> Tensor:
    padded = []
    for dim, component in zip(data.shape.spatial.names, data.unstack('vector')):
        padded.append(math.pad(component, {d: (0, 1) for d in data.shape.spatial.without(dim).names}, mode=math.extrapolation.ZERO))
    return math.channel_stack(padded, 'vector')


def _validate_staggered_values(values: TensorStack):
    if 'vector' in values.shape:
        return values
    else:
        if 'staggered' in values.shape:
            return values.staggered.as_channel('vector')
        else:
            raise ValueError("values needs to have 'vector' or 'staggered' dimension")


def grid(values: Geometry or Field or Number or math.Tensor or Callable or tuple or list,
         resolution: Shape,
         bounds: Box = None,
         extrapolation: math.Extrapolation = math.extrapolation.ZERO,
         type: type = CenteredGrid) -> GridType:
    """
    Creates a `CenteredGrid` or `StaggeredGrid` from `values`.

    Args:
        values: Values to use for the grid.
            Has to be one of the following:

            * `phi.geom.Geometry`: sets inside values to 1, outside to 0
            * `Field`: resamples the Field to the staggered sample points
            * `Number`: uses the value for all sample points
            * `tuple` or `list`: interprets the sequence as vector, used for all sample points
            * `phi.math.Tensor` compatible with grid dims: uses tensor values as grid values
            * Function `values(x)` where `x` is a `phi.math.Tensor` representing the physical location.

        resolution: Grid resolution as purely spatial `phi.math.Shape`.
        bounds: Physical grid bounds as `phi.geom.Box`.
        extrapolation: Grid extrapolation as `phi.math.Extrapolation`.
        type: Grid type, either `CenteredGrid` or `StaggeredGrid`

    Returns:
        `CenteredGrid` or `StaggeredGrid`, depending on `type`.
    """
    if bounds is None:
        bounds = Box(0, math.wrap(resolution, 'vector'))
    assert resolution.spatial_rank == bounds.spatial_rank, f"Resolution {resolution} does not match bounds {bounds}"
    if isinstance(values, Geometry):
        values = HardGeometryMask(values)
    if isinstance(values, Field):
        ref_grid = grid(0, resolution=resolution, bounds=bounds, extrapolation=extrapolation, type=type)
        sampled_values = reduce_sample(values, ref_grid.elements)
        return ref_grid.with_(values=sampled_values)
    else:
        if callable(values):
            cells = GridCell(resolution, bounds)
            x = cells.center if type == CenteredGrid else cells.face_centers()
            values = values(x)
        if not isinstance(values, math.Tensor):
            values = math.tensor(values)
        if values.dtype.kind not in (float, complex):
            values = math.to_float(values)
        if type == CenteredGrid:
            values = math.expand(values, resolution)
            return CenteredGrid(values, bounds, extrapolation)
        else:
            assert type == StaggeredGrid, "type must be CenteredGrid or StaggeredGrid"
            components = values.vector.unstack(resolution.spatial_rank)
            tensors = []
            for dim, component in zip(resolution.spatial.names, components):
                comp_cells = GridCell(resolution, bounds).extend_symmetric(dim, 1)
                tensors.append(math.expand(component, comp_cells.resolution))
            return StaggeredGrid(math.channel_stack(tensors, 'vector'), bounds, extrapolation)
