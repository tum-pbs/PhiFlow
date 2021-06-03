from numbers import Number
from typing import TypeVar, Any

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

    def __value_attrs__(self):
        return '_values', '_extrapolation'

    def __variable_attrs__(self):
        return '_values',

    def __eq__(self, other):
        if not type(self) == type(other):
            return False
        assert self._values is None, "Can only compare grids in key mode."
        return self._bounds == other._bounds and self._resolution == other._resolution and self._extrapolation == other._extrapolation

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
        if self._values is not None:
            return f"{self.__class__.__name__}[{self.shape.non_spatial & self.resolution}, size={self.box.size}, extrapolation={self._extrapolation}]"
        else:
            return f"{self.__class__.__name__}[{self.resolution}, size={self.box.size}, extrapolation={self._extrapolation}]"


GridType = TypeVar('GridType', bound=Grid)


class CenteredGrid(Grid):
    """
    N-dimensional grid with values sampled at the cell centers.
    A centered grid is defined through its `CenteredGrid.values` `phi.math.Tensor`, its `CenteredGrid.bounds` `phi.geom.Box` describing the physical size, and its `CenteredGrid.extrapolation` (`phi.math.Extrapolation`).
    
    Centered grids support batch, spatial and channel dimensions.

    See Also:
        `StaggeredGrid`,
        `Grid`,
        `SampledField`,
        `Field`,
        module documentation at https://tum-pbs.github.io/PhiFlow/Fields.html
    """

    def __init__(self,
                 values: Any,
                 extrapolation: math.Extrapolation or math.Tensor,
                 bounds: Box = None,
                 resolution: Shape = None,
                 **resolution_: int or Tensor):
        """
        Args:
            values: Values to use for the grid.
                Has to be one of the following:

                * `phi.geom.Geometry`: sets inside values to 1, outside to 0
                * `Field`: resamples the Field to the staggered sample points
                * `Number`: uses the value for all sample points
                * `tuple` or `list`: interprets the sequence as vector, used for all sample points
                * `phi.math.Tensor` compatible with grid dims: uses tensor values as grid values
                * Function `values(x)` where `x` is a `phi.math.Tensor` representing the physical location.

            extrapolation: The grid extrapolation determines the value outside the `values` tensor.
            bounds: Physical size and location of the grid as `phi.geom.Box`.
            resolution: Grid resolution as purely spatial `phi.math.Shape`.
            **resolution_: Spatial dimensions as keyword arguments. Typically either `resolution` or `spatial_dims` are specified.
        """
        if resolution is None and not resolution_:
            assert isinstance(values, math.Tensor), "Grid resolution must be specified when 'values' is not a Tensor."
            resolution = values.shape.spatial
            bounds = bounds or Box(0, math.wrap(resolution, 'vector'))
            elements = GridCell(resolution, bounds)
        else:
            resolution = (resolution or math.EMPTY_SHAPE) & math.spatial_shape(resolution_)
            bounds = bounds or Box(0, math.wrap(resolution, 'vector'))
            elements = GridCell(resolution, bounds)
            if isinstance(values, math.Tensor):
                values = math.expand(values, resolution)
            elif isinstance(values, Geometry):
                values = reduce_sample(HardGeometryMask(values), elements)
            elif isinstance(values, Field):
                values = reduce_sample(values, elements)
            elif callable(values):
                values = values(elements.center)
                assert isinstance(values, math.Tensor), f"values function must return a Tensor but returned {type(values)}"
            else:
                values = math.expand(math.tensor(values), resolution)
        if values.dtype.kind not in (float, complex):
            values = math.to_float(values)
        assert resolution.spatial_rank == bounds.spatial_rank, f"Resolution {resolution} does not match bounds {bounds}"
        if not isinstance(extrapolation, math.Extrapolation):
            extrapolation = math.extrapolation.ConstantExtrapolation(math.tensor(extrapolation))
        Grid.__init__(self, elements, values, extrapolation, values.shape.spatial, bounds)

    def with_(self,
              elements: Geometry or None = None,
              values: Tensor = None,
              extrapolation: math.Extrapolation = None,
              **other_attributes) -> 'CenteredGrid':
        assert elements is None
        assert not other_attributes, f"Invalid attributes for type {type(self)}: {other_attributes}"
        values = values if values is not None else self.values
        extrapolation = extrapolation if extrapolation is not None else self._extrapolation
        return CenteredGrid(values, extrapolation, self.bounds)

    def __getitem__(self, item: dict):
        values = self._values[{dim: slice(sel, sel + 1) if isinstance(sel, int) and dim in self.shape.spatial else sel for dim, sel in item.items()}]
        extrapolation = self._extrapolation[item]
        bounds = self.elements[item].bounds
        return CenteredGrid(values, bounds=bounds, extrapolation=extrapolation)

    def _sample(self, geometry: Geometry) -> Tensor:
        if geometry == self.bounds:
            return math.mean(self._values, self._resolution)
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
        lower = math.to_int32(math.ceil(math.maximum(0, self.box.lower - bounds.lower) / self.dx - threshold))
        upper = math.to_int32(math.ceil(math.maximum(0, bounds.upper - self.box.upper) / self.dx - threshold))
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
    
    Staggered grids support batch and spatial dimensions but only one channel dimension for the staggered vector components.


    See Also:
        `CenteredGrid`,
        `Grid`,
        `SampledField`,
        `Field`,
        module documentation at https://tum-pbs.github.io/PhiFlow/Fields.html
    """

    def __init__(self,
                 values: Any,
                 extrapolation: Number or math.Extrapolation or math.Tensor,
                 bounds: Box = None,
                 resolution: Shape = None,
                 **resolution_: int or Tensor):
        """
        Args:
            values: Values to use for the grid.
                Has to be one of the following:

                * `phi.geom.Geometry`: sets inside values to 1, outside to 0
                * `Field`: resamples the Field to the staggered sample points
                * `Number`: uses the value for all sample points
                * `tuple` or `list`: interprets the sequence as vector, used for all sample points
                * `phi.math.Tensor` with staggered shape: uses tensor values as grid values.
                  Must contain a `vector` dimension with each slice consisting of one more element along the dimension they describe.
                  Use `phi.math.channel_stack()` to manually create this non-uniform tensor.
                * Function `values(x)` where `x` is a `phi.math.Tensor` representing the physical location.

            extrapolation: The grid extrapolation determines the value outside the `values` tensor.
            bounds: Physical size and location of the grid.
            resolution: Grid resolution as purely spatial `phi.math.Shape`.
            **resolution_: Spatial dimensions as keyword arguments. Typically either `resolution` or `spatial_dims` are specified.
        """
        if resolution is None and not resolution_:
            assert isinstance(values, TensorStack), "Grid resolution must be specified when 'values' is not a Tensor."
            values = _validate_staggered_values(values)
            any_dim = values.shape.spatial.names[0]
            x = values.vector[any_dim]
            resolution = x.shape.spatial.with_size(any_dim, x.shape.get_size(any_dim) - 1)
            bounds = bounds or Box(0, math.wrap(resolution, 'vector'))
            elements = staggered_elements(resolution, bounds)
        else:
            resolution = (resolution or math.EMPTY_SHAPE) & math.spatial_shape(resolution_)
            bounds = bounds or Box(0, math.wrap(resolution, 'vector'))
            elements = staggered_elements(resolution, bounds)
            if isinstance(values, math.Tensor):
                values = expand_staggered(values, resolution)
            elif isinstance(values, Geometry):
                values = reduce_sample(HardGeometryMask(values), elements)
            elif isinstance(values, Field):
                values = reduce_sample(values, elements)
            elif callable(values):
                values = values(elements.center)
                assert isinstance(values, TensorStack), f"values function must return a staggered Tensor but returned {type(values)}"
            else:
                values = expand_staggered(math.tensor(values), resolution)
        if values.dtype.kind not in (float, complex):
            values = math.to_float(values)
        assert resolution.spatial_rank == bounds.spatial_rank, f"Resolution {resolution} does not match bounds {bounds}"
        if not isinstance(extrapolation, math.Extrapolation):
            extrapolation = math.extrapolation.ConstantExtrapolation(math.tensor(extrapolation))
        Grid.__init__(self, elements, values, extrapolation, resolution, bounds)

    def with_(self,
              elements: Geometry or None = None,
              values: TensorStack = None,
              extrapolation: math.Extrapolation = None,
              **other_attributes) -> 'StaggeredGrid':
        assert elements is None
        assert not other_attributes, f"Invalid attributes for type {type(self)}: {other_attributes}"
        values = _validate_staggered_values(values) if values is not None else self.values
        return StaggeredGrid(values, bounds=self.bounds, extrapolation=extrapolation if extrapolation is not None else self._extrapolation)

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
        return CenteredGrid(self, resolution=self.resolution, bounds=self.bounds, extrapolation=self.extrapolation)

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
                return CenteredGrid(values, bounds=comp_cells.bounds, extrapolation=extrapolation)
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


def staggered_elements(resolution: Shape, bounds: Box):
    grids = [GridCell(resolution, bounds).extend_symmetric(dim, 1) for dim in resolution.names]
    return GeometryStack(grids, 'vector_', CHANNEL_DIM)


def expand_staggered(values: Tensor, resolution: Shape):
    bounds = Box(0, [1] * resolution.rank)
    components = values.vector.unstack(resolution.spatial_rank)
    tensors = []
    for dim, component in zip(resolution.spatial.names, components):
        comp_cells = GridCell(resolution, bounds).extend_symmetric(dim, 1)
        tensors.append(math.expand(component, comp_cells.resolution))
    return math.channel_stack(tensors, 'vector')
