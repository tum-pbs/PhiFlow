from typing import TypeVar, Any, Tuple, List, Union

from phiml.math import Solve

from phi import math, geom
from phi.geom import Box, Geometry, GridCell
from ._embed import FieldEmbedding
from ._field import SampledField, Field, sample, reduce_sample, as_extrapolation
from ..geom._stack import GeometryStack
from phiml.math import Shape, NUMPY
from phiml.math._shape import spatial, channel, parse_dim_order
from phiml.math._tensors import TensorStack, Tensor
from phiml.math.extrapolation import Extrapolation
from phiml.math.magic import slicing_dict


class Grid(SampledField):
    """
    Base class for `CenteredGrid` and `StaggeredGrid`.
    """

    def __init__(self, elements: Geometry, values: Tensor, extrapolation: Union[float, Extrapolation], resolution: Union[Shape, int], bounds: Union[Box, float]):
        assert isinstance(bounds, Box)
        assert isinstance(resolution, Shape)
        if bounds.size.vector.item_names is None:
            with NUMPY:
                bounds = bounds.shifted(math.zeros(channel(vector=spatial(values).names)))
        SampledField.__init__(self, elements, values, extrapolation, bounds)
        assert values.shape.spatial_rank == elements.spatial_rank, f"Spatial dimensions of values ({values.shape}) do not match elements {elements}"
        assert values.shape.spatial_rank == bounds.spatial_rank, f"Spatial dimensions of values ({values.shape}) do not match elements {elements}"
        assert values.shape.instance_rank == 0, f"Instance dimensions not supported for grids. Got values with shape {values.shape}"
        assert set(resolution.names) == set(bounds.vector.item_names), f"Resolution does not match bounds"
        self._resolution = resolution.only(bounds.vector.item_names, reorder=True)

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

    def _sample(self, geometry: Geometry, **kwargs) -> math.Tensor:
        raise NotImplementedError(self)

    def with_values(self, values):
        if isinstance(values, math.Tensor):
            assert set(spatial(values).names) == set(self.bounds.vector.item_names), f"StaggeredGrid.with_values() only accepts tensor with same spatial dimensiosn but got {spatial(values)} for {self}"
            return type(self)(values, extrapolation=self.extrapolation, bounds=self.bounds)
        else:
            return type(self)(values, extrapolation=self.extrapolation, bounds=self.bounds, resolution=self._resolution)

    def with_extrapolation(self, extrapolation: Extrapolation):
        return type(self)(self.values, extrapolation=extrapolation, bounds=self.bounds)

    def with_bounds(self, bounds: Box):
        return type(self)(self.values, extrapolation=self.extrapolation, bounds=bounds)

    def __value_attrs__(self):
        return '_values', '_extrapolation'

    def __variable_attrs__(self):
        return '_values',

    def __expand__(self, dims: Shape, **kwargs) -> 'Grid':
        return self.with_values(math.expand(self.values, dims, **kwargs))

    def __replace_dims__(self, dims: Tuple[str, ...], new_dims: Shape, **kwargs) -> 'Grid':
        for dim in dims:
            if dim in self._resolution:
                return NotImplemented
        values = math.rename_dims(self.values, dims, new_dims)
        extrapolation = math.rename_dims(self.extrapolation, dims, new_dims, **kwargs)
        bounds = math.rename_dims(self.bounds, dims, new_dims, **kwargs)
        return type(self)(values, extrapolation=extrapolation, bounds=bounds, resolution=self._resolution)


    def __eq__(self, other):
        if not type(self) == type(other):
            return False
        if not (self._bounds == other._bounds and self._resolution == other._resolution and self._extrapolation == other._extrapolation):
            return False
        if self.values is None:
            return other.values is None
        if other.values is None:
            return False
        if not math.all_available(self.values) or not math.all_available(other.values):  # tracers involved
            if math.all_available(self.values) != math.all_available(other.values):
                return False
            else:  # both tracers
                return self.values.shape == other.values.shape
        if self.values.shape != other.values.shape:
            return False
        return bool((self.values == other.values).all)

    def __getitem__(self, item) -> 'Grid':
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
        return self.bounds.size / self.resolution

    def __repr__(self):
        if self._values is not None:
            return f"{self.__class__.__name__}[{self.shape.non_spatial & self.resolution}, size={self.box.size}, extrapolation={self._extrapolation}]"
        else:
            return f"{self.__class__.__name__}[{self.resolution}, size={self.box.size}, extrapolation={self._extrapolation}]"

    def uniform_values(self):
        """
        Returns a uniform tensor containing `values`.

        For periodic grids, which always have a uniform value tensor, `values' is returned directly.
        If `values` is not uniform, it is padded as in `StaggeredGrid.staggered_tensor()`.
        """
        return self.values


GridType = TypeVar('GridType', bound=Grid)


class CenteredGrid(Grid):
    """
    N-dimensional grid with values sampled at the cell centers.
    A centered grid is defined through its `CenteredGrid.values` `phiml.math.Tensor`, its `CenteredGrid.bounds` `phi.geom.Box` describing the physical size, and its `CenteredGrid.extrapolation` (`phiml.math.extrapolation.Extrapolation`).
    
    Centered grids support batch, spatial and channel dimensions.

    See Also:
        `StaggeredGrid`,
        `Grid`,
        `SampledField`,
        `Field`,
        module documentation at https://tum-pbs.github.io/PhiFlow/Fields.html
    """

    def __init__(self,
                 values: Any = 0.,
                 extrapolation: Union[float, Extrapolation, dict, Field] = 0.,
                 bounds: Union[Box, float] = None,
                 resolution: Union[int, Shape] = None,
                 **resolution_: Union[int, Tensor]):
        """
        Args:
            values: Values to use for the grid.
                Has to be one of the following:

                * `phi.geom.Geometry`: sets inside values to 1, outside to 0
                * `Field`: resamples the Field to the staggered sample points
                * `Number`: uses the value for all sample points
                * `tuple` or `list`: interprets the sequence as vector, used for all sample points
                * `phiml.math.Tensor` compatible with grid dims: uses tensor values as grid values
                * Function `values(x)` where `x` is a `phiml.math.Tensor` representing the physical location.
                    The spatial dimensions of the grid will be passed as batch dimensions to the function.

            extrapolation: The grid extrapolation determines the value outside the `values` tensor.
                Allowed types: `float`, `phiml.math.Tensor`, `phiml.math.extrapolation.Extrapolation`.
            bounds: Physical size and location of the grid as `phi.geom.Box`.
                If the resolution is determined through `resolution` of `values`, a `float` can be passed for `bounds` to create a unit box.
            resolution: Grid resolution as purely spatial `phiml.math.Shape`.
                If `bounds` is given as a `Box`, the resolution may be specified as an `int` to be equal along all axes.
            **resolution_: Spatial dimensions as keyword arguments. Typically either `resolution` or `spatial_dims` are specified.
        """
        if resolution is None and not resolution_:
            assert isinstance(values, math.Tensor), "Grid resolution must be specified when 'values' is not a Tensor."
            resolution = values.shape.spatial
            bounds = _get_bounds(bounds, resolution)
            elements = GridCell(resolution, bounds)
        else:
            resolution = _get_resolution(resolution, resolution_, bounds)
            bounds = _get_bounds(bounds, resolution)
            elements = GridCell(resolution, bounds)
            if isinstance(values, math.Tensor):
                values = math.expand(values, resolution)
            elif isinstance(values, Geometry):
                values = reduce_sample(values, elements)
            elif isinstance(values, Field):
                values = reduce_sample(values, elements)
            elif callable(values):
                values = _sample_function(values, elements)
            else:
                if isinstance(values, (tuple, list)) and len(values) == resolution.rank:
                    values = math.tensor(values, channel(vector=resolution.names))
                values = math.expand(math.tensor(values), resolution)
        if values.dtype.kind not in (float, complex):
            values = math.to_float(values)
        assert resolution.spatial_rank == bounds.spatial_rank, f"Resolution {resolution} does not match bounds {bounds}"
        Grid.__init__(self, elements, values, extrapolation, values.shape.spatial, bounds)

    def __getitem__(self, item):
        item = slicing_dict(self, item)
        if not item:
            return self
        values = self._values[item]
        extrapolation = self._extrapolation[item]
        keep_dims = [dim for dim in self.resolution.names if dim not in item or not isinstance(item[dim], int)]
        bounds = self.elements[item].bounds[{'vector': keep_dims}]
        return CenteredGrid(values, bounds=bounds, extrapolation=extrapolation)

    def _sample(self, geometry: Geometry, **kwargs) -> Tensor:
        if geometry == self.bounds:
            return math.mean(self._values, self._resolution)
        if isinstance(geometry, GeometryStack):
            sampled = [self._sample(g, **kwargs) for g in geometry.geometries]
            return math.stack(sampled, geometry.geometries.shape)
        if isinstance(geometry, GridCell):
            if self.elements == geometry:
                return self.values
            elif math.close(self.dx, geometry.size):
                if all([math.close(offset, geometry.half_size) or math.close(offset, 0)
                        for offset in math.abs(self.bounds.lower - geometry.bounds.lower)]):
                    dyadic_interpolated = self._dyadic_interplate(geometry.resolution, geometry.bounds, **kwargs)
                    if dyadic_interpolated is not NotImplemented:
                        return dyadic_interpolated
                if 'order' in kwargs and kwargs['order'] != 2:
                    raise NotImplementedError(f"Only 6th-order implicit and 2nd-order resampling supported but got order={kwargs['order']}")
                fast_resampled = self._shift_resample(geometry.resolution, geometry.bounds)
                if fast_resampled is not NotImplemented:
                    return fast_resampled
        points = geometry.center
        local_points = self.box.global_to_local(points) * self.resolution - 0.5
        resampled_values = math.grid_sample(self.values, local_points, self.extrapolation, bounds=self.bounds)
        if isinstance(self._extrapolation, FieldEmbedding):
            if isinstance(geometry, GridCell) and ((geometry.bounds.upper <= self.bounds.upper).all or (geometry.bounds.lower >= self.bounds.lower).all):
                # geometry is a subgrid of self
                return resampled_values
            else:  # otherwise we also sample the extrapolation Field
                ext_values = self._extrapolation.field._sample(geometry, **kwargs)
                inside = self.bounds.lies_inside(points)
                return math.where(inside, resampled_values, ext_values)
        return resampled_values

    def _dyadic_interplate(self, resolution: Shape, bounds: Box, order=2, implicit: Solve = None):
        offsets = bounds.lower - self.bounds.lower
        interpolation_dirs = [0 if math.close(offset, 0) else int(math.sign(offset)) for offset in offsets]
        return _dyadic_interpolate(self.values, interpolation_dirs, self.extrapolation, order, implicit)

    def _shift_resample(self, resolution: Shape, bounds: Box, threshold=1e-5, max_padding=20):
        assert math.all_available(bounds.lower, bounds.upper), "Shift resampling requires 'bounds' to be available."
        lower = math.to_int32(math.ceil(math.maximum(0, self.box.lower - bounds.lower) / self.dx - threshold))
        upper = math.to_int32(math.ceil(math.maximum(0, bounds.upper - self.box.upper) / self.dx - threshold))
        total_padding = (math.sum(lower) + math.sum(upper)).numpy()
        if total_padding > max_padding and self.extrapolation.native_grid_sample_mode:
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
                 values: Any = 0.,
                 extrapolation: Union[float, Extrapolation, dict, Field] = 0,
                 bounds: Union[Box, float] = None,
                 resolution: Union[Shape, int] = None,
                 **resolution_: Union[int, Tensor]):
        """
        Args:
            values: Values to use for the grid.
                Has to be one of the following:

                * `phi.geom.Geometry`: sets inside values to 1, outside to 0
                * `Field`: resamples the Field to the staggered sample points
                * `Number`: uses the value for all sample points
                * `tuple` or `list`: interprets the sequence as vector, used for all sample points
                * `phiml.math.Tensor` with staggered shape: uses tensor values as grid values.
                  Must contain a `vector` dimension with each slice consisting of one more element along the dimension they describe.
                  Use `phiml.math.stack()` to manually create this non-uniform tensor.
                * Function `values(x)` where `x` is a `phiml.math.Tensor` representing the physical location.
                    The spatial dimensions of the grid will be passed as batch dimensions to the function.

            extrapolation: The grid extrapolation determines the value outside the `values` tensor.
                Allowed types: `float`, `phiml.math.Tensor`, `phiml.math.extrapolation.Extrapolation`.
            bounds: Physical size and location of the grid as `phi.geom.Box`.
                If the resolution is determined through `resolution` of `values`, a `float` can be passed for `bounds` to create a unit box.
            resolution: Grid resolution as purely spatial `phiml.math.Shape`.
                If `bounds` is given as a `Box`, the resolution may be specified as an `int` to be equal along all axes.
            **resolution_: Spatial dimensions as keyword arguments. Typically either `resolution` or `spatial_dims` are specified.
        """
        extrapolation = as_extrapolation(extrapolation)
        if resolution is None and not resolution_:
            assert isinstance(values, Tensor), "Grid resolution must be specified when 'values' is not a Tensor."
            if not all(extrapolation.valid_outer_faces(d)[0] != extrapolation.valid_outer_faces(d)[1] for d in spatial(values).names):  # non-uniform values required
                if values.shape.is_uniform:
                    values = unstack_staggered_tensor(values, extrapolation)
                resolution = resolution_from_staggered_tensor(values, extrapolation)
            else:
                resolution = spatial(values)
            bounds = _get_bounds(bounds, resolution)
            bounds = bounds or Box(math.const_vec(0, resolution), math.wrap(resolution, channel('vector')))
            elements = staggered_elements(resolution, bounds, extrapolation)
        else:
            resolution = _get_resolution(resolution, resolution_, bounds)
            bounds = _get_bounds(bounds, resolution)
            elements = staggered_elements(resolution, bounds, extrapolation)
            if isinstance(values, math.Tensor):
                if not spatial(values):
                    values = expand_staggered(values, resolution, extrapolation)
                if not all(extrapolation.valid_outer_faces(d)[0] != extrapolation.valid_outer_faces(d)[1] for d in resolution.names):  # non-uniform values required
                    if values.shape.is_uniform:
                        values = unstack_staggered_tensor(values, extrapolation)
                    else:  # Keep dim order from data and check it matches resolution
                        assert set(resolution_from_staggered_tensor(values, extrapolation)) == set(resolution), f"Failed to create StaggeredGrid: values {values.shape} do not match given resolution {resolution} for extrapolation {extrapolation}. See https://tum-pbs.github.io/PhiFlow/Staggered_Grids.html"
            elif isinstance(values, Geometry):
                values = reduce_sample(values, elements)
            elif isinstance(values, Field):
                values = reduce_sample(values, elements)
            elif callable(values):
                values = _sample_function(values, elements)
                if elements.shape.shape.rank > 1:  # Different number of X and Y faces
                    assert isinstance(values, TensorStack), f"values function must return a staggered Tensor but returned {type(values)}"
                assert 'staggered_direction' in values.shape
                if 'vector' in values.shape:
                    values = math.stack([values.staggered_direction[i].vector[i] for i in range(resolution.rank)], channel(vector=resolution))
                else:
                    values = values.staggered_direction.as_channel('vector')
            else:
                values = expand_staggered(math.tensor(values), resolution, extrapolation)
        if values.dtype.kind not in (float, complex):
            values = math.to_float(values)
        assert resolution.spatial_rank == bounds.spatial_rank, f"Resolution {resolution} does not match bounds {bounds}"
        Grid.__init__(self, elements, values, extrapolation, resolution, bounds)

    @property
    def cells(self):
        return GridCell(self.resolution, self.bounds)

    def with_extrapolation(self, extrapolation: Extrapolation):
        extrapolation = as_extrapolation(extrapolation)
        if all([extrapolation.valid_outer_faces(dim) == self.extrapolation.valid_outer_faces(dim) for dim in self.resolution.names]):
            return StaggeredGrid(self.values, extrapolation=extrapolation, bounds=self.bounds)
        else:
            values = []
            for dim, component in zip(self.shape.spatial.names, self.values.vector):
                old_lo, old_hi = [int(v) for v in self.extrapolation.valid_outer_faces(dim)]
                new_lo, new_hi = [int(v) for v in extrapolation.valid_outer_faces(dim)]
                widths = (new_lo - old_lo, new_hi - old_hi)
                values.append(math.pad(component, {dim: widths}, self.extrapolation, bounds=self.bounds))
            values = math.stack(values, channel(vector=self.resolution))
            return StaggeredGrid(values, extrapolation, bounds=self.bounds)

    def _sample(self, geometry: Geometry, **kwargs) -> Tensor:
        channels = [sample(component, geometry, **kwargs) for component in self.vector.unstack()]
        return math.stack(channels, geometry.shape['vector'])

    def closest_values(self, points: Geometry):
        if 'staggered_direction' in points.shape:
            points_ = points.unstack('staggered_direction')
            channels = [component.closest_values(p) for p, component in zip(points_, self.vector.unstack())]
        else:
            channels = [component.closest_values(points) for component in self.vector.unstack()]
        return math.stack(channels, points.shape['vector'])

    def at_centers(self) -> CenteredGrid:
        """
        Interpolates the staggered values to the cell centers.

        Returns:
            `CenteredGrid` sampled at cell centers.
        """
        return CenteredGrid(self, resolution=self.resolution, bounds=self.bounds, extrapolation=self.extrapolation)

    def __getitem__(self, item):
        item = slicing_dict(self, item)
        if not item:
            return self
        if 'vector' in item:
            selection = item['vector']
            if isinstance(selection, int):
                item['vector'] = self.resolution.names[selection]
        values = self._values[{dim: sel for dim, sel in item.items() if dim not in self.shape.spatial}]
        for dim, sel in item.items():
            if dim in self.shape.spatial:
                raise AssertionError("Cannot slice StaggeredGrid along spatial dimensions.")
                # sel = slice(sel, sel + 1) if isinstance(sel, int) else sel
                # values = []
                # for vdim, val in zip(self.shape.spatial.names, self.values.unstack('vector')):
                #     if vdim == dim:
                #         values.append(val[{dim: slice(sel.start, sel.stop + 1)}])
                #     else:
                #         values.append(val[{dim: sel}])
                # values = math.stack(values, channel('vector'))
        extrapolation = self._extrapolation[item]
        bounds = GridCell(self._resolution, self._bounds)[item].bounds
        if 'vector' in item:
            selection = item['vector']
            if isinstance(selection, str) and ',' in selection:
                selection = parse_dim_order(selection)
            if isinstance(selection, str):  # single item name
                item_names = self.shape.get_item_names('vector', fallback_spatial=True)
                assert selection in item_names, f"Accessing field.vector['{selection}'] failed. Item names are {item_names}."
                selection = item_names.index(selection)
            if isinstance(selection, int):
                dim = self.resolution.names[selection]
                comp_cells = GridCell(self.resolution, bounds).stagger(dim, *self.extrapolation.valid_outer_faces(dim))
                return CenteredGrid(values, bounds=comp_cells.bounds, extrapolation=extrapolation)
            else:
                assert isinstance(selection, slice) and not selection.start and not selection.stop
        return StaggeredGrid(values, bounds=bounds, extrapolation=extrapolation)

    def uniform_values(self):
        if self.values.shape.is_uniform:
            return self.values
        else:
            return self.staggered_tensor()

    def staggered_tensor(self) -> Tensor:
        """
        Stacks all component grids into a single uniform `phiml.math.Tensor`.
        The individual components are padded to a common (larger) shape before being stacked.
        The shape of the returned tensor is exactly one cell larger than the grid `resolution` in every spatial dimension.

        Returns:
            Uniform `phiml.math.Tensor`.
        """
        padded = []
        for dim, component in zip(self.resolution.names, math.unstack(self.values, 'vector')):
            widths = {d: (0, 1) for d in self.resolution.names}
            lo_valid, up_valid = self.extrapolation.valid_outer_faces(dim)
            widths[dim] = (int(not lo_valid), int(not up_valid))
            padded.append(math.pad(component, widths, self.extrapolation[{'vector': dim}], bounds=self.bounds))
        result = math.stack(padded, channel(vector=self.resolution))
        assert result.shape.is_uniform
        return result

    def _op2(self, other, operator):
        if isinstance(other, StaggeredGrid) and self.bounds == other.bounds and self.shape.spatial == other.shape.spatial:
            values = operator(self._values, other.values)
            extrapolation_ = operator(self._extrapolation, other.extrapolation)
            return StaggeredGrid(values=values, extrapolation=extrapolation_, bounds=self.bounds)
        else:
            return SampledField._op2(self, other, operator)


def unstack_staggered_tensor(data: Tensor, extrapolation: Extrapolation) -> TensorStack:
    sliced = []
    for dim, component in zip(data.shape.spatial.names, data.vector):
        lo_valid, up_valid = extrapolation.valid_outer_faces(dim)
        slices = {d: slice(0, -1) for d in data.shape.spatial.names}
        slices[dim] = slice(int(not lo_valid), - int(not up_valid) or None)
        sliced.append(component[slices])
    return math.stack(sliced, channel(vector=spatial(data)))


def staggered_elements(resolution: Shape, bounds: Box, extrapolation: Extrapolation):
    cells = GridCell(resolution, bounds)
    grids = {}
    for dim in bounds.vector.item_names:
        lower, upper = extrapolation.valid_outer_faces(dim)
        grids[dim] = cells.stagger(dim, lower, upper)
    return geom.stack(grids, channel('staggered_direction'))


def expand_staggered(values: Tensor, resolution: Shape, extrapolation: Extrapolation):
    """ Add missing spatial dimensions to `values` """
    cells = GridCell(resolution, Box(math.const_vec(0, resolution), math.const_vec(1, resolution)))
    components = values.vector.unstack(resolution.spatial_rank)
    tensors = []
    for dim, component in zip(resolution.spatial.names, components):
        comp_cells = cells.stagger(dim, *extrapolation.valid_outer_faces(dim))
        tensors.append(math.expand(component, comp_cells.resolution))
    return math.stack(tensors, channel(vector=resolution.names))


def resolution_from_staggered_tensor(values: Tensor, extrapolation: Extrapolation):
    any_dim = values.shape.spatial.names[0]
    x_shape = values.shape.after_gather({'vector': any_dim})
    ext_lower, ext_upper = extrapolation.valid_outer_faces(any_dim)
    delta = int(ext_lower) + int(ext_upper) - 1
    resolution = x_shape.spatial._replace_single_size(any_dim, x_shape.get_size(any_dim) - delta)
    return resolution


def _sample_function(f, elements: Geometry):
    from phiml.math._functional import get_function_parameters
    try:
        params = get_function_parameters(f)
        dims = elements.shape.get_size('vector')
        names_match = tuple(params.keys())[:dims] == elements.shape.get_item_names('vector')
        num_positional = 0
        has_varargs = False
        for n, p in params.items():
            if p.default is p.empty:
                num_positional += 1
            if p.kind == 2:  # _ParameterKind.VAR_POSITIONAL
                has_varargs = True
        assert num_positional <= dims, f"Cannot sample {f.__name__}({', '.join(tuple(params))}) on physical space {elements.shape.get_item_names('vector')}"
        pass_varargs = has_varargs or names_match or num_positional > 1 or num_positional == dims
        if num_positional > 1 and not has_varargs:
            assert names_match, f"Positional arguments of {f.__name__}({', '.join(tuple(params))}) should match physical space {elements.shape.get_item_names('vector')}"
    except ValueError as err:  # signature not available for all functions
        pass_varargs = False
    if pass_varargs:
        values = math.map_s2b(f)(*elements.center.vector)
    else:
        values = math.map_s2b(f)(elements.center)
    assert isinstance(values, math.Tensor), f"values function must return a Tensor but returned {type(values)}"
    return values


def _get_bounds(bounds: Union[Box, float, None], resolution: Shape):
    if bounds is None:
        return Box(math.const_vec(0, resolution), math.wrap(resolution, channel(vector=resolution.names)))
    if isinstance(bounds, Box):
        assert set(bounds.vector.item_names) == set(resolution.names), f"bounds dimensions {bounds.vector.item_names} must match resolution {resolution}"
        return bounds
    if isinstance(bounds, (int, float)):
        return Box(math.const_vec(0, resolution), math.const_vec(bounds, resolution))
    raise ValueError(f"bounds must be a Box, float or None but got {type(bounds).__name__}")


def _get_resolution(resolution: Shape, resolution_: dict, bounds: Box):
    assert 'boundaries' not in resolution_, "'boundaries' is not a valid grid argument. Use 'extrapolation' instead, passing a value or math.extrapolation.Extrapolation object. See https://tum-pbs.github.io/PhiFlow/phi/math/extrapolation.html"
    if isinstance(resolution, int):
        assert not resolution_, "Cannot specify keyword resolution and integer resolution at the same time."
        resolution = spatial(**{dim: resolution for dim in bounds.size.shape.get_item_names('vector')})
    try:
        resolution_ = spatial(**resolution_)
    except AssertionError as err:
        raise ValueError(f"Invalid grid resolution: {', '.join(f'{dim}={size}' for dim, size in resolution_.items())}. Pass an int for all sizes.") from err
    return (resolution or math.EMPTY_SHAPE) & resolution_


def _dyadic_interpolate(grid: Tensor, interpolation_dirs: List, padding: Extrapolation, order: int, implicit: Solve):
    """
    Samples a sub-grid from `grid` with an offset of half a grid cell in directions defined by `interpolation_dirs`.

    Args:
        grid: `Tensor` to be resampled.
        interpolation_dirs: List which defines for every spatial dimension of `grid` if interpolation should be performed,
            in positive direction `1` / negative direction `-1` / no interpolation`0`
            len(interpolation_dirs) == len(grid.shape.spatial.names) is assumed
            Example: With `grid.shape.spatial.names=['x', 'y']` and `interpolation_dirs: [1, -1]`
                     grid will be interpolated half a grid cell in positive x direction and half a grid cell in negative y direction
        padding: Extrapolation used for the needed out of Domain values
        order: finite difference `Scheme` used for interpolation

    Returns:
      Sub-grid as `Tensor`
    """
    if implicit:
        if order == 6:
            values, needed_shifts = [1 / 20, 3 / 4, 3 / 4, 1 / 20], (-1, 0, 1, 2)
            values_rhs, needed_shifts_rhs = [3 / 10, 1, 3 / 10], (-1, 0, 1)
        else:
            return NotImplemented
    else:
        return NotImplemented
    result = grid
    for dim, dir in zip(grid.shape.spatial.names, interpolation_dirs):
        if dir == 0: continue
        is_neg_dir = dir == -1
        current_widths = [abs(min(needed_shifts)) + is_neg_dir, max(needed_shifts) - is_neg_dir]
        padded = math.pad(result, {dim: tuple(current_widths)}, padding)
        shifted = math.shift(padded, needed_shifts, [dim], padding=None, stack_dim=None)
        result = sum([value * shift_ for value, shift_ in zip(values, shifted)])
        if implicit:
            implicit.x0 = result
            result = math.solve_linear(dyadic_interpolate_lhs, result, implicit, values_rhs=values_rhs, needed_shifts_rhs=needed_shifts_rhs, dim=dim, padding=padding)
    return result


@math.jit_compile_linear(auxiliary_args="values_rhs, needed_shifts_rhs")
def dyadic_interpolate_lhs(x, values_rhs, needed_shifts_rhs, dim, padding):
    shifted = math.shift(x, needed_shifts_rhs, stack_dim=None, dims=[dim], padding=padding)
    return sum([value * shift_ for value, shift_ in zip(values_rhs, shifted)])
