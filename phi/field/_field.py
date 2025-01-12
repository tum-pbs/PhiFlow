import warnings
from dataclasses import dataclass
from numbers import Number
from typing import Callable, Union, Tuple, Optional

from phi import math
from phi.geom import Geometry, Box, Point, BaseBox, UniformGrid, Mesh, Sphere, Graph
from phi.geom._geom import slice_off_constant_faces
from phi.math import Shape, Tensor, channel, non_batch, expand, instance, spatial, wrap, dual, non_dual
from phi.math.extrapolation import Extrapolation
from phi.math.magic import BoundDim, slicing_dict
from phiml.dataclasses import sliceable
from phiml.math import batch, Solve, DimFilter, unstack, concat_shapes, pack_dims, shape
from phiml.math.extrapolation import domain_slice


class FieldInitializer:

    def _sample(self, geometry: Geometry, at: str, boundaries: Extrapolation, **kwargs) -> math.Tensor:
        """ For internal use only. Use `sample()` instead. """
        raise NotImplementedError(self)


class _FieldType(type):
    """Metaclass for the Field constructor."""
    def __call__(cls,
                 geometry: Union[Geometry, Tensor],
                 values: Union[Tensor, Number, bool, Callable, FieldInitializer, Geometry, 'Field'],
                 boundary: Union[Number, Extrapolation, 'Field', dict] = 0.,
                 variable_attrs=('values',),
                 value_attrs=('values',),
                 **sampling_kwargs):
        assert isinstance(geometry, Geometry), f"geometry must be a Geometry object but got {type(geometry).__name__}"
        boundary = as_boundary(boundary, geometry)
        if values is not None:
            if isinstance(values, (Tensor, Number, bool)):
                values = wrap(values)
            else:
                from ._resample import sample
                values = sample(values, geometry, 'center', boundary, **sampling_kwargs)
            matching_sets = [s for s, s_shape in geometry.sets.items() if s_shape in values.shape]
            if not matching_sets:
                values = expand(wrap(values), non_batch(geometry) - 'vector')
        result = cls.__new__(cls, geometry, values, boundary, variable_attrs, value_attrs)
        result.__init__(geometry, values, boundary, variable_attrs, value_attrs)  # also calls __post_init__()
        return result


@sliceable
@dataclass(frozen=True)
class Field(metaclass=_FieldType):
    """
    A `Field` represents a discretized physical quantity (like temperature field or velocity field).
    The sample points and their relation are encoded in the `geometry` property and the corresponding values are stored as one `Tensor` in `values`.
    The boundary conditions and values outside the geometry are determined by `boundary`.

    Examples:
        Create a periodic 2D grid, initialized via noise fluctuations.
        >>> Field(UniformGrid(x=32, y=32), values=Noise(), boundary=PERIODIC)

        Create a field on an unstructured mesh loaded from a .gmsh file
        >>> mesh = phi.geom.load_gmsh('cylinder.msh', ('y-', 'x+', 'y+', 'x-', 'cyl+', 'cyl-'))
        >>> Field(mesh, values=vec(x=1, y=0), boundary={'x': ZERO_GRADIENT, 'y': 0, 'cyl': 0})

        Create two cubes and compute a scalar values for each.
        >>> Field(Cuboid(vec(x=[0, 2], y=0), x=1, y=1), values=lambda x,y: x)

    See the `phi.field` module documentation at https://tum-pbs.github.io/PhiFlow/Fields.html
    """

    geometry: Geometry
    """ Discretization `Geometry`. This determines where in space the `values` are sampled as well as their relationship and interpretation. """
    values: Tensor
    """ The sampled values, matching some point set of `geometry`, e.g. center points, see `Geometry.sets`."""
    boundary: Extrapolation = 0.
    """ Boundary conditions describe the values outside of `geometry` and are used by numerical solvers to compute edge values. """

    variable_attrs: Tuple[str, ...] = ('values',)
    """ Which of the three attributes (geometry,values,boundary) should be traced / optimized. See `phiml.math.magic.PhiTreeNode.__variable_attrs__`"""
    value_attrs: Tuple[str, ...] = ('values',)
    """ Which of the three attributes (geometry,values,boundary) are considered values. See `phiml.math.magic.PhiTreeNode.__value_attrs__`"""

    def __post_init__(self):
        math.merge_shapes(self.values, non_batch(self.sampled_elements).non_channel)  # shape check

    @property
    def grid(self) -> UniformGrid:
        """Cast `self.geometry` to a `phi.geom.UniformGrid`."""
        assert isinstance(self.geometry, UniformGrid), f"Geometry is not a UniformGrid but {type(self.geometry)}"
        return self.geometry

    @property
    def mesh(self) -> Mesh:
        """Cast `self.geometry` to a `phi.geom.Mesh`."""
        assert isinstance(self.geometry, Mesh), f"Geometry is not a mesh but {type(self.geometry)}"
        return self.geometry

    @property
    def graph(self) -> Graph:
        """Cast `self.geometry` to a `phi.geom.Graph`."""
        assert isinstance(self.geometry, Graph), f"Geometry is not a mesh but {type(self.geometry)}"
        return self.geometry

    @property
    def faces(self):
        return get_faces(self.geometry, self.boundary)

    @property
    def face_centers(self):
        return self.geometry.face_centers
        # return slice_off_constant_faces(self.geometry.face_centers, self.geometry.boundary_faces, self.boundary)

    @property
    def face_normals(self):
        return self.geometry.face_normals
        # return slice_off_constant_faces(self.geometry.face_normals, self.geometry.boundary_faces, self.boundary)

    @property
    def face_areas(self):
        return self.geometry.face_areas
        # return slice_off_constant_faces(self.geometry.face_areas, self.geometry.boundary_faces, self.boundary)

    @property
    def sampled_elements(self) -> Geometry:
        """
        If the values represent are sampled at the element centers or represent the whole element, returns `self.geometry`.
        If the values are sampled at the faces, returns `self.faces`.
        """
        return get_faces(self.geometry, self.boundary) if is_staggered(self.values, self.geometry) else self.geometry

    @property
    def elements(self):
        # raise SyntaxError("Field.elements is deprecated. Use Field.geometry or Field.sampled_elements instead.")
        warnings.warn("Field.elements is deprecated. Use Field.geometry or Field.sampled_elements instead. Field.elements now defaults to Field.geometry.", DeprecationWarning, stacklevel=2)
        return self.geometry

    @property
    def is_centered(self):
        return not self.is_staggered

    @property
    def is_staggered(self):
        return is_staggered(self.values, self.geometry)

    @property
    def center(self) -> Tensor:
        """ Returns the center points of the `elements` of this `Field`. """
        all_points = self.geometry.get_points(self.sampled_at)
        boundary = self.geometry.get_boundary(self.sampled_at)
        return slice_off_constant_faces(all_points, boundary, self.extrapolation)

    @property
    def points(self):
        return self.center

    @property
    def data(self) -> Tensor:
        return self.values

    def numpy(self, order: DimFilter = None):
        """
        Return the field values as `NumPy` array(s).

        Args:
            order: Dimension order as `str` or `Shape`.

        Returns:
            A single NumPy array for uniform values, else a list of NumPy arrays.
        """
        if order is None and self.is_grid:
            axes = self.values.shape.only(self.geometry.vector.item_names, reorder=True)
            order = concat_shapes(self.values.shape.dual, self.values.shape.batch, axes, self.values.shape.channel)
        if self.values.shape.is_uniform:
            return self.values.numpy(order)
        else:
            assert order is not None, f"order must be specified for non-uniform Field values"
            order = self.values.shape.only(order, reorder=True)
            stack_dims = order.non_uniform_shape
            inner_order = order.without(stack_dims).names
            return [v.numpy(inner_order) for v in unstack(self.values, stack_dims)]

    def uniform_values(self):
        """
        Returns a uniform tensor containing `values`.

        For periodic grids, which always have a uniform value tensor, `values' is returned directly.
        If `values` is not uniform, it is padded as in `StaggeredGrid.staggered_tensor()`.
        """
        if self.values.shape.is_uniform:
            return self.values
        else:
            return self.staggered_tensor()

    @property
    def extrapolation(self) -> Extrapolation:
        """ Returns the `Extrapolation` of this `Field`. """
        return self.boundary

    @property
    def shape(self) -> Shape:
        """
        Returns a shape with the following properties
        
        * The spatial dimension names match the dimensions of this Field
        * The batch dimensions match the batch dimensions of this Field
        * The channel dimensions match the channels of this Field
        """
        if self.is_grid and '~vector' in self.values.shape:
            return batch(self.geometry) & self.resolution & non_dual(self.values).without(self.resolution) & self.geometry.shape['vector']
        set_shape = self.geometry.sets[self.sampled_at]
        return batch(self.geometry) & (channel(self.geometry) - 'vector') & set_shape & self.values.shape

    @property
    def resolution(self):
        return self.geometry.shape.non_channel.non_dual.non_batch

    @property
    def spatial_rank(self) -> int:
        """
        Spatial rank of the field (1 for 1D, 2 for 2D, 3 for 3D).
        This is equal to the spatial rank of the `data`.
        """
        return self.geometry.spatial_rank

    @property
    def bounds(self) -> BaseBox:
        """
        The bounds represent the area inside which the values of this `Field` are valid.
        The bounds will also be used as axis limits for plots.

        The bounds can be set manually in the constructor, otherwise default bounds will be generated.

        For fields that are valid without bounds, the lower and upper limit of `bounds` is set to `-inf` and `inf`, respectively.

        Fields whose spatial rank is determined only during sampling return an empty `Box`.
        """
        if hasattr(self.geometry, 'bounds') and isinstance(self.geometry.bounds, BaseBox):
            return self.geometry.bounds
        extent = self.geometry.bounding_half_extent().vector.as_dual('_extent')
        points = self.geometry.center + extent
        lower = math.min(points, dim=points.shape.non_batch.non_channel)
        upper = math.max(points, dim=points.shape.non_batch.non_channel)
        return Box(lower, upper)

    box = bounds

    @property
    def boundary_names(self):
        return tuple(self.geometry.boundary_faces)

    @property
    def is_grid(self):
        """A Field represents grid data if its `geometry` is a `phi.geom.UniformGrid` instance."""
        return isinstance(self.geometry, UniformGrid)

    @property
    def is_mesh(self):
        """A Field represents mesh data if its `geometry` is a `phi.geom.Mesh` instance."""
        return isinstance(self.geometry, Mesh)

    @property
    def is_graph(self):
        """A Field represents graph data if its `geometry` is a `phi.geom.Graph` instance."""
        return isinstance(self.geometry, Graph)

    @property
    def is_point_cloud(self):
        """A Field represents graph data if its `geometry` is not a set of connected elements, but rather individual geometric objects."""
        if isinstance(self.geometry, (UniformGrid, Mesh, Graph)):
            return False
        if isinstance(self.geometry, (BaseBox, Sphere, Point)):
            return True
        return True

    @property
    def dx(self) -> Tensor:
        assert spatial(self.geometry), f"dx is only defined for elements with spatial dims but Field has elements {self.geometry.shape}"
        return self.bounds.size / self.resolution

    @property
    def cells(self):
        assert isinstance(self.geometry, (UniformGrid, Mesh))
        return self.geometry

    def to_grid(self, resolution=math.EMPTY_SHAPE, bounds=None, **resolution_):
        resolution = resolution.spatial & spatial(**resolution_)
        if self.is_grid and (not resolution or resolution == self.resolution) and (bounds is None or bounds == self.bounds):
            return self
        bounds = self.bounds if bounds is None else bounds
        if not resolution:
            half_sizes = self.geometry.bounding_half_extent()
            if (half_sizes > 0).all:
                size = math.min(2 * half_sizes, non_batch(half_sizes).non_channel)
            else:
                cell_count = non_batch(self.geometry).non_channel.non_dual.volume
                size = (bounds.volume / cell_count) ** (1 / self.spatial_rank)
            res = math.maximum(1, math.round(bounds.size / size))
            resolution = spatial(**res.vector)
        return Field(UniformGrid(resolution, bounds), self, self.boundary)

    def as_points(self, list_dim: Optional[Shape] = instance('elements')) -> 'Field':
        """
        Returns this field as a PointCloud.
        This replaces the `Field.geometry` with a `phi.geom.Point` instance while leaving the sample points unchanged.

        See Also:
            `Field.as_spheres()`.

        Args:
            list_dim: If not `None`, packs spatial, instance and dual dims.
                Defaults to `instance('elements')`.

        Returns:
            `Field` with same values and boundaries but `Point` geometry.
        """
        points = self.sampled_elements.center
        values = self.values
        if list_dim:
            dims = non_batch(points).non_channel & non_batch(points).non_channel
            points = pack_dims(points, dims, list_dim)
            values = pack_dims(values, dims, list_dim)
        return Field(Point(points), values, self.boundary)

    def as_spheres(self, list_dim: Optional[Shape] = instance('elements')) -> 'Field':
        """
        Returns this field as a PointCloud with spherical / circular elements, preserving element volumes.
        This replaces the `Field.geometry` with a `phi.geom.Sphere` instance while leaving the sample points unchanged.

        See Also:
            `Field.as_points()`.

        Args:
            list_dim: If not `None`, packs spatial, instance and dual dims.
                Defaults to `instance('elements')`.

        Returns:
            `Field` with same values and boundaries but `Sphere` geometry.
        """
        points = self.sampled_elements.center
        volumes = self.sampled_elements.volume
        values = self.values
        if list_dim:
            dims = non_batch(points).non_channel & non_batch(points).non_channel
            points = pack_dims(points, dims, list_dim)
            values = pack_dims(values, dims, list_dim)
            volumes = pack_dims(volumes, dims, list_dim)
        return Field(Sphere(points, volume=volumes), values, self.boundary)

    def at_centers(self, **kwargs) -> 'Field':
        """
        Interpolates the values to the cell centers.

        See Also:
            `Field.at_faces()`, `Field.at()`, `resample`.

        Args:
            **kwargs: Sampling arguments.

        Returns:
            `CenteredGrid` sampled at cell centers.
        """
        if self.is_centered:
            return self
        from ._resample import sample
        values = sample(self, self.geometry, at='center', boundary=self.boundary, **kwargs)
        return Field(self.geometry, values, self.boundary)

    def at_faces(self, boundary=None, **kwargs) -> 'Field':
        if self.is_staggered and not boundary:
            return self
        boundary = as_boundary(boundary, self.geometry) if boundary else self.boundary
        from ._resample import sample
        values = sample(self, self.geometry, at='face', boundary=boundary, **kwargs)
        return Field(self.geometry, values, boundary)

    @property
    def sampled_at(self):
        matching_sets = [s for s, s_shape in self.geometry.sets.items() if s_shape.non_batch in self.values.shape]
        return matching_sets[-1]

    def at(self, representation: Union['Field', Geometry], keep_boundary=False, **kwargs) -> 'Field':
        """
        Short for `resample(self, representation)`

        See Also
            `resample()`.

        Returns:
            Field object of same type as `representation`
        """
        from ._resample import resample
        return resample(self, representation, keep_boundary, **kwargs)

    def sample(self, where: Union[Geometry, 'Field', Tensor], at: str = 'center', **kwargs) -> 'Tensor':
        """
        Sample the values of this `Field` at the given location or geometry.

        Args:
            where: Location `Tensor` or `Geometry` or
            at: `'center'` or `'face'`.
            **kwargs: Sampling arguments.

        Returns:
            `Tensor`
        """
        from ._resample import sample
        return sample(self, where, at, **kwargs)

    def closest_values(self, points: Tensor):
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
        warnings.warn("Field.closest_values() is deprecated.", DeprecationWarning, stacklevel=2)
        if isinstance(points, Geometry):
            points = points.center
        # --- CenteredGrid ---
        local_points = self.box.global_to_local(points) * self.resolution - 0.5
        return math.closest_grid_values(self.values, local_points, self.extrapolation)
        # --- StaggeredGrid ---
        if 'staggered_direction' in points.shape:
            points_ = math.unstack(points, '~vector')
            channels = [component.closest_values(p) for p, component in zip(points_, self.vector.unstack())]
        else:
            channels = [component.closest_values(points) for component in self.vector.unstack()]
        return math.stack(channels, points.shape['~vector'])

    def with_values(self, values, **sampling_kwargs):
        """ Returns a copy of this field with `values` replaced. """
        if not isinstance(values, (Tensor, Number)):
            from ._resample import sample
            values = sample(values, self.geometry, self.sampled_at, self.boundary, dot_face_normal=self.geometry if 'vector' not in self.values.shape else None, **sampling_kwargs)
        else:
            if not spatial(values):
                geo_shape = self.sampled_elements.shape if self.is_staggered else self.geometry.shape
                if '~vector' in geo_shape and 'vector' in shape(values) and '~vector' not in shape(values):
                    values = values.vector.as_dual()
                values = expand(wrap(values), geo_shape.non_batch.non_channel)
        return Field(self.geometry, values, self.boundary)

    def with_boundary(self, boundary):
        """ Returns a copy of this field with the `boundary` replaced. """
        boundary = as_boundary(boundary, self.geometry)
        boundary_elements = 'boundary_faces' if self.is_staggered else 'boundary_elements'
        old_determined_slices = {k: s for k, s in getattr(self.geometry, boundary_elements).items() if self.boundary.determines_boundary_values(k)}
        new_determined_slices = {k: s for k, s in getattr(self.geometry, boundary_elements).items() if boundary.determines_boundary_values(k)}
        if old_determined_slices.values() == new_determined_slices.values():
            return Field(self.geometry, self.values, boundary)  # ToDo unnecessary once the rest is implemented
        to_add = {k: sl for k, sl in old_determined_slices.items() if sl not in new_determined_slices.values()}
        to_remove = [sl for sl in new_determined_slices.values() if sl not in old_determined_slices.values()]
        values = math.slice_off(self.values, *to_remove)
        if to_add:
            if self.is_mesh:
                values = self.mesh.pad_boundary(values, to_add, self.boundary)
            elif self.is_grid and self.is_staggered:
                values = self.values.vector.dual.as_channel()
                to_add = {k: {'vector' if dim == '~vector' else dim: v for dim, v in sl.items()} for k, sl in to_add.items()}
                values = math.pad(values, list(to_add.values()), self.boundary, bounds=self.bounds)
                values = values.vector.as_dual()
            else:
                values = math.pad(values, list(to_add.values()), self.boundary, bounds=self.bounds)
        return Field(self.geometry, values, boundary)

    with_extrapolation = with_boundary

    def with_bounds(self, bounds: Box):
        """ Returns a copy of this field with `bounds` replaced. """
        order = list(bounds.vector.item_names)
        geometry = self.geometry.vector[order]
        new_shape = self.values.shape.without(order) & self.values.shape.only(order, reorder=True)
        values = math.swap_axes(self.values, new_shape)
        return Field(geometry, values, self.boundary)

    def with_geometry(self, elements: Geometry):
        """ Returns a copy of this field with `elements` replaced. """
        assert non_batch(elements) == non_batch(self.geometry), f"Field.with_elements() only accepts elements with equal non-batch dimensions but got {elements.shape} for Field with shape {self.geometry.shape}"
        return Field(elements, self.values, self.boundary)

    with_elements = with_geometry

    def shifted(self, delta: Tensor) -> 'Field':
        """
        Move the positions of this field's `geometry` by `delta`.

        See Also:
            `Field.shifted_to`.

        Args:
            delta: Shift amount for each center position of `geometry`.

        Returns:
            New `Field` sampled at `geometry.center + delta`.
        """
        return self.with_geometry(self.geometry.shifted(delta))

    def shifted_to(self, position: Tensor) -> 'Field':
        """
        Move the positions of this field's `geometry` to `positions`.

        See Also:
            `Field.shifted`.

        Args:
            position: New center positions of `geometry`.

        Returns:
            New `Field` sampled at given positions.
        """
        return self.with_geometry(self.geometry.at(position))

    def pad(self, widths: Union[int, tuple, list, dict]) -> 'Field':
        """
        Alias for `phi.field.pad()`.

        Pads this `Field` using its extrapolation.

        Unlike padding the values, this function also affects the `geometry` of the field, changing its size and origin depending on `widths`.

        Args:
            widths: Either `int` or `(lower, upper)` to pad the same number of cells in all spatial dimensions
                or `dict` mapping dimension names to `(lower, upper)`.

        Returns:
            Padded `Field`
        """
        from ._field_math import pad
        return pad(self, widths)

    def gradient(self,
                 boundary: Extrapolation = None,
                 at: str = 'center',
                 dims: math.DimFilter = spatial,
                 stack_dim: Union[Shape, str] = channel('vector'),
                 order=2,
                 implicit: Solve = None,
                 scheme=None,
                 upwind: 'Field' = None,
                 gradient_extrapolation: Extrapolation = None):
        """Alias for `phi.field.spatial_gradient`"""
        from ._field_math import spatial_gradient
        return spatial_gradient(self, boundary=boundary, at=at, dims=dims, stack_dim=stack_dim, order=order, implicit=implicit, scheme=scheme, upwind=upwind, gradient_extrapolation=gradient_extrapolation)

    def divergence(self, order=2, implicit: Solve = None, upwind: 'Field' = None):
        """Alias for `phi.field.divergence`"""
        from ._field_math import divergence
        return divergence(self, order=order, implicit=implicit, upwind=upwind)

    def curl(self, at='corner'):
        """Alias for `phi.field.curl`"""
        from ._field_math import curl
        return curl(self, at=at)

    def laplace(self,
                axes: DimFilter = spatial,
                gradient: 'Field' = None,
                order=2,
                implicit: math.Solve = None,
                weights: Union[Tensor, 'Field'] = None,
                upwind: 'Field' = None,
                correct_skew=True):
        """Alias for `phi.field.laplace`"""
        from ._field_math import laplace
        return laplace(self, axes=axes, gradient=gradient, order=order, implicit=implicit, weights=weights, upwind=upwind, correct_skew=correct_skew)

    def downsample(self, factor: int):
        from ._field_math import downsample2x
        result = self
        while factor >= 2:
            result = downsample2x(result)
            factor /= 2
        if math.close(factor, 1.):
            return result
        from ._resample import resample
        raise NotImplementedError(f"downsample does not support fractional re-sampling. Only 2^n currently supported.")

    def staggered_tensor(self) -> Tensor:
        """
        Stacks all component grids into a single uniform `phi.math.Tensor`.
        The individual components are padded to a common (larger) shape before being stacked.
        The shape of the returned tensor is exactly one cell larger than the grid `resolution` in every spatial dimension.

        Returns:
            Uniform `phi.math.Tensor`.
        """
        assert self.resolution.names == self.shape.get_item_names('vector'), "Field.staggered_tensor() only defined for Fields whose vector components match the resolution"
        padded = []
        for dim, component in zip(self.resolution.names, self.vector):
            widths = {d: (0, 1) for d in self.resolution.names}
            lo_valid, up_valid = self.extrapolation.valid_outer_faces(dim)
            widths[dim] = (int(not lo_valid), int(not up_valid))
            padded.append(math.pad(component.values, widths, self.extrapolation[{'vector': dim}], bounds=self.bounds))
        result = math.stack(padded, channel(vector=self.resolution))
        assert result.shape.is_uniform
        return result

    @staticmethod
    def __stack__(values: tuple, dim: Shape, **kwargs) -> 'Field':
        from ._field_math import stack
        return stack(values, dim, kwargs.get('bounds', None))

    @staticmethod
    def __concat__(values: tuple, dim: str, **kwargs) -> 'Field':
        from ._field_math import concat
        return concat(values, dim)

    def __and__(self, other):
        assert isinstance(other, Field)
        assert instance(self).rank == instance(other).rank == 1, f"Can only use & on PointClouds that have a single instance dimension but got shapes {self.shape} & {other.shape}"
        from ._field_math import concat
        return concat([self, other], instance(self))

    def __matmul__(self, other: 'Field'):  # value @ representation
        # Deprecated. Use `resample(value, field)` instead.
        warnings.warn("value @ field is deprecated. Use resample(value, field) instead.", DeprecationWarning)
        from ._resample import resample
        return resample(self, to=other, keep_boundary=False)

    def __rmatmul__(self, other):  # values @ representation
        if isinstance(other, (Geometry, Number, tuple, list, FieldInitializer)):
            warnings.warn("value @ field is deprecated. Use resample(value, field) instead.", DeprecationWarning)
            from ._resample import resample
            return resample(other, to=self, keep_boundary=False)
        return NotImplemented

    def __rshift__(self, other):
        if isinstance(other, (Field, Geometry)):
            warnings.warn(">> operator for Fields is deprecated. Use field.at(), the constructor or obj @ field instead.", SyntaxWarning, stacklevel=2)
            return self.at(other, keep_boundary=False)
        else:
            return NotImplemented

    def __rrshift__(self, other):
        return self.with_values(other)

    def __lshift__(self, other):
        return self.with_values(other)

    def __rrshift__(self, other):
        warnings.warn(">> operator for Fields is deprecated. Use field.at(), the constructor or obj @ field instead.", SyntaxWarning, stacklevel=2)
        if not isinstance(self, Field):
            return NotImplemented
        if isinstance(other, (Geometry, float, int, complex, tuple, list, FieldInitializer)):
            from ._resample import resample
            return resample(other, to=self, keep_boundary=False)
        return NotImplemented

    def __getitem__(self, item) -> 'Field':
        """
        Access a slice of the Field.
        The returned `Field` may be of a different type than `self`.

        Args:
            item: `dict` mapping dimensions (`str`) to selections (`int` or `slice`) or other supported type, such as `int` or `str`.

        Returns:
            Sliced `Field`.
        """
        item = slicing_dict(self, item)
        if not item:
            return self
        boundary = domain_slice(self.boundary, item, domain_dims=self.boundary_names)
        item_without_vec = {dim: selection for dim, selection in item.items() if dim != 'vector'}
        geometry = self.geometry[item_without_vec]
        if self.is_staggered and 'vector' in item and '~vector' in self.geometry.face_shape:
            assert isinstance(self.geometry, UniformGrid), f"Vector slicing is only supported for grids"
            dims = item['vector']
            dims_ = self.geometry.shape['vector'].after_gather({'vector': dims})
            dims = dims_.item_names[0] if dims_ else [dims] if isinstance(dims, str) else [self.geometry.shape['vector'].item_names[0][dims]]
            proj_dims = set(self.resolution.names) - set(dims)
            if any(dim not in item for dim in proj_dims):
                # warnings.warn(f"Projecting a staggered grid (by slicing 'vector' without the corresponding spatial dims) will return a non-staggered grid. The projected dims {proj_dims} were not sliced off.\nFull slice: {item}")
                item['~vector'] = item['vector']
                del item['vector']
                geometry = self.sampled_elements[item]
            else:
                item['~vector'] = dims
                del item['vector']
        values = self.values[item]
        return Field(geometry, values, boundary)

    def dimension(self, name: str):
        """
        Returns a reference to one of the dimensions of this field.

        The dimension reference can be used the same way as a `Tensor` dimension reference.
        Notable properties and methods of a dimension reference are:
        indexing using `[index]`, `unstack()`, `size`, `exists`, `is_batch`, `is_spatial`, `is_channel`.

        A shortcut to calling this function is the syntax `field.<dim_name>` which calls `field.dimension(<dim_name>)`.

        Args:
            name: dimension name

        Returns:
            dimension reference

        """
        return BoundDim(self, name)

    def __expand__(self, dims: Shape, **kwargs) -> 'Field':
        return self.with_values(expand(self.values, dims, **kwargs))

    def __replace_dims__(self, dims: Tuple[str, ...], new_dims: Shape, **kwargs) -> 'Field':
        elements = math.rename_dims(self.geometry, dims, new_dims)
        values = math.rename_dims(self.values, dims, new_dims)
        extrapolation = math.rename_dims(self.boundary, dims, new_dims, **kwargs)
        return Field(elements, values, extrapolation)

    def __eq__(self, other):
        if not isinstance(other, Field):
            return False
        if self.geometry != other.geometry:
            return False
        if self.boundary != other.boundary:
            return False
        return math.always_close(self.values, other.values)

    def __hash__(self):
        return hash((self.geometry, self.boundary))

    def __mul__(self, other):
        return self._op2(other, lambda d1, d2: d1 * d2)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self._op2(other, lambda d1, d2: d1 / d2)

    def __rtruediv__(self, other):
        return self._op2(other, lambda d1, d2: d2 / d1)

    def __sub__(self, other):
        return self._op2(other, lambda d1, d2: d1 - d2)

    def __rsub__(self, other):
        return self._op2(other, lambda d1, d2: d2 - d1)

    def __add__(self, other):
        return self._op2(other, lambda d1, d2: d1 + d2)

    __radd__ = __add__

    def __pow__(self, power, modulo=None):
        return self._op2(power, lambda f, p: f ** p)

    def __neg__(self):
        return self._op1(lambda x: -x)

    def __gt__(self, other):
        return self._op2(other, lambda x, y: x > y)

    def __ge__(self, other):
        return self._op2(other, lambda x, y: x >= y)

    def __lt__(self, other):
        return self._op2(other, lambda x, y: x < y)

    def __le__(self, other):
        return self._op2(other, lambda x, y: x <= y)

    def __abs__(self):
        return self._op1(lambda x: abs(x))

    def _op1(self: 'Field', operator: Callable) -> 'Field':
        """
        Perform an operation on the data of this field.

        Args:
          operator: function that accepts tensors and extrapolations and returns objects of the same type and dimensions

        Returns:
          Field of same type
        """
        values = operator(self.values)
        extrapolation_ = operator(self.boundary)
        return self.with_values(values).with_extrapolation(extrapolation_)

    def _op2(self, other, operator) -> 'Field':
        if isinstance(other, Geometry):
            raise ValueError(f"Cannot combine {self.__class__.__name__} with a Geometry, got {type(other)}")
        if isinstance(other, Field):
            if self.geometry == other.geometry:
                values = operator(self.values, other.values)
                extrapolation_ = operator(self.boundary, other.extrapolation)
                return Field(self.geometry, values, extrapolation_)
            from ._resample import sample
            other_values = sample(other, self.geometry, self.sampled_at, self.boundary, dot_face_normal=self.geometry)
            values = operator(self.values, other_values)
            boundary = operator(self.boundary, other.extrapolation)
            return Field(self.geometry, values, boundary)
        else:
            if isinstance(other, (tuple, list)) and len(other) == self.spatial_rank:
                other = math.wrap(other, self.geometry.shape['vector'])
            else:
                other = math.wrap(other)
            # try:
            #     boundary = operator(self.boundary, as_boundary(other, self.geometry))
            # except TypeError:  # e.g. ZERO_GRADIENT + constant
            boundary = self.boundary  # constants don't affect the boundary conditions (legacy reasons)
            if 'vector' in self.shape and 'vector' not in self.values.shape and '~vector' in self.values.shape:
                other = other.vector.as_dual()
            values = operator(self.values, other)
            return Field(self.geometry, values, boundary)

    def __repr__(self):
        if self.is_grid:
            type_name = "Grid" if self.is_centered else "Grid faces"
        elif self.is_mesh:
            type_name = "Mesh" if self.is_centered else "Mesh faces"
        elif self.is_point_cloud:
            type_name = "Point cloud" if self.is_centered else "Point cloud edges"
        elif self.is_graph:
            type_name = "Graph" if self.is_centered else "Graph edges"
        else:
            type_name = self.__class__.__name__
        if self.values is not None:
            return f"{type_name}[{self.values}, ext={self.boundary}]"
        else:
            return f"{type_name}[{self.resolution}, ext={self.boundary}]"

    def grid_scatter(self, *args, **kwargs):
        """Deprecated. Use `sample` with `scatter=True` instead."""
        warnings.warn("Field.grid_scatter() is deprecated. Use field.sample() with scatter=True instead.", DeprecationWarning, stacklevel=2)
        from ._resample import grid_scatter
        return grid_scatter(self, *args, **kwargs)

    def as_boundary(self) -> Extrapolation:
        """
        Returns an `Extrapolation` representing this 'Field''s values as a Dirichlet (constant) boundary.
        If this `Field` encloses the required boundaries, its values will be interpolated to the required boundaries.
        If boundaries outside of this `Field`'s sampled domain are required, this `Field`'s boundary conditions will be applied to determine the boundary values.

        Returns:
            `Extrapolation`
        """
        from ._embed import FieldEmbedding
        return FieldEmbedding(self)


def as_boundary(obj: Union[Extrapolation, Tensor, float, Field, None], _geometry=None) -> Extrapolation:
    """
    Returns an `Extrapolation` representing `obj`.

    Args:
        obj: One of

            * `float` or `Tensor`: Extrapolate with a constant value
            * `Extrapolation`: Use as-is.
            * `Field`: Sample values from `obj`, embedding another field inside `obj`.

    Returns:
        `Extrapolation`
    """
    return obj.as_boundary() if isinstance(obj, Field) else math.extrapolation.as_extrapolation(obj)


def is_staggered(values: Tensor, geometry: Geometry):
    return bool(dual(values)) and geometry.face_shape.dual in dual(values)


def get_faces(geometry: Geometry, boundary: Extrapolation):
    return slice_off_constant_faces(geometry.faces, geometry.boundary_faces, boundary)


def get_sample_points(geometry: Geometry, at: str, boundary: Extrapolation):
    if at == 'center':
        return slice_off_constant_faces(geometry.center, geometry.boundary_elements, boundary)
    elif at == 'face':
        return slice_off_constant_faces(geometry.face_centers, geometry.boundary_faces, boundary)
    raise ValueError(at)
