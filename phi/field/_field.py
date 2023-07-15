import warnings
from numbers import Number
from typing import Callable, Union, Tuple, Dict, Any

from phi import math
from phi.math import Shape, Tensor, channel, non_batch, expand, instance, spatial, wrap, dual, non_dual
from phi.math.extrapolation import Extrapolation, ConstantExtrapolation
from phi.geom import Geometry, Box, Point, BaseBox, UniformGrid, UnstructuredMesh, Sphere
from phi.math.magic import BoundDim, slicing_dict


class FieldInitializer:  # ToDo replace by simple function

    def _sample(self, geometry: Geometry, at: str, boundaries: Extrapolation, **kwargs) -> math.Tensor:
        """ For internal use only. Use `sample()` instead. """
        raise NotImplementedError(self)


class Field:
    """
    Base class for all fields.
    
    Important implementations:
    
    * CenteredGrid
    * StaggeredGrid
    * PointCloud
    * Noise
    
    See the `phi.field` module documentation at https://tum-pbs.github.io/PhiFlow/Fields.html
    """

    def __init__(self,
                 elements: Union[Geometry, Tensor],
                 values: Union[Tensor, Number, bool, Callable, FieldInitializer, Geometry, 'Field'],
                 boundary: Union[Number, Extrapolation, 'Field', dict],
                 **sampling_kwargs):
        """
        Args:
          elements: Geometry object specifying the sample points and sizes
          values: values corresponding to elements
          extrapolation: values outside elements
        """
        assert isinstance(elements, Geometry), elements
        self._boundary: Extrapolation = as_boundary(boundary)
        self._elements: Geometry = elements
        if isinstance(values, (Tensor, Number, bool)):
            values = wrap(values)
        else:
            from ._resample import sample
            values = sample(values, elements, 'center', self._boundary, **sampling_kwargs)
        if non_batch(elements).non_channel not in values.shape:
            values = expand(wrap(values), non_batch(elements).non_channel)
        self._values: Tensor = values

    @property
    def elements(self) -> Geometry:
        """
        Returns a geometrical representation of the discrete volume elements.
        The result is a tuple of Geometry objects, each of which can have additional spatial (but not batch) dimensions.
        
        For grids, the geometries are boxes while particle fields may be represented as spheres.
        
        If this Field has no discrete points, this method returns an empty geometry.
        """
        return self._elements

    @property
    def is_centered(self):
        return dual(self._values).is_empty

    @property
    def is_staggered(self):
        return bool(dual(self._values)) and dual(self._values) in self._elements.face_shape

    @property
    def points(self) -> Tensor:
        """ Returns the center points of the `elements` of this `Field`. """
        # { ~vector, x }      for staggered grid
        # { ~cells }          for unstructured mesh
        # { particles }       for graph
        # { ~particles }      for graph edges
        if self.is_centered:
            return slice_off_constant(self._elements.center, self._elements.boundary_elements, self.extrapolation)
        elif self.is_staggered:
            return slice_off_constant(self._elements.face_centers, self._elements.boundary_faces, self.extrapolation)
        else:
            raise NotImplementedError

    @property
    def values(self) -> Tensor:
        """ Returns the `values` of this `Field`. """
        return self._values

    data = values

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
    def boundary(self) -> Extrapolation:
        """
        Returns the boundary conditions set for this `Field`.

        Returns:
            Single `Extrapolation` instance that encodes the (varying) boundary conditions for all boundaries of this field's `elements`.
        """
        return self._boundary

    @property
    def extrapolation(self) -> Extrapolation:
        """ Returns the `Extrapolation` of this `Field`. """
        return self._boundary

    @property
    def shape(self) -> Shape:
        """
        Returns a shape with the following properties
        
        * The spatial dimension names match the dimensions of this Field
        * The batch dimensions match the batch dimensions of this Field
        * The channel dimensions match the channels of this Field
        """
        if self.is_staggered:
            return self.resolution & non_dual(self._values).without(self.resolution) & self._elements.shape['vector']
        return self.resolution & non_dual(self._values).without(self.resolution)

    @property
    def resolution(self):
        return self._elements.shape.non_channel.non_dual.non_batch

    @property
    def spatial_rank(self) -> int:
        """
        Spatial rank of the field (1 for 1D, 2 for 2D, 3 for 3D).
        This is equal to the spatial rank of the `data`.
        """
        return self._elements.spatial_rank

    @property
    def bounds(self) -> BaseBox:
        """
        The bounds represent the area inside which the values of this `Field` are valid.
        The bounds will also be used as axis limits for plots.

        The bounds can be set manually in the constructor, otherwise default bounds will be generated.

        For fields that are valid without bounds, the lower and upper limit of `bounds` is set to `-inf` and `inf`, respectively.

        Fields whose spatial rank is determined only during sampling return an empty `Box`.
        """
        if isinstance(self._elements, UniformGrid):
            return self._elements.bounds
        else:
            return None

    box = bounds

    @property
    def is_grid(self):
        return isinstance(self._elements, UniformGrid)

    @property
    def is_mesh(self):
        return isinstance(self._elements, UnstructuredMesh)

    @property
    def is_point_cloud(self):
        if isinstance(self._elements, (UniformGrid, UnstructuredMesh)):
            return False
        if isinstance(self._elements, (BaseBox, Sphere, Point)):
            return True
        return True

    @property
    def dx(self) -> Tensor:
        assert spatial(self._elements), f"dx is only defined for elements with spatial dims but Field has elements {self._elements.shape}"
        return self.bounds.size / self.resolution

    @property
    def cells(self):
        assert isinstance(self._elements, (UniformGrid, UnstructuredMesh))
        return self._elements

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
        values = sample(self, self._elements, at='center', extrapolation=self._boundary, **kwargs)
        return Field(self._elements, values, self._boundary)

    def at_faces(self, **kwargs) -> 'Field':
        if self.is_staggered:
            return self
        from ._resample import sample
        values = sample(self, self._elements, at='face', extrapolation=self._boundary, **kwargs)
        return Field(self._elements, values, self._boundary)

    def at(self, representation: 'Field', keep_extrapolation=False, **kwargs) -> 'Field':
        """
        Short for `resample(self, representation)`

        See Also
            `resample()`.

        Returns:
            Field object of same type as `representation`
        """
        from ._resample import resample
        return resample(self, representation, keep_extrapolation, **kwargs)

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
        warnings.warn("Field.closest_values() is deprecated.", DeprecationWarning, stacklevel=2)
        # --- CenteredGrid ---
        local_points = self.box.global_to_local(points.center) * self.resolution - 0.5
        return math.closest_grid_values(self.values, local_points, self.extrapolation)
        # --- StaggeredGrid ---
        if 'staggered_direction' in points.shape:
            points_ = math.unstack(points, 'staggered_direction')
            channels = [component.closest_values(p) for p, component in zip(points_, self.vector.unstack())]
        else:
            channels = [component.closest_values(points) for component in self.vector.unstack()]
        return math.stack(channels, points.shape['vector'])

    def with_values(self, values):
        """ Returns a copy of this field with `values` replaced. """
        return Field(self._elements, values, self._boundary)

    def with_extrapolation(self, extrapolation: Extrapolation):
        """ Returns a copy of this field with `values` replaced. """
        extrapolation = as_boundary(extrapolation)
        return Field(self._elements, self._values, extrapolation)
        # ToDo StaggeredGrid
        if all([extrapolation.valid_outer_faces(dim) == self.extrapolation.valid_outer_faces(dim) for dim in self.resolution.names]):
            return StaggeredGrid(self.values, extrapolation=extrapolation, bounds=self.bounds)
        else:
            values = []
            for dim, component in zip(self.shape.spatial.names, math.unstack(self.values, 'vector')):
                old_lo, old_hi = [int(v) for v in self.extrapolation.valid_outer_faces(dim)]
                new_lo, new_hi = [int(v) for v in extrapolation.valid_outer_faces(dim)]
                widths = (new_lo - old_lo, new_hi - old_hi)
                values.append(math.pad(component, {dim: widths}, self.extrapolation, bounds=self.bounds))
            values = math.stack(values, channel(vector=self.resolution))
            return StaggeredGrid(values, extrapolation, bounds=self.bounds)

    def with_bounds(self, bounds: Box):
        """ Returns a copy of this field with `bounds` replaced. """
        return Field(self._elements, self._values, self._boundary)

    def with_elements(self, elements: Geometry):
        """ Returns a copy of this field with `elements` replaced. """
        assert non_batch(elements) == non_batch(self._elements), f"Field.with_elements() only accepts elements with equal non-batch dimensions but got {elements.shape} for Field with shape {self._elements.shape}"
        return Field(elements, self._values, self._boundary)

    def shifted(self, delta):
        return self.with_elements(self.elements.shifted(delta))

    def staggered_tensor(self) -> Tensor:
        """
        Stacks all component grids into a single uniform `phi.math.Tensor`.
        The individual components are padded to a common (larger) shape before being stacked.
        The shape of the returned tensor is exactly one cell larger than the grid `resolution` in every spatial dimension.

        Returns:
            Uniform `phi.math.Tensor`.
        """
        assert self.resolution.names == self._values.shape.get_item_names('vector'), "Field.staggered_tensor() only defined for Fields whose vector components match the resolution"
        padded = []
        for dim, component in zip(self.resolution.names, math.unstack(self.values, 'vector')):
            widths = {d: (0, 1) for d in self.resolution.names}
            lo_valid, up_valid = self.extrapolation.valid_outer_faces(dim)
            widths[dim] = (int(not lo_valid), int(not up_valid))
            padded.append(math.pad(component, widths, self.extrapolation[{'vector': dim}], bounds=self.bounds))
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
        return resample(self, to=other, keep_extrapolation=False)

    def __rmatmul__(self, other):  # values @ representation
        if isinstance(other, (Geometry, Number, tuple, list, FieldInitializer)):
            warnings.warn("value @ field is deprecated. Use resample(value, field) instead.", DeprecationWarning)
            from ._resample import resample
            return resample(other, to=self, keep_extrapolation=False)
        return NotImplemented

    def __rshift__(self, other):
        warnings.warn(">> operator for Fields is deprecated. Use field.at(), the constructor or obj @ field instead.", SyntaxWarning, stacklevel=2)
        return self.at(other, keep_extrapolation=False)

    def __rrshift__(self, other):
        warnings.warn(">> operator for Fields is deprecated. Use field.at(), the constructor or obj @ field instead.", SyntaxWarning, stacklevel=2)
        if not isinstance(self, Field):
            return NotImplemented
        if isinstance(other, (Geometry, float, int, complex, tuple, list, FieldInitializer)):
            from ._resample import resample
            return resample(other, to=self, keep_extrapolation=False)
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
        if self.is_staggered:
            if 'vector' in item:
                assert isinstance(self._elements, UniformGrid), f"Vector slicing is only supported for grids"
                item['~vector'] = item['vector']
                del item['vector']
        item_without_vec = {dim: selection for dim, selection in item.items() if dim != 'vector'}
        elements = self.elements[item_without_vec]
        values = self._values[item]
        extrapolation = self._boundary[item]
        return Field(elements, values, extrapolation)

    def __getattr__(self, name: str) -> BoundDim:
        return BoundDim(self, name)

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

    def __value_attrs__(self):
        return '_values', '_boundary'

    def __variable_attrs__(self):
        return '_values', '_elements'

    def __expand__(self, dims: Shape, **kwargs) -> 'Field':
        return self.with_values(expand(self.values, dims, **kwargs))

    def __replace_dims__(self, dims: Tuple[str, ...], new_dims: Shape, **kwargs) -> 'Field':
        elements = math.rename_dims(self._elements, dims, new_dims)
        values = math.rename_dims(self._values, dims, new_dims)
        extrapolation = math.rename_dims(self._boundary, dims, new_dims, **kwargs)
        return Field(elements, values, extrapolation)

    def __eq__(self, other):
        if not isinstance(other, Field):
            return False
        # Check everything but __variable_attrs__ (values): elements type, extrapolation, add_overlapping
        if type(self._elements) is not type(other._elements):
            return False
        if self._boundary != other.boundary:
            return False
        if self._values is None:
            return other._values is None
        if other._values is None:
            return False
        if not math.all_available(self._values) or not math.all_available(other._values):  # tracers involved
            if math.all_available(self._values) != math.all_available(other._values):
                return False
            else:  # both tracers
                return self._values.shape == other._values.shape
        if self._values.shape != other._values.shape:
            return False
        return bool((self._values == other._values).all)

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
        def inner_add(d1, d2):
            from phiml.math._tensors import TensorStack
            if isinstance(d1, TensorStack) and isinstance(d2, TensorStack):
                print("test")
                pass
            return d1 + d2

        return self._op2(other, inner_add)

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
        extrapolation_ = operator(self._boundary)
        return self.with_values(values).with_extrapolation(extrapolation_)

    def _op2(self, other, operator) -> 'Field':
        if isinstance(other, Geometry):
            raise ValueError(f"Cannot combine {self.__class__.__name__} with a Geometry, got {type(other)}")
        if isinstance(other, Field):
            if self._elements == other._elements:
                values = operator(self._values, other.values)
                extrapolation_ = operator(self._boundary, other.extrapolation)
                return Field(self._elements, values, extrapolation_)
            from ._resample import sample
            other_values = sample(other, self._elements)
            values = operator(self._values, other_values)
            extrapolation_ = operator(self._boundary, other.extrapolation)
            return self.with_values(values).with_extrapolation(extrapolation_)
        else:
            if isinstance(other, (tuple, list)) and len(other) == self.spatial_rank:
                other = math.wrap(other, self.points.shape['vector'])
            else:
                other = math.wrap(other)
            values = operator(self._values, other)
            return self.with_values(values)

    def __repr__(self):
        if self.is_grid:
            type_name = "Grid" if self.is_centered else "Grid faces"
        elif self.is_mesh:
            type_name = "Mesh" if self.is_centered else "Mesh faces"
        elif self.is_point_cloud:
            type_name = "Point cloud" if self.is_centered else "Point cloud faces"
        else:
            type_name = self.__class__.__name__
        if self._values is not None:
            return f"{type_name}[{self.values}, ext={self._boundary}]"
        else:
            return f"{type_name}[{self.resolution}, ext={self._boundary}]"

    def grid_scatter(self, *args, **kwargs):
        """Deprecated. Use `sample` with `scatter=True` instead."""
        warnings.warn("Field.grid_scatter() is deprecated. Use field.sample() with scatter=True instead.", DeprecationWarning, stacklevel=2)
        from ._resample import grid_scatter
        return grid_scatter(self, *args, **kwargs)


def as_boundary(obj: Union[Extrapolation, float, Field, None]) -> Extrapolation:
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
    if isinstance(obj, Field):
        from ._embed import FieldEmbedding
        return FieldEmbedding(obj)
    else:
        return math.extrapolation.as_extrapolation(obj)


def slice_off_constant(values: Union[Tensor, Shape], boundary_slices: Dict[str, Tuple[Dict[str, slice], Dict[str, slice]]], extrapolation: Extrapolation):
    for name, (lower, upper) in boundary_slices.items():
        lower_valid, upper_valid = extrapolation.valid_outer_faces(name)
        if not lower_valid:
            slice_off_(lower)
        if not upper_valid:
            slice_off_(upper)
        # if len(boundary_slice) == 1:
        #     values = values[{dim: _invert_slice(s) for dim, s in boundary_slice.items()}]
        #     raise NotImplementedError  # ToDo this has not been tested
        # elif len(boundary_slice) == 2:
        #     unstack_dim = values.shape[next(iter(boundary_slice))]
        #     val_list = list(math.unstack(values, unstack_dim))
        #     idx = unstack_dim.item_names[0].index(boundary_slice[unstack_dim.name])
        #     val_list[idx] = val_list[idx][{dim: _invert_slice(s) for dim, s in boundary_slice.items() if dim != unstack_dim.name}]
        #     values = math.stack(val_list, unstack_dim)
        # else:
        #     raise NotImplementedError("Higher-order non-uniform slices not yet supported")
    return values


def _invert_slice(s: slice):
    raise NotImplementedError
