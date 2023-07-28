import warnings
from numbers import Number
from typing import TypeVar, Callable, Union

from phi import math
from phiml.math import Shape, Tensor, channel
from phiml.math.extrapolation import Extrapolation
from phi.geom import Geometry, Box, Point
from phiml.math.magic import BoundDim


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

    @property
    def shape(self) -> Shape:
        """
        Returns a shape with the following properties
        
        * The spatial dimension names match the dimensions of this Field
        * The batch dimensions match the batch dimensions of this Field
        * The channel dimensions match the channels of this Field
        """
        raise NotImplementedError

    @property
    def spatial_rank(self) -> int:
        """
        Spatial rank of the field (1 for 1D, 2 for 2D, 3 for 3D).
        This is equal to the spatial rank of the `data`.
        """
        raise NotImplementedError

    @property
    def bounds(self) -> Box:
        """
        The bounds represent the area inside which the values of this `Field` are valid.
        The bounds will also be used as axis limits for plots.

        The bounds can be set manually in the constructor, otherwise default bounds will be generated.

        For fields that are valid without bounds, the lower and upper limit of `bounds` is set to `-inf` and `inf`, respectively.

        Fields whose spatial rank is determined only during sampling return an empty `Box`.
        """
        raise NotImplementedError

    def _sample(self, geometry: Geometry, **kwargs) -> math.Tensor:
        """ For internal use only. Use `sample()` instead. """
        raise NotImplementedError(self)

    def at(self, representation: 'SampledField', keep_extrapolation=False, **kwargs) -> 'SampledFieldType':
        """
        Short for `resample(self, representation)`

        See Also
            `resample()`.

        Returns:
            Field object of same type as `representation`
        """
        return resample(self, representation, keep_extrapolation, **kwargs)

    def __matmul__(self, other: 'SampledField'):  # value @ representation
        # Deprecated. Use `resample(value, field)` instead.
        warnings.warn("value @ field is deprecated. Use resample(value, field) instead.", DeprecationWarning)
        return self.at(other, keep_extrapolation=False)

    def __rmatmul__(self, other):  # values @ representation
        if not isinstance(self, SampledField):
            return NotImplemented
        if isinstance(other, (Geometry, Number, tuple, list)):
            return self.with_values(other)
        return NotImplemented

    def __rshift__(self, other):
        warnings.warn(">> operator for Fields is deprecated. Use field.at(), the constructor or obj @ field instead.", SyntaxWarning, stacklevel=2)
        return self.at(other, keep_extrapolation=False)

    def __rrshift__(self, other):
        warnings.warn(">> operator for Fields is deprecated. Use field.at(), the constructor or obj @ field instead.", SyntaxWarning, stacklevel=2)
        if not isinstance(self, SampledField):
            return NotImplemented
        if isinstance(other, (Geometry, float, int, complex, tuple, list)):
            return self.with_values(other)
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
        raise NotImplementedError(self)

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

    def __repr__(self):
        return f"{self.__class__.__name__} {self.shape}"

    @property
    def is_grid(self):
        """ `isinstance(self, Grid)`. Added for forward compatibility with 2.5. """
        from ._grid import Grid
        return isinstance(self, Grid)

    @property
    def is_point_cloud(self):
        """ `isinstance(self, PointCloud)`. Added for forward compatibility with 2.5. """
        from ._point_cloud import PointCloud
        return isinstance(self, PointCloud)

    @property
    def is_staggered(self):
        """ Whether the field values are sampled between elements. Added for forward compatibility with 2.5. """
        from ._grid import StaggeredGrid
        return isinstance(self, StaggeredGrid)


class SampledField(Field):
    """
    Base class for fields that are sampled at specific locations such as grids or point clouds.
    """

    def __init__(self,
                 elements: Union[Geometry, Tensor],
                 values: Tensor,
                 extrapolation: Union[float, Extrapolation, Field, None],
                 bounds: Union[Box, None]):
        """
        Args:
          elements: Geometry object specifying the sample points and sizes
          values: values corresponding to elements
          extrapolation: values outside elements
        """
        if isinstance(elements, Tensor):
            elements = Point(elements)
        assert isinstance(elements, Geometry), elements
        assert isinstance(values, Tensor), f"Values must be a Tensor but got {values}."
        assert bounds is None or isinstance(bounds, Box), 'Invalid bounds.'
        self._bounds = bounds
        self._elements: Geometry = elements
        self._values: Tensor = values
        self._extrapolation: Extrapolation = as_extrapolation(extrapolation)

    @property
    def bounds(self) -> Box:
        raise NotImplementedError(self.__class__)

    def _sample(self, geometry: Geometry, **kwargs) -> math.Tensor:
        raise NotImplementedError(self.__class__)

    def with_values(self, values):
        """ Returns a copy of this field with `values` replaced. """
        raise NotImplementedError(self)

    def with_extrapolation(self, extrapolation: Extrapolation):
        """ Returns a copy of this field with `values` replaced. """
        raise NotImplementedError(self)

    @property
    def shape(self):
        raise NotImplementedError()

    @property
    def spatial_rank(self) -> int:
        return self._elements.spatial_rank

    def __getitem__(self: 'FieldType', item) -> 'FieldType':
        raise NotImplementedError(self)

    @staticmethod
    def __stack__(values: tuple, dim: Shape, **kwargs) -> 'FieldType':
        from ._field_math import stack
        return stack(values, dim, kwargs.get('bounds', None))

    @staticmethod
    def __concat__(values: tuple, dim: str, **kwargs) -> 'FieldType':
        from ._field_math import concat
        return concat(values, dim)

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
    def points(self) -> Tensor:
        return self.elements.center

    @property
    def values(self) -> Tensor:
        return self._values

    data = values

    @property
    def extrapolation(self) -> Extrapolation:
        return self._extrapolation

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

    def _op1(self: 'SampledFieldType', operator: Callable) -> 'SampledFieldType':
        """
        Perform an operation on the data of this field.

        Args:
          operator: function that accepts tensors and extrapolations and returns objects of the same type and dimensions

        Returns:
          Field of same type
        """
        values = operator(self.values)
        extrapolation_ = operator(self._extrapolation)
        return self.with_values(values).with_extrapolation(extrapolation_)

    def _op2(self, other, operator) -> 'SampledField':
        if isinstance(other, Geometry):
            raise ValueError(f"Cannot combine {self.__class__.__name__} with a Geometry, got {type(other)}")
        if isinstance(other, Field):
            other_values = reduce_sample(other, self._elements)
            values = operator(self._values, other_values)
            extrapolation_ = operator(self._extrapolation, other.extrapolation)
            return self.with_values(values).with_extrapolation(extrapolation_)
        else:
            if isinstance(other, (tuple, list)) and len(other) == self.spatial_rank:
                other = math.wrap(other, self.points.shape['vector'])
            else:
                other = math.wrap(other)
            values = operator(self._values, other)
            return self.with_values(values)


def sample(field: Union[Field, Geometry],
           geometry: Union[Geometry, SampledField, Tensor],
           **kwargs) -> math.Tensor:
    """
    Computes the field value inside the volume of the (batched) `geometry`.

    The field value may be determined by integrating over the volume, sampling the central value or any other way.

    The batch dimensions of `geometry` are matched with this field.
    The `geometry` must not share any channel dimensions with this field.
    Spatial dimensions of `geometry` can be used to sample a grid of geometries.

    See Also:
        `reduce_sample()`, `Field.at()`, [Resampling overview](https://tum-pbs.github.io/PhiFlow/Fields.html#resampling-fields).

    Args:
        field: Source `Field` to sample.
        geometry: Single or batched `phi.geom.Geometry` or `SampledField` or location `Tensor`.
            When passing a `SampledField`, its `elements` are used as sample points.
            When passing a vector-valued `Tensor`, a `Point` geometry will be created.
        **kwargs: Sampling arguments, e.g. to specify the numerical scheme.
            By default, linear interpolation is used.
            Grids also support 6th order implicit sampling at mid-points.

    Returns:
        Sampled values as a `phiml.math.Tensor`
    """
    geometry = _get_geometry(geometry)
    if isinstance(field, Geometry):
        from ._field_math import mask
        field = mask(field)
    geom_ch = channel(geometry).without('vector')
    assert all(dim not in field.shape for dim in geom_ch)
    if isinstance(field, SampledField) and field.elements.shallow_equals(geometry) and not geom_ch:
        return field.values
    if geom_ch:
        sampled = [field._sample(p, **kwargs) for p in geometry.unstack(geom_ch.name)]
        return math.stack(sampled, geom_ch)
    else:
        return field._sample(geometry, **kwargs)


def reduce_sample(field: Union[Field, Geometry],
                  geometry: Union[Geometry, SampledField, Tensor],
                  dim=channel('vector'),
                  **kwargs) -> math.Tensor:
    """
    Similar to `sample()`, but matches channel dimensions of `geometry` with channel dimensions of this field.
    Currently, `geometry` may have at most one channel dimension.

    See Also:
        `sample()`, `Field.at()`, [Resampling overview](https://tum-pbs.github.io/PhiFlow/Fields.html#resampling-fields).

    Args:
        field: Source `Field` to sample.
        geometry: Single or batched `phi.geom.Geometry` or `SampledField` or location `Tensor`.
            When passing a `SampledField`, its `elements` are used as sample points.
            When passing a vector-valued `Tensor`, a `Point` geometry will be created.
        dim: Dimension of result, resulting from reduction of channel dimensions.
        **kwargs: Sampling arguments, e.g. to specify the numerical scheme.
            By default, linear interpolation is used.
            Grids also support 6th order implicit sampling at mid-points.

    Returns:
        Sampled values as a `phiml.math.Tensor`
    """
    geometry = _get_geometry(geometry)
    if isinstance(field, Geometry):
        from ._field_math import mask
        field = mask(field)
    if isinstance(field, SampledField) and field.elements.shallow_equals(geometry):
        return field.values
    if channel(geometry).without('vector'):  # Reduce this dimension
        geom_ch = channel(geometry).without('vector')
        assert geom_ch.rank == 1, "Only single-dimension reduction supported."
        if field.shape.channel.volume > 1:
            assert field.shape.channel.volume == geom_ch.volume, f"Cannot sample field with channels {field.shape.channel} at elements with channels {geometry.shape.channel}."
            components = math.unstack(field, field.shape.channel.name)
            sampled = [c._sample(p, **kwargs) for c, p in zip(components, geometry.unstack(geom_ch.name))]
        else:
            sampled = [field._sample(p, **kwargs) for p in geometry.unstack(channel(geometry).without('vector').name)]
        dim = dim.with_size(geometry.shape.channel.item_names[0])
        return math.stack(sampled, dim)
    else:  # Nothing to reduce
        return field._sample(geometry, **kwargs)


def resample(value: Union[Field, Geometry, Tensor, float], to: SampledField, keep_extrapolation=False, **kwargs):
    """
    Samples a `Field`, `Geometry` or value at the sample points of the field `to`.
    The result will approximate `value` on the data structure of `to`.
    Unlike `sample()`, this method returns a `Field` object, not a `Tensor`.

    Aliases:
        `value.at(to)`, (and the deprecated `value @ to`).

    See Also:
        `sample()`, `reduce_sample()`, `Field.at()`, [Resampling overview](https://tum-pbs.github.io/PhiFlow/Fields.html#resampling-fields).

    Args:
        value: Object containing values to resample.
            This can be
        to: `SampledField` (`CenteredGrid`, `StaggeredGrid` or `PointCloud`) object defining the sample points.
            The current values of `to` are ignored.
        keep_extrapolation: Only available if `self` is a `SampledField`.
            If True, the resampled field will inherit the extrapolation from `self` instead of `representation`.
            This can result in non-compatible value tensors for staggered grids where the tensor size depends on the extrapolation type.
        **kwargs: Sampling arguments, e.g. to specify the numerical scheme.
            By default, linear interpolation is used.
            Grids also support 6th order implicit sampling at mid-points.

    Returns:
        Field object of same type as `representation`

    Examples:
        >>> grid = CenteredGrid(x=64, y=32)
        >>> field.resample(Noise(), to=grid)
        CenteredGrid[(xˢ=64, yˢ=32), size=(x=64, y=32), extrapolation=float64 0.0]
        >>> field.resample(1, to=grid)
        CenteredGrid[(xˢ=64, yˢ=32), size=(x=64, y=32), extrapolation=float64 0.0]
        >>> field.resample(Box(x=1, y=2), to=grid)
        CenteredGrid[(xˢ=64, yˢ=32), size=(x=64, y=32), extrapolation=float64 0.0]
        >>> field.resample(grid, to=grid) == grid
        True
    """
    if not isinstance(value, (Field, Geometry)):
        return to.with_values(value)
    resampled = reduce_sample(value, to.elements, **kwargs)
    extrap = value.extrapolation if isinstance(value, SampledField) and keep_extrapolation else to.extrapolation
    return to.with_values(resampled).with_extrapolation(extrap)


def _get_geometry(geometry):
    if isinstance(geometry, SampledField):
        return geometry.elements
    elif isinstance(geometry, Tensor) and 'vector' in geometry.shape:
        return Point(geometry)
    elif isinstance(geometry, Geometry):
        return geometry
    else:
        raise ValueError(f"A Geometry, SampledField or location Tensor is required but got {geometry}")


FieldType = TypeVar('FieldType', bound=Field)
SampledFieldType = TypeVar('SampledFieldType', bound=SampledField)


def as_extrapolation(obj: Union[Extrapolation, float, Field, None]) -> Extrapolation:
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
