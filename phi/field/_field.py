from typing import TypeVar, Callable

from phi import math
from phi.geom import Geometry, Box
from phi.math import Shape, Tensor, Extrapolation, channel
from phi.math._tensors import Sliceable, BoundDim


class Field(Sliceable):
    """
    Base class for all fields.
    
    Important implementations:
    
    * CenteredGrid
    * StaggeredGrid
    * PointCloud
    * Noise
    
    See the `phi.field` module documentation at https://tum-pbs.github.io/PhiFlow/Fields.html
    """

    def __init__(self, bounds: Box or None):
        """
        Args:
            bounds: Bounds inside which the values of this `Field` are valid.
                The bounds will also be used as axis limits for plots.
        """
        assert bounds is None or isinstance(bounds, Box), 'Invalid bounds.'
        self._bounds = bounds

    @property
    def shape(self) -> Shape:
        """
        Returns a shape with the following properties
        
        * The spatial dimension names match the dimensions of this Field
        * The batch dimensions match the batch dimensions of this Field
        * The channel dimensions match the channels of this Field
        """
        raise NotImplementedError()

    @property
    def spatial_rank(self) -> int:
        """
        Spatial rank of the field (1 for 1D, 2 for 2D, 3 for 3D).
        This is equal to the spatial rank of the `data`.
        """
        return self.shape.spatial.rank

    @property
    def bounds(self) -> Box:
        """
        The bounds represent the area inside which the values of this `Field` are valid.
        The bounds will also be used as axis limits for plots.

        The bounds can be set manually in the constructor, otherwise default bounds will be generated.
        """
        raise NotImplementedError()

    def _sample(self, geometry: Geometry) -> math.Tensor:
        """ For internal use only. Use `sample()` instead. """
        raise NotImplementedError(self)

    def at(self, representation: 'SampledField', keep_extrapolation=False) -> 'SampledField':
        """
        Samples this field at the sample points of `representation`.
        The result will approximate the values of this field on the data structure of `representation`.
        
        Unlike `Field.sample()`, this method returns a `Field` object, not a `Tensor`.

        Operator alias:
            `self @ representation`.

        See Also:
            `sample()`, `reduce_sample()`, [Resampling overview](https://tum-pbs.github.io/PhiFlow/Fields.html#resampling-fields).

        Args:
            representation: Field object defining the sample points. The values of `representation` are ignored.
            keep_extrapolation: Only available if `self` is a `SampledField`.
                If True, the resampled field will inherit the extrapolation from `self` instead of `representation`.
                This can result in non-compatible value tensors for staggered grids where the tensor size depends on the extrapolation type.

        Returns:
            Field object of same type as `representation`
        """
        resampled = reduce_sample(self, representation.elements)
        extrap = self.extrapolation if isinstance(self, SampledField) and keep_extrapolation else representation.extrapolation
        return representation._op1(lambda old: extrap if isinstance(old, math.extrapolation.Extrapolation) else resampled)

    def __matmul__(self, other: 'SampledField'):  # values @ representation
        """
        Resampling operator with change of extrapolation.

        Args:
            other: instance of SampledField

        Returns:
            Copy of other with values and extrapolation from this Field.
        """
        return self.at(other, keep_extrapolation=False)

    def __rmatmul__(self, other):  # values @ representation
        if not isinstance(self, SampledField):
            return NotImplemented
        if isinstance(other, (Geometry, float, int, complex, tuple, list)):
            return self.with_values(other)
        return NotImplemented

    def __getitem__(self, item: dict) -> 'Field':
        """
        Access a slice of the Field.
        The returned `Field` may be of a different type than `self`.

        Args:
            item: `dict` mapping dimensions (`str`) to selections (`int` or `slice`)

        Returns:
            Sliced `Field`.
        """
        raise NotImplementedError(self)

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


class SampledField(Field):
    """
    Base class for fields that are sampled at specific locations such as grids or point clouds.
    """

    def __init__(self, elements: Geometry, values: Tensor, extrapolation: float or math.Extrapolation, bounds: Box or None):
        """
        Args:
          elements: Geometry object specifying the sample points and sizes
          values: values corresponding to elements
          extrapolation: values outside elements
        """
        super().__init__(bounds)
        if not isinstance(extrapolation, math.Extrapolation):
            extrapolation = math.extrapolation.ConstantExtrapolation(extrapolation)
        assert isinstance(extrapolation, Extrapolation), f"Not a valid extrapolation: {extrapolation}"
        assert isinstance(elements, Geometry), elements
        assert isinstance(values, Tensor), f"Values must be a Tensor but got {values}."
        self._elements = elements
        self._values = values
        self._extrapolation = extrapolation

    def with_values(self, values):
        """ Returns a copy of this field with `values` replaced. """
        raise NotImplementedError(self)

    def with_extrapolation(self, extrapolation: math.Extrapolation):
        """ Returns a copy of this field with `values` replaced. """
        raise NotImplementedError(self)

    @property
    def shape(self):
        raise NotImplementedError()

    def __getitem__(self: 'FieldType', item: dict) -> 'FieldType':
        raise NotImplementedError(self)

    @property
    def elements(self) -> Geometry:
        """
        Returns a geometrical representation of the discretized volume elements.
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
        if isinstance(other, Field):
            other_values = reduce_sample(other, self._elements)
            values = operator(self._values, other_values)
            extrapolation_ = operator(self._extrapolation, other.extrapolation)
            return self.with_values(values).with_extrapolation(extrapolation_)
        else:
            if isinstance(other, (tuple, list)) and len(other) == self.spatial_rank:
                other = math.tensor(other, self.points.shape['vector'])
            else:
                other = math.tensor(other)
            values = operator(self._values, other)
            return self.with_values(values)


def unstack(field: Field, dim: str) -> tuple:
    """
    Unstack `field` along one of its dimensions.
    The dimension can be batch, spatial or channel.

    Args:
        field: `Field` to unstack.
        dim: name of the dimension to unstack, must be part of `self.shape`

    Returns:
        `tuple` of `Fields`. The returned fields may be of different types than `field`.
    """
    size = field.shape.get_size(dim)
    if isinstance(size, Tensor):
        size = math.min(size)  # unstack StaggeredGrid along x or y
    return tuple([field[{dim: i}] for i in range(size)])


def sample(field: Field, geometry: Geometry) -> math.Tensor:
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
        geometry: Single or batched `phi.geom.Geometry`.

    Returns:
        Sampled values as a `phi.math.Tensor`
    """
    assert all(dim not in field.shape for dim in geometry.shape.channel)
    if isinstance(field, SampledField) and field.elements.shallow_equals(geometry) and not geometry.shape.channel:
        return field.values
    if geometry.shape.channel:
        sampled = [field._sample(p) for p in geometry.unstack(geometry.shape.channel.name)]
        return math.stack(sampled, geometry.shape.channel)
    else:
        return field._sample(geometry)


def reduce_sample(field: Field, geometry: Geometry, dim=channel('vector')) -> math.Tensor:
    """
    Similar to `sample()`, but matches channel dimensions of `geometry` with channel dimensions of this field.
    Currently, `geometry` may have at most one channel dimension.

    See Also:
        `sample()`, `Field.at()`, [Resampling overview](https://tum-pbs.github.io/PhiFlow/Fields.html#resampling-fields).

    Args:
        field: Source `Field` to sample.
        geometry: Single or batched `phi.geom.Geometry`.
        dim: Dimension of result, resulting from reduction of channel dimensions.

    Returns:
        Sampled values as a `phi.math.Tensor`
    """
    if isinstance(field, SampledField) and field.elements.shallow_equals(geometry):
        return field.values
    if geometry.shape.channel:  # Reduce this dimension
        assert geometry.shape.channel.rank == 1, "Only single-dimension reduction supported."
        if field.shape.channel.volume > 1:
            assert field.shape.channel.volume == geometry.shape.channel.volume, f"Cannot sample field with channels {field.shape.channel} at elements with channels {geometry.shape.channel}."
            components = unstack(field, field.shape.channel.name)
            sampled = [c._sample(p) for c, p in zip(components, geometry.unstack(geometry.shape.channel.name))]
        else:
            sampled = [field._sample(p) for p in geometry.unstack(geometry.shape.channel.name)]
        dim = dim._with_item_names(geometry.shape.channel.item_names)
        return math.stack(sampled, dim)
    else:  # Nothing to reduce
        return field._sample(geometry)


FieldType = TypeVar('FieldType', bound=Field)
SampledFieldType = TypeVar('SampledFieldType', bound=SampledField)
