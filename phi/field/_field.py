import warnings
from typing import TypeVar, Callable

from phi import math
from phi.geom import Geometry
from phi.math import Shape, Tensor, Extrapolation
from phi.math._shape import SPATIAL_DIM, BATCH_DIM, CHANNEL_DIM


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
        raise NotImplementedError()

    @property
    def spatial_rank(self) -> int:
        """
        Spatial rank of the field (1 for 1D, 2 for 2D, 3 for 3D).
        This is equal to the spatial rank of the `data`.
        """
        return self.shape.spatial.rank

    def _sample(self, geometry: Geometry) -> math.Tensor:
        """ For internal use only. Use `sample()` instead. """
        raise NotImplementedError(self)

    def at(self, representation: 'SampledField') -> 'SampledField':
        """
        Samples this field at the sample points of `representation`.
        The result will approximate the values of this field on the data structure of `representation`.
        
        Unlike `Field.sample()`, this method returns a `Field` object, not a `Tensor`.

        Equal to `self >> representation`.

        Args:
          representation: Field object defining the sample points. The values of `representation` are ignored.
          representation: SampledField: 

        Returns:
          Field object of same type as `representation`

        """
        resampled = reduce_sample(self, representation.elements)
        extrap = self.extrapolation if isinstance(self, SampledField) else representation.extrapolation
        return representation._op1(lambda old: extrap if isinstance(old, math.extrapolation.Extrapolation) else resampled)

    def __rshift__(self, other: 'SampledField'):
        """
        Resampling operator.

        Args:
            other: instance of SampledField

        Returns:
            Copy of other with values and extrapolation from this Field.
        """
        return self.at(other)

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
        return _FieldDim(self, name)

    def __getattr__(self, name: str) -> '_FieldDim':
        if name.startswith('_'):
            raise AttributeError(f"'{type(self)}' object has no attribute '{name}'")
        if hasattr(self.__class__, name):
            raise RuntimeError(f"Failed to get attribute '{name}' of {self.__class__}")
        return _FieldDim(self, name)

    def __repr__(self):
        return f"{self.__class__.__name__} {self.shape}"


class SampledField(Field):

    def __init__(self, elements: Geometry, values: Tensor or float or int, extrapolation: math.Extrapolation):
        """
        Base class for fields that are sampled at specific locations such as grids or point clouds.

        Args:
          elements: Geometry object specifying the sample points and sizes
          values: values corresponding to elements
          extrapolation: values outside elements
        """
        assert isinstance(extrapolation, (Extrapolation, tuple, list)), extrapolation
        assert isinstance(elements, Geometry), elements
        self._elements = elements
        self._values = math.wrap(values)
        self._extrapolation = extrapolation

    def with_(self,
              elements: Geometry or None = None,
              values: Tensor = None,
              extrapolation: math.Extrapolation = None,
              **other_attributes) -> 'SampledFieldType':
        """ Creates a copy of this field with one or more properties changed. `None` keeps the current value. """
        raise NotImplementedError(self)

    @property
    def shape(self):
        raise NotImplementedError()

    def __getitem__(self, item: dict) -> 'Field':
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

    @property
    def shape(self) -> Shape:
        return self._shape

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
        return self.with_(values=values, extrapolation=extrapolation_)

    def _op2(self, other, operator) -> 'SampledField':
        if isinstance(other, Field):
            other_values = reduce_sample(other, self._elements)
            values = operator(self._values, other_values)
            extrapolation_ = operator(self._extrapolation, other.extrapolation)
            return self.with_(values=values, extrapolation=extrapolation_)
        else:
            other = math.tensor(other)
            values = operator(self._values, other)
            return self.with_(values=values)


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
    return tuple(field[{dim: i}] for i in range(size))


def sample(field: Field, geometry: Geometry) -> math.Tensor:
    """
    Computes the field value inside the volume of the (batched) `geometry`.

    The field value may be determined by integrating over the volume, sampling the central value or any other way.

    The batch dimensions of `geometry` are matched with this field.
    The `geometry` must not share any channel dimensions with this field.
    Spatial dimensions of `geometry` can be used to sample a grid of geometries.

    See Also:
        `reduce_sample()`, `Field.at()`.

    Args:
        field: Source `Field` to sample.
        geometry: Single or batched `phi.geom.Geometry`.

    Returns:
        Sampled values as a `phi.math.Tensor`
    """
    assert all(dim not in field.shape for dim in geometry.shape.channel)
    if isinstance(field, SampledField) and field.elements.shallow_equals(geometry) and 'vector_' not in geometry.shape:
        return field.values
    assert 'vector' not in geometry.shape
    if 'vector_' in geometry.shape:
        sampled = [field._sample(p) for p in geometry.unstack('vector_')]
        return math.channel_stack(sampled, 'vector_')
    else:
        return field._sample(geometry)


def reduce_sample(field: Field, geometry: Geometry) -> math.Tensor:
    """
    Similar to `sample()`, but matches an optional `vector_` dimension of `geometry` with the `vector` dimension of this field.

    See Also:
        `sample()`, `Field.at()`.

    Args:
        field: Source `Field` to sample.
        geometry: Single or batched `phi.geom.Geometry`.

    Returns:
        Sampled values as a `phi.math.Tensor`
    """
    if isinstance(field, SampledField) and field.elements.shallow_equals(geometry):
        return field.values
    assert 'vector' not in geometry.shape
    if 'vector_' in geometry.shape:
        components = unstack(field, 'vector') if 'vector' in field.shape else (field,) * geometry.shape.get_size('vector_')
        sampled = [c._sample(p) for c, p in zip(components, geometry.unstack('vector_'))]
        return math.channel_stack(sampled, 'vector')
    else:
        return field._sample(geometry)


class _FieldDim:

    def __init__(self, field: Field, name: str):
        self.field = field
        self.name = name

    @property
    def exists(self):
        return self.name in self.field.shape

    def __str__(self):
        return self.name

    def unstack(self, size: int or None = None):
        if size is None:
            return unstack(self.field, self.name)
        else:
            if self.exists:
                unstacked = unstack(self.field, self.name)
                assert len(unstacked) == size, f"Size of dimension {self.name} does not match {size}."
                return unstacked
            else:
                return (self.field,) * size

    @property
    def size(self):
        return self.field.shape.get_size(self.name)

    @property
    def dim_type(self):
        return self.field.shape.get_type(self.name)

    @property
    def is_spatial(self):
        return self.dim_type == SPATIAL_DIM

    @property
    def is_batch(self):
        return self.dim_type == BATCH_DIM

    @property
    def is_channel(self):
        return self.dim_type == CHANNEL_DIM

    def __getitem__(self, item):
        if isinstance(item, str):
            item = self.field.shape.spatial.index(item)
        return self.field[{self.name: item}]

    def __call__(self, *args, **kwargs):
        raise TypeError(f"Method {type(self.field).__name__}.{self.name}() does not exist.")


FieldType = TypeVar('FieldType', bound=Field)
SampledFieldType = TypeVar('SampledFieldType', bound=SampledField)
