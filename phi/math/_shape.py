from __future__ import annotations

import warnings

from phi import math


BATCH_DIM = 'batch'
SPATIAL_DIM = 'spatial'
CHANNEL_DIM = 'channel'


class Shape:
    """Shapes enumerate dimensions, each consisting of a name, size and type."""

    def __init__(self, sizes: tuple or list, names: tuple or list, types: tuple or list):
        """
        To construct a Shape manually, use `shape()` instead.
        This constructor is meant for internal use only.

        Construct a Shape from sizes, names and types sequences.
        All arguments must have same length.

        To create a Shape with inferred dimension types, use :func:`shape(**dims)` instead.

        Args:
            sizes: Ordered dimension sizes
            names: Ordered dimension names, either strings (spatial, batch) or integers (channel)
            types: Ordered types, all values should be one of (CHANNEL_DIM, SPATIAL_DIM, BATCH_DIM)
        """
        assert len(sizes) == len(names) == len(types), f"sizes={sizes} ({len(sizes)}), names={names} ({len(names)}), types={types} ({len(types)})"
        self.sizes = tuple(sizes)
        """ Ordered dimension sizes as `tuple`  """
        self.names = tuple(names)
        """ Ordered dimension names as `tuple` of `str` """
        assert all(isinstance(n, str) for n in names), f"All names must be of type string but got {names}"
        self.types = tuple(types)  # undocumented, may be private

    @property
    def named_sizes(self):
        """
        For iterating over names and sizes

            for name, size in shape.named_sizes:

        Returns:
            iterable
        """
        return zip(self.names, self.sizes)

    @property
    def spatial_dict(self) -> dict:
        """ Ordered dictionary mapping dimension names to their respective sizes for all spatial dimensions. """
        return {n: s for s, n, t in zip(self.sizes, self.names, self.types) if t == SPATIAL_DIM}

    @property
    def dimensions(self):
        """
        For iterating over sizes, names and types.
        Meant for internal use.

        See `Shape.named_sizes()`.
        """
        return zip(self.sizes, self.names, self.types)

    def __len__(self):
        return len(self.sizes)

    def __contains__(self, item):
        return item in self.names

    def index(self, name: str or list or tuple or Shape or None):
        """
        Finds the index of the dimension(s) within this Shape.

        Args:
          name: dimension name or sequence thereof, including Shape object
          name: str or list or tuple or Shape: 

        Returns:
          single index or sequence of indices

        """
        if name is None:
            return None
        if isinstance(name, (list, tuple)):
            return tuple(self.index(n) for n in name)
        if isinstance(name, Shape):
            return tuple(self.index(n) for n in name.names)
        for idx, dim_name in enumerate(self.names):
            if dim_name == name:
                return idx
        raise ValueError("Shape %s does not contain dimension with name '%s'" % (self, name))

    def indices(self, names: tuple or list or Shape):
        if isinstance(names, (list, tuple)):
            return tuple(self.index(n) for n in names)
        if isinstance(names, Shape):
            return tuple(self.index(n) for n in names.names)
        else:
            raise ValueError(names)

    def get_size(self, dim: str or tuple or list):
        """
        Args:
            dim: dimension name or sequence of dimension names

        Returns:
            size associated with `dim`
        """
        if isinstance(dim, str):
            return self.sizes[self.names.index(dim)]
        elif isinstance(dim, (tuple, list)):
            return tuple(self.get_size(n) for n in dim)
        else:
            raise ValueError(dim)

    def __getattr__(self, name):
        if name == 'names':
            raise AssertionError("Attribute missing: %s" % name)
        if name in self.names:
            return self.get_size(name)
        raise AttributeError("Shape has no attribute '%s'" % (name,))

    def get_type(self, name: str or tuple or list or Shape):
        if isinstance(name, str):
            return self.types[self.names.index(name)]
        elif isinstance(name, (tuple, list)):
            return tuple(self.get_type(n) for n in name)
        elif isinstance(name, Shape):
            return tuple(self.get_type(n) for n in name.names)
        else:
            raise ValueError(name)

    def __getitem__(self, selection):
        if isinstance(selection, int):
            return self.sizes[selection]
        elif isinstance(selection, slice):
            return Shape(self.sizes[selection], self.names[selection], self.types[selection])
        return Shape([self.sizes[i] for i in selection], [self.names[i] for i in selection], [self.types[i] for i in selection])

    @property
    def batch(self) -> Shape:
        """
        Filters this shape, returning only the batch dimensions as a new `Shape` object.

        See also:

        * Dimension type filters: `Shape.batch`, `Shape.spatial`, `Shape.channel`, `Shape.non_batch`, `Shape.non_spatial`, `Shape.non_channel`
        * Dimension size filters: `Shape.singleton`, `Shape.non_singleton`, `Shape.zero`, `Shape.non_zero`, `Shape.undefined`, `Shape.defined`

        Returns:
            New `Shape` object
        """
        return self[[i for i, t in enumerate(self.types) if t == BATCH_DIM]]

    @property
    def non_batch(self) -> Shape:
        """
        Filters this shape, returning only the spatial and channel dimensions as a new `Shape` object.

        See also:

        * Dimension type filters: `Shape.batch`, `Shape.spatial`, `Shape.channel`, `Shape.non_batch`, `Shape.non_spatial`, `Shape.non_channel`
        * Dimension size filters: `Shape.singleton`, `Shape.non_singleton`, `Shape.zero`, `Shape.non_zero`, `Shape.undefined`, `Shape.defined`

        Returns:
            New `Shape` object
        """
        return self[[i for i, t in enumerate(self.types) if t != BATCH_DIM]]

    @property
    def spatial(self) -> Shape:
        """
        Filters this shape, returning only the spatial dimensions as a new `Shape` object.

        See also:

        * Dimension type filters: `Shape.batch`, `Shape.spatial`, `Shape.channel`, `Shape.non_batch`, `Shape.non_spatial`, `Shape.non_channel`
        * Dimension size filters: `Shape.singleton`, `Shape.non_singleton`, `Shape.zero`, `Shape.non_zero`, `Shape.undefined`, `Shape.defined`

        Returns:
            New `Shape` object
        """
        return self[[i for i, t in enumerate(self.types) if t == SPATIAL_DIM]]

    @property
    def non_spatial(self) -> Shape:
        """
        Filters this shape, returning only the batch and channel dimensions as a new `Shape` object.

        See also:

        * Dimension type filters: `Shape.batch`, `Shape.spatial`, `Shape.channel`, `Shape.non_batch`, `Shape.non_spatial`, `Shape.non_channel`
        * Dimension size filters: `Shape.singleton`, `Shape.non_singleton`, `Shape.zero`, `Shape.non_zero`, `Shape.undefined`, `Shape.defined`

        Returns:
            New `Shape` object
        """
        return self[[i for i, t in enumerate(self.types) if t != SPATIAL_DIM]]

    @property
    def channel(self) -> Shape:
        """
        Filters this shape, returning only the channel dimensions as a new `Shape` object.

        See also:

        * Dimension type filters: `Shape.batch`, `Shape.spatial`, `Shape.channel`, `Shape.non_batch`, `Shape.non_spatial`, `Shape.non_channel`
        * Dimension size filters: `Shape.singleton`, `Shape.non_singleton`, `Shape.zero`, `Shape.non_zero`, `Shape.undefined`, `Shape.defined`

        Returns:
            New `Shape` object
        """
        return self[[i for i, t in enumerate(self.types) if t == CHANNEL_DIM]]

    @property
    def non_channel(self) -> Shape:
        """
        Filters this shape, returning only the batch and spatial dimensions as a new `Shape` object.

        See also:

        * Dimension type filters: `Shape.batch`, `Shape.spatial`, `Shape.channel`, `Shape.non_batch`, `Shape.non_spatial`, `Shape.non_channel`
        * Dimension size filters: `Shape.singleton`, `Shape.non_singleton`, `Shape.zero`, `Shape.non_zero`, `Shape.undefined`, `Shape.defined`

        Returns:
            New `Shape` object
        """
        return self[[i for i, t in enumerate(self.types) if t != CHANNEL_DIM]]

    @property
    def singleton(self) -> Shape:
        """
        Filters this shape, returning only the dimensions with a size of 1 as a new `Shape` object.

        See also:

        * Dimension type filters: `Shape.batch`, `Shape.spatial`, `Shape.channel`, `Shape.non_batch`, `Shape.non_spatial`, `Shape.non_channel`
        * Dimension size filters: `Shape.singleton`, `Shape.non_singleton`, `Shape.zero`, `Shape.non_zero`, `Shape.undefined`, `Shape.defined`

        Returns:
            New `Shape` object
        """
        return self[[i for i, size in enumerate(self.sizes) if size == 1]]

    @property
    def non_singleton(self) -> Shape:
        """
        Filters this shape, returning only the dimensions with a size different from 1 as a new `Shape` object.

        See also:

        * Dimension type filters: `Shape.batch`, `Shape.spatial`, `Shape.channel`, `Shape.non_batch`, `Shape.non_spatial`, `Shape.non_channel`
        * Dimension size filters: `Shape.singleton`, `Shape.non_singleton`, `Shape.zero`, `Shape.non_zero`, `Shape.undefined`, `Shape.defined`

        Returns:
            New `Shape` object
        """
        return self[[i for i, size in enumerate(self.sizes) if size != 1]]

    @property
    def zero(self) -> Shape:
        """
        Filters this shape, returning only the dimensions with a size of 0 as a new `Shape` object.

        See also:

        * Dimension type filters: `Shape.batch`, `Shape.spatial`, `Shape.channel`, `Shape.non_batch`, `Shape.non_spatial`, `Shape.non_channel`
        * Dimension size filters: `Shape.singleton`, `Shape.non_singleton`, `Shape.zero`, `Shape.non_zero`, `Shape.undefined`, `Shape.defined`

        Returns:
            New `Shape` object
        """
        return self[[i for i, size in enumerate(self.sizes) if size == 0]]

    @property
    def non_zero(self) -> Shape:
        """
        Filters this shape, returning only the dimensions with a size different from 0 as a new `Shape` object.

        See also:

        * Dimension type filters: `Shape.batch`, `Shape.spatial`, `Shape.channel`, `Shape.non_batch`, `Shape.non_spatial`, `Shape.non_channel`
        * Dimension size filters: `Shape.singleton`, `Shape.non_singleton`, `Shape.zero`, `Shape.non_zero`, `Shape.undefined`, `Shape.defined`

        Returns:
            New `Shape` object
        """
        return self[[i for i, size in enumerate(self.sizes) if size != 0]]

    @property
    def undefined(self) -> Shape:
        """
        Filters this shape, returning only the dimensions with a size of `None` as a new `Shape` object.

        See also:

        * Dimension type filters: `Shape.batch`, `Shape.spatial`, `Shape.channel`, `Shape.non_batch`, `Shape.non_spatial`, `Shape.non_channel`
        * Dimension size filters: `Shape.singleton`, `Shape.non_singleton`, `Shape.zero`, `Shape.non_zero`, `Shape.undefined`, `Shape.defined`

        Returns:
            New `Shape` object
        """
        return self[[i for i, size in enumerate(self.sizes) if size is None]]

    @property
    def defined(self) -> Shape:
        """
        Filters this shape, returning only the dimensions with a size different from `None` as a new `Shape` object.

        See also:

        * Dimension type filters: `Shape.batch`, `Shape.spatial`, `Shape.channel`, `Shape.non_batch`, `Shape.non_spatial`, `Shape.non_channel`
        * Dimension size filters: `Shape.singleton`, `Shape.non_singleton`, `Shape.zero`, `Shape.non_zero`, `Shape.undefined`, `Shape.defined`

        Returns:
            New `Shape` object
        """
        return self[[i for i, size in enumerate(self.sizes) if size is not None]]

    def unstack(self, name='dims'):
        if name == 'dims':
            return tuple(Shape([self.sizes[i]], [self.names[i]], [self.types[i]]) for i in range(self.rank))
        if name not in self:
            return tuple([self])
        else:
            from ._tensors import Tensor
            inner = self.without(name)
            sizes = []
            dim_size = self.get_size(name)
            for size in inner.sizes:
                if isinstance(size, Tensor) and name in size.shape:
                    sizes.append(size.unstack(name))
                    dim_size = size.shape.get_size(name)
                else:
                    sizes.append(size)
            assert isinstance(dim_size, int)
            shapes = tuple(Shape([int(size[i]) if isinstance(size, tuple) else size for size in sizes], inner.names, inner.types) for i in range(dim_size))
            return shapes

    @property
    def name(self) -> str:
        """ Only for shapes with a single dimension. Returns the name of the dimension. """
        assert self.rank == 1, 'Shape.name is only defined for shapes of rank 1.'
        return self.names[0]

    @property
    def is_batch(self) -> bool:
        """ Tests if all dimensions are of type *batch* """
        return all([t == BATCH_DIM for t in self.types])

    @property
    def is_spatial(self) -> bool:
        """ Tests if all dimensions are of type *spatial* """
        return all([t == SPATIAL_DIM for t in self.types])

    @property
    def is_channel(self) -> bool:
        """ Tests if all dimensions are of type *channel* """
        return all([t == CHANNEL_DIM for t in self.types])

    def mask(self, names: tuple or list or set):
        """
        Returns a binary sequence corresponding to the names of this Shape.
        A value of 1 means that a dimension of this Shape is contained in `names`.

        Args:
          names: collection of dimension
          names: tuple or list or set: 

        Returns:
          binary sequence

        """
        if isinstance(names, str):
            names = [names]
        mask = [1 if name in names else 0 for name in self.names]
        return tuple(mask)

    def __repr__(self):
        strings = ['%s=%s' % (name, size) for size, name, _ in self.dimensions]
        return '(' + ', '.join(strings) + ')'

    def __eq__(self, other):
        if not isinstance(other, Shape):
            return False
        return self.names == other.names and self.types == other.types and self.sizes == other.sizes

    def __ne__(self, other):
        return not self == other

    def normal_order(self):
        sizes = self.batch.sizes + self.spatial.sizes + self.channel.sizes
        names = self.batch.names + self.spatial.names + self.channel.names
        types = self.batch.types + self.spatial.types + self.channel.types
        return Shape(sizes, names, types)

    def reorder(self, names: tuple or list):
        assert len(names) == self.rank
        order = [self.index(n) for n in names]
        return self[order]

    def order_group(self, names: tuple or list or Shape):
        if isinstance(names, Shape):
            names = names.names
        order = []
        for name in self.names:
            if name not in order:
                if name in names:
                    order.extend(names)
                else:
                    order.append(name)
        return order

    def combined(self, other: Shape, combine_spatial=False) -> Shape:
        """
        Returns a Shape object that both `self` and `other` can be broadcast to.
        If `self` and `other` are incompatible, raises a ValueError.

        Args:
          other: Shape
          other: Shape: 
          combine_spatial:  (Default value = False)

        Returns:
          combined shape
          :raise: ValueError if shapes don't match

        """
        return combine_safe(self, other, check_exact=[] if combine_spatial else [SPATIAL_DIM])

    def __and__(self, other):
        return combine_safe(self, other, check_exact=[SPATIAL_DIM])

    def expand_batch(self, size, name: str, pos=None) -> Shape:
        return self.expand(size, name, BATCH_DIM, pos)

    def expand_spatial(self, size, name: str, pos=None) -> Shape:
        return self.expand(size, name, SPATIAL_DIM, pos)

    def expand_channel(self, size, name: str, pos=None) -> Shape:
        return self.expand(size, name, CHANNEL_DIM, pos)

    def expand(self, size, name: str, dim_type: str, pos=None) -> Shape:
        """
        Add a dimension to the shape.
        
        The resulting shape has linear indices.

        Args:
          size: 
          name: str: 
          dim_type: str: 
          pos:  (Default value = None)

        Returns:

        """
        if pos is None:
            same_type_dims = self[[i for i, t in enumerate(self.types) if t == dim_type]]
            if len(same_type_dims) > 0:
                pos = self.index(same_type_dims.names[0])
            else:
                pos = {BATCH_DIM: 0, SPATIAL_DIM: self.batch.rank, CHANNEL_DIM: self.rank + 1}[dim_type]
        elif pos < 0:
            pos += self.rank + 1
        sizes = list(self.sizes)
        names = list(self.names)
        types = list(self.types)
        sizes.insert(pos, size)
        names.insert(pos, name)
        types.insert(pos, dim_type)
        return Shape(sizes, names, types)

    def extend(self, other: Shape, pos=-1) -> Shape:
        if pos == -1:
            return Shape(self.sizes + other.sizes, self.names + other.names, self.types + other.types)
        elif pos == None:
            result = self
            for size, name, dim_type in other.dimensions:
                result = result.expand(size, name, dim_type)
            return result
        else:
            raise NotImplementedError(pos)

    def without(self, dims: str or tuple or list or Shape or None) -> Shape:
        """
        Builds a new shape from this one that is missing all given dimensions.
        Dimensions in `dims` that are not part of this Shape are ignored.
        
        The complementary operation is :func:`Shape.only`.

        Args:
          dims: single dimension (str) or collection of dimensions (tuple, list, Shape)
          dims: str or tuple or list or Shape or None: 

        Returns:
          Shape without specified dimensions

        """
        if isinstance(dims, str):
            return self[[i for i in range(self.rank) if self.names[i] != dims]]
        if isinstance(dims, (tuple, list)):
            return self[[i for i in range(self.rank) if self.names[i] not in dims]]
        elif isinstance(dims, Shape):
            return self[[i for i in range(self.rank) if self.names[i] not in dims.names]]
        elif dims is None:  # subtract all
            return EMPTY_SHAPE
        else:
            raise ValueError(dims)

    reduce = without

    def only(self, dims: str or tuple or list or Shape):
        """
        Builds a new shape from this one that only contains the given dimensions.
        Dimensions in `dims` that are not part of this Shape are ignored.
        
        The complementary operation is :func:`Shape.without`.

        Args:
          dims: single dimension (str) or collection of dimensions (tuple, list, Shape)
          dims: str or tuple or list or Shape: 

        Returns:
          Shape containing only specified dimensions

        """
        if isinstance(dims, str):
            return self[[i for i in range(self.rank) if self.names[i] == dims]]
        if isinstance(dims, (tuple, list)):
            return self[[i for i in range(self.rank) if self.names[i] in dims]]
        elif isinstance(dims, Shape):
            return self[[i for i in range(self.rank) if self.names[i] in dims.names]]
        elif dims is None:  # keep all
            return self
        else:
            raise ValueError(dims)

    def select(self, *names):
        indices = [self.index(name) for name in names]
        return self[indices]

    @property
    def rank(self) -> int:
        """
        Returns the number of dimensions.
        Equal to `len(shape)`.

        See `Shape.is_empty`, `Shape.batch_rank`, `Shape.spatial_rank`, `Shape.channel_rank`.
        """
        return len(self.sizes)

    @property
    def batch_rank(self) -> int:
        """ Number of batch dimensions """
        r = 0
        for ty in self.types:
            if ty == BATCH_DIM:
                r += 1
        return r

    @property
    def spatial_rank(self) -> int:
        """ Number of spatial dimensions """
        r = 0
        for ty in self.types:
            if ty == SPATIAL_DIM:
                r += 1
        return r

    @property
    def channel_rank(self) -> int:
        """ Number of channel dimensions """
        r = 0
        for ty in self.types:
            if ty == CHANNEL_DIM:
                r += 1
        return r

    def to_batch(self, dims: tuple or list or None = None) -> Shape:
        """
        Returns a shape like this Shape but with `dims` being of type `batch`.
        
        Leaves this Shape object untouched.

        Args:
          dims: sequence of dimension names to convert or None to convert all dimensions
          dims: tuple or list or None:  (Default value = None)

        Returns:
          new Shape object

        """
        if dims is None:
            return Shape(self.sizes, self.names, [BATCH_DIM] * self.rank)
        else:
            return Shape(self.sizes, self.names, [BATCH_DIM if dim in dims else self.types[i] for i, dim in enumerate(self.names)])

    @property
    def well_defined(self):
        """ Returns True if no dimension is `None`. """
        return None not in self.sizes

    @property
    def shape(self, list_dim='dims') -> Shape:
        """
        Returns the shape of this `Shape`.
        The returned shape will always contain the dimension `list_dim` with a size equal to the `Shape.rank` of this shape.

        Sizes of type `Tensor` can cause the result to have additional dimensions.

        Args:
            list_dim: name of dimension listing the dimensions of this shape

        Returns:
            second order shape
        """
        from phi.math import Tensor
        shape = Shape([self.rank], [list_dim], [CHANNEL_DIM])
        for size in self.sizes:
            if isinstance(size, Tensor):
                shape = shape & size.shape
        return shape

    @property
    def is_non_uniform(self) -> bool:
        """
        A shape is non-uniform if the size of any dimension varies along another dimension.

        See `Shape.shape`.
        """
        from phi.math import Tensor
        for size in self.sizes:
            if isinstance(size, Tensor) and size.rank > 0:
                return True
        return False

    def with_sizes(self, sizes: tuple or list or Shape):
        if isinstance(sizes, Shape):
            sizes = [sizes.get_size(dim) if dim in sizes else self.sizes[i] for i, dim in enumerate(self.names)]
            return Shape(sizes, self.names, self.types)
        else:
            assert len(sizes) == len(self.sizes)
            return Shape(sizes, self.names, self.types)

    def with_size(self, name, size):
        new_sizes = list(self.sizes)
        new_sizes[self.index(name)] = size
        return self.with_sizes(new_sizes)

    def with_names(self, names: str or tuple or list):
        if isinstance(names, str):
            names = parse_dim_names(names, self.rank)
            names = [n if n is not None else o for n, o in zip(names, self.names)]
        return Shape(self.sizes, names, self.types)

    def with_types(self, types: Shape):
        return Shape(self.sizes, self.names, [types.get_type(name) if name in types else self_type for name, self_type in zip(self.names, self.types)])

    def perm(self, names):
        assert set(names) == set(self.names), 'names must match existing dimensions %s but got %s' % (self.names, names)
        perm = [self.names.index(name) for name in names]
        return perm

    @property
    def volume(self) -> int or None:
        """
        Returns the total number of values contained in a tensor of this shape.
        This is the product of all dimension sizes.

        Returns:
            volume as `int` or `Tensor` or `None` if the shape is not `Shape.well_defined`
        """
        result = 1
        for size in self.sizes:
            if size is None:
                return None
            result *= size
        return result

    @property
    def is_empty(self) -> bool:
        """ True if this shape has no dimensions. Equivalent to `Shape.rank` `== 0`. """
        return len(self.sizes) == 0

    def order(self, sequence, default=None) -> Shape:
        """
        If sequence is a dict with dimension names as keys, orders its values according to this shape.
        
        Otherwise, the sequence is returned unchanged.

        Args:
          sequence(dict or list or tuple): sequence or dict to be ordered
          default: default value used for dimensions not contained in sequence

        Returns:
          ordered sequence of values
        """
        if isinstance(sequence, dict):
            result = [sequence.get(name, default) for name in self.names]
            return result
        if isinstance(sequence, (tuple, list)):
            assert len(sequence) == self.rank
            return sequence
        else:  # just a constant
            return sequence

    def sequence_get(self, sequence, name):
        if isinstance(sequence, dict):
            return sequence[name]
        if isinstance(sequence, (tuple, list)):
            assert len(sequence) == self.rank
            return sequence[self.names.index(name)]
        if math.is_tensor(sequence):
            assert math.staticshape(sequence) == (self.rank,)
            return sequence[self.names.index(name)]
        else:  # just a constant
            return sequence

    def after_pad(self, widths: dict):
        sizes = list(self.sizes)
        for dim, (lo, up) in widths.items():
            sizes[self.index(dim)] += lo + up
        return Shape(sizes, self.names, self.types)

    def after_gather(self, selection: dict):
        result = self
        for name, selection in selection.items():
            if isinstance(selection, int):
                result = result.without(name)
            elif isinstance(selection, slice):
                assert selection.step is None
                start = selection.start or 0
                stop = selection.stop or self.get_size(name)
                if stop < 0:
                    stop += self.get_size(name)
                result = result.with_size(name, stop - start)
            else:
                raise NotImplementedError(f"{type(selection)} not supported. Only (int, slice) allowed.")
        return result

    def meshgrid(self):
        """Builds a sequence containing all multi-indices within a tensor of this shape."""
        indices = [0] * self.rank
        while True:
            yield {name: index for name, index in zip(self.names, indices)}
            for i in range(self.rank-1, -1, -1):
                indices[i] = (indices[i] + 1) % self.sizes[i]
                if indices[i] != 0:
                    break
            else:
                return

    product = meshgrid

    def __add__(self, other):
        return self._op1(other, lambda s, o: s + o)

    def __radd__(self, other):
        return self._op1(other, lambda s, o: o + s)

    def __sub__(self, other):
        return self._op1(other, lambda s, o: s - o)

    def __rsub__(self, other):
        return self._op1(other, lambda s, o: o - s)

    def __mul__(self, other):
        return self._op1(other, lambda s, o: s * o)

    def __rmul__(self, other):
        return self._op1(other, lambda s, o: o * s)

    def _op1(self, other, fun):
        if isinstance(other, int):
            return Shape([fun(s, other) for s in self.sizes], self.names, self.types)
        elif isinstance(other, Shape):
            assert self.names == other.names, f"{self.names, other.names}"
            return Shape([fun(s, o) for s, o in zip(self.sizes, other.sizes)], self.names, self.types)
        else:
            return NotImplemented

    def __hash__(self):
        return hash(self.sizes)


EMPTY_SHAPE = Shape((), (), ())


class IncompatibleShapes(ValueError):
    def __init__(self, message, *shapes: Shape):
        ValueError.__init__(self, message, *shapes)


def parse_dim_names(obj: str or tuple or list or Shape, count: int) -> tuple:
    if isinstance(obj, str):
        parts = obj.split(',')
        result = []
        for part in parts:
            part = part.strip()
            if part == '...':
                result.extend([None] * (count - len(parts) + 1))
            elif part == ':':
                result.append(None)
            else:
                result.append(part)
        assert len(result) == count, f"Number of specified names in '{obj}' does not match number of dimensions ({count})"
        return tuple(result)
    elif isinstance(obj, Shape):
        assert len(obj) == count, f"Number of specified names in {obj} does not match number of dimensions ({count})"
        return obj.names
    elif isinstance(obj, (tuple, list)):
        assert len(obj) == count, f"Number of specified names in {obj} does not match number of dimensions ({count})"
        return tuple(obj)
    raise ValueError(obj)


def parse_dim_order(order: str or tuple or list or Shape or None) -> tuple or None:
    if order is None:
        return None
    elif isinstance(order, Shape):
        return order.names
    elif isinstance(order, (tuple, list)):
        return order
    elif isinstance(order, str):
        parts = order.split(',')
        parts = [p.strip() for p in parts]
        return tuple(parts)


def shape(**dims: int) -> Shape:
    """
    Creates a Shape from the dimension names and their respective sizes.
    
    Dimension types are inferred from the names according to the following rules:
    
    * single letter -> spatial dimension
    * starts with 'vector' -> channel dimension
    * else -> batch dimension

    Args:
      dims: names -> size
      **dims: int: 

    Returns:
      Shape

    """
    types = []
    for name, size in dims.items():
        types.append(_infer_dim_type_from_name(name))
    return Shape(dims.values(), dims.keys(), types)


def _infer_dim_type_from_name(name):
    if len(name) == 1:
        return SPATIAL_DIM
    elif name.startswith('vector'):
        return CHANNEL_DIM
    else:
        return BATCH_DIM


def _infer_dim_group_counts(rank, constraints: list):
    known_sum = sum([dim or 0 for dim in constraints])
    unknown_count = sum([1 if dim is None else 0 for dim in constraints])
    if known_sum == rank:
        return [dim or 0 for dim in constraints]
    if unknown_count == 1:
        return [rank - known_sum if dim is None else dim for dim in constraints]
    return None


def batch_shape(sizes: Shape or dict or tuple or list, names: tuple or list = None):
    """
    Creates a Shape with the following properties:
    
    * All dimensions are of type 'batch'
    * The shape's `names` match `names`, if provided
    
    Depending on the type of `sizes`, returns
    
    * Shape -> (reordered) spatial sub-shape
    * dict[dim: str -> size] -> (reordered) shape with given names and sizes
    * tuple/list of sizes -> matches names to sizes and keeps order

    Args:
      sizes: list of integers or dict or Shape
      names: Order of dimensions. Optional if isinstance(sizes, (dict, Shape))
      sizes: Shape or dict or tuple or list: 
      names: tuple or list:  (Default value = None)

    Returns:
      Shape containing only spatial dimensions

    """
    return _pure_shape(sizes, names, BATCH_DIM)


def spatial_shape(sizes: Shape or dict or tuple or list, names: tuple or list = None) -> Shape:
    """
    Creates a Shape with the following properties:
    
    * All dimensions are of type 'spatial'
    * The shape's `names` match `names`, if provided
    
    Depending on the type of `sizes`, returns
    
    * Shape -> (reordered) spatial sub-shape
    * dict[dim: str -> size] -> (reordered) shape with given names and sizes
    * tuple/list of sizes -> matches names to sizes and keeps order

    Args:
      sizes: list of integers or dict or Shape
      names: Order of dimensions. Optional if isinstance(sizes, (dict, Shape))
      sizes: Shape or dict or tuple or list: 
      names: tuple or list:  (Default value = None)

    Returns:
      Shape containing only spatial dimensions

    """
    return _pure_shape(sizes, names, SPATIAL_DIM)


def channel_shape(sizes: Shape or dict or list or tuple, names: tuple or list = None) -> Shape:
    """
    Creates a Shape with the following properties:
    
    * All dimensions are of type 'channel'
    * The shape's `names` match `names`, if provided
    
    Depending on the type of `sizes`, returns
    
    * Shape -> (reordered) spatial sub-shape
    * dict[dim: str -> size] -> (reordered) shape with given names and sizes
    * tuple/list of sizes -> matches names to sizes and keeps order

    Args:
      sizes: list of integers or dict or Shape
      names: Order of dimensions. Optional if isinstance(sizes, (dict, Shape))
      sizes: Shape or dict or list or tuple: 
      names: tuple or list:  (Default value = None)

    Returns:
      Shape containing only spatial dimensions

    """
    return _pure_shape(sizes, names, SPATIAL_DIM)


def _pure_shape(sizes: Shape or dict or tuple or list, names: tuple or list, dim_type: str) -> Shape:
    """
    Creates a Shape with the following properties:
    
    * All dimensions are of type `dim_type`
    * The shape's `names` match `names`, if provided
    
    Depending on the type of `sizes`, returns
    
    * Shape -> (reordered) spatial sub-shape
    * dict[dim: str -> size] -> (reordered) shape with given names and sizes
    * tuple/list of sizes -> matches names to sizes and keeps order

    Args:
      sizes: list of integers or dict or Shape
      names: Order of dimensions. Optional if isinstance(sizes, (dict, Shape))
      sizes: Shape or dict or tuple or list: 
      names: tuple or list: 
      dim_type: str: 

    Returns:
      Shape containing only spatial dimensions

    """
    if isinstance(sizes, Shape):
        s = sizes[[i for i, t in enumerate(sizes.types) if t == dim_type]]
        if names is None:
            return s
        else:
            assert s.rank == len(names)
            return s.reorder(names)
    elif isinstance(sizes, dict):
        s = Shape(sizes.values(), sizes.keys(), (dim_type,) * len(sizes))
        if names is None:
            return s
        else:
            assert s.rank == len(names)
            return s.reorder(names)
    elif isinstance(sizes, (tuple, list)):
        assert names is not None
        assert len(names) == len(sizes)
        return Shape(sizes, names, (dim_type,) * len(sizes))
    else:
        raise ValueError(sizes)


def check_singleton(shape):
    for i, (size, dim_type) in enumerate(zip(shape.sizes, shape.types)):
        if isinstance(size, int) and size == 1 and dim_type != SPATIAL_DIM and check_singleton:
            warnings.warn("Dimension '%s' at index %d of shape %s has size 1. Is this intentional? Singleton dimensions are not supported." % (shape.names[i], i, shape.sizes))


def combine_safe(*shapes: Shape, check_exact: tuple or list = ()):
    _check_exact_match(*shapes, check_exact)
    sizes = list(shapes[0].sizes)
    names = list(shapes[0].names)
    types = list(shapes[0].types)
    for other in shapes[1:]:
        for size, name, type in other.dimensions:
            if name not in names:
                if type in types:
                    index = len(types) - types[::-1].index(type)
                elif type == BATCH_DIM:
                    index = 0
                elif type == CHANNEL_DIM:
                    index = len(names)
                elif type == SPATIAL_DIM:
                    index = min([len(names), *[i for i in range(len(names)) if types[i] == CHANNEL_DIM]])
                else:
                    raise ValueError(type)
                names.insert(index, name)
                sizes.insert(index, size)
                types.insert(index, type)
            else:
                existing_size = sizes[names.index(name)]
                if size != existing_size:
                    raise IncompatibleShapes(*shapes)
    return Shape(sizes, names, types)


def _check_exact_match(*shapes: Shape, check_types: tuple or list = ()):
    shape0 = shapes[0]
    for check_type in check_types:
        dims0 = shape0[[i for i, t in enumerate(shape0.types) if t == check_type]]
        for other in shapes[1:]:
            dims_other = other[[i for i, t in enumerate(shape0.types) if t == check_type]]
            if len(dims0) == 0:
                dims0 = dims_other
            elif len(dims_other) > 0:
                if dims0 != dims_other:
                    raise IncompatibleShapes(f"Incompatible dimensions of type '{check_type}", *shapes)


def shape_stack(stack_dim: str, stack_type: str, *shapes: Shape):
    names = list(shapes[0].names)
    types = list(shapes[0].types)
    for other in shapes[1:]:
        for size, name, type in other.dimensions:
            if name not in names:
                if type in types:
                    index = len(types) - types[::-1].index(type)
                elif type == BATCH_DIM:
                    index = 0
                elif type == CHANNEL_DIM:
                    index = len(names)
                elif type == SPATIAL_DIM:
                    index = min([len(names), *[i for i in range(len(names)) if types[i] == CHANNEL_DIM]])
                else:
                    raise ValueError(type)
                names.insert(index, name)
                types.insert(index, type)
    sizes = []
    for name in names:
        dim_sizes = [(shape.get_size(name) if name in shape else 1) for shape in shapes]
        if min(dim_sizes) == max(dim_sizes):
            dim_sizes = dim_sizes[0]
        else:
            from ._functions import _stack
            from ._tensors import tensor
            dim_sizes = [tensor(d) for d in dim_sizes]
            dim_sizes = _stack(dim_sizes, stack_dim, stack_type)
        sizes.append(dim_sizes)
    return Shape(sizes, names, types).expand(len(shapes), stack_dim, stack_type)
