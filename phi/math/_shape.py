from __future__ import annotations

import warnings

from phi import math


BATCH_DIM = 'batch'
SPATIAL_DIM = 'spatial'
CHANNEL_DIM = 'channel'


class Shape:
    """
    Shapes enumerate dimensions, each consisting of a name, size and type.
    """

    def __init__(self, sizes: tuple or list, names: tuple or list, types: tuple or list):
        """
        Construct a Shape from sizes, names and types sequences.
        All arguments must have same length.

        To create a Shape with inferred dimension types, use :func:`shape(**dims)` instead.

        :param sizes: ordered dimension sizes
        :param names: ordered dimension names, either strings (spatial, batch) or integers (channel)
        :param types: ordered types, all values should be one of (CHANNEL_DIM, SPATIAL_DIM, BATCH_DIM)
        """
        assert len(sizes) == len(names) == len(types), "sizes=%s, names=%s, types=%s" % (sizes, names, types)
        self.sizes = tuple(sizes)
        self.names = tuple(names)
        assert all(isinstance(n, str) for n in names), names
        self.types = tuple(types)

    @property
    def named_sizes(self):
        return zip(self.names, self.sizes)

    @property
    def dimensions(self):
        return zip(self.sizes, self.names, self.types)

    def __len__(self):
        return len(self.sizes)

    def __contains__(self, item):
        return item in self.names

    def index(self, name):
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

    def get_size(self, name):
        if isinstance(name, str):
            return self.sizes[self.names.index(name)]
        elif isinstance(name, (tuple, list)):
            return tuple(self.get_size(n) for n in name)
        else:
            raise ValueError(name)

    def __getattr__(self, name):
        if name == 'names':
            raise AssertionError("Attribute missing: %s" % name)
        if name in self.names:
            return self.get_size(name)
        raise AttributeError("Shape has no attribute '%s'" % (name,))

    def get_type(self, name):
        if isinstance(name, str):
            return self.types[self.names.index(name)]
        elif isinstance(name, (tuple, list)):
            return tuple(self.get_type(n) for n in name)
        else:
            raise ValueError(name)

    def __getitem__(self, selection):
        if isinstance(selection, int):
            return self.sizes[selection]
        return Shape([self.sizes[i] for i in selection], [self.names[i] for i in selection], [self.types[i] for i in selection])

    @property
    def batch(self):
        return self[[i for i, t in enumerate(self.types) if t == BATCH_DIM]]

    @property
    def non_batch(self):
        return self[[i for i, t in enumerate(self.types) if t != BATCH_DIM]]

    @property
    def spatial(self) -> Shape:
        return self[[i for i, t in enumerate(self.types) if t == SPATIAL_DIM]]

    @property
    def non_spatial(self):
        return self[[i for i, t in enumerate(self.types) if t != SPATIAL_DIM]]

    @property
    def channel(self):
        return self[[i for i, t in enumerate(self.types) if t == CHANNEL_DIM]]

    @property
    def non_channel(self):
        return self[[i for i, t in enumerate(self.types) if t != CHANNEL_DIM]]

    @property
    def singleton(self):
        return self[[i for i, size in enumerate(self.sizes) if size == 1]]

    @property
    def non_singleton(self):
        return self[[i for i, size in enumerate(self.sizes) if size != 1]]

    @property
    def zero(self):
        return self[[i for i, size in enumerate(self.sizes) if size == 0]]

    @property
    def non_zero(self):
        return self[[i for i, size in enumerate(self.sizes) if size != 0]]

    @property
    def undefined(self):
        return self[[i for i, size in enumerate(self.sizes) if size is None]]

    @property
    def defined(self):
        return self[[i for i, size in enumerate(self.sizes) if size is not None]]

    def unstack(self):
        return tuple(Shape([self.sizes[i]], [self.names[i]], [self.types[i]]) for i in range(self.rank))

    @property
    def name(self):
        assert self.rank == 1, 'Shape.name is only defined for shapes of rank 1.'
        return self.names[0]

    @property
    def is_batch(self):
        return all([t == BATCH_DIM for t in self.types])

    @property
    def is_spatial(self):
        return all([t == SPATIAL_DIM for t in self.types])

    @property
    def is_channel(self):
        return all([t == CHANNEL_DIM for t in self.types])

    def mask(self, names: tuple or list or set):
        """
        Returns a binary sequence corresponding to the names of this Shape.
        A value of 1 means that a dimension of this Shape is contained in `names`.

        :param names: collection of dimension
        :return: binary sequence
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

    def reorder(self, names):
        assert len(names) == self.rank
        order = [self.index(n) for n in names]
        return self[order]

    def order_group(self, names):
        order = []
        for name in self.names:
            if name not in order:
                if name in names:
                    order.extend(names)
                else:
                    order.append(name)
        return order

    def combined(self, other, combine_spatial=False):
        """
        Returns a Shape object that both `self` and `other` can be broadcast to.
        If `self` and `other` are incompatible, raises a ValueError.
        :param other: Shape
        :return:
        :raise: ValueError if shapes don't match
        """
        return combine_safe(self, other, check_exact=[] if combine_spatial else [SPATIAL_DIM])

    def __and__(self, other):
        return combine_safe(self, other, check_exact=[SPATIAL_DIM])

    def expand_batch(self, size, name, pos=None):
        return self.expand(size, name, BATCH_DIM, pos)

    def expand_spatial(self, size, name, pos=None):
        return self.expand(size, name, SPATIAL_DIM, pos)

    def expand_channel(self, size, name, pos=None):
        return self.expand(size, name, CHANNEL_DIM, pos)

    def expand(self, size, name, dim_type, pos=None):
        """
        Add a dimension to the shape.

        The resulting shape has linear indices.
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

    def extend(self, other: Shape):
        return Shape(self.sizes + other.sizes, self.names + other.names, self.types + other.types)

    def without(self, dims) -> Shape:
        """
        Builds a new shape from this one that is missing all given dimensions.
        Dimensions in `dims` that are not part of this Shape are ignored.

        The complementary operation is :func:`Shape.only`.

        :param dims: single dimension (str) or collection of dimensions (tuple, list, Shape)
        :return: Shape without specified dimensions
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

    def only(self, dims):
        """
        Builds a new shape from this one that only contains the given dimensions.
        Dimensions in `dims` that are not part of this Shape are ignored.

        The complementary operation is :func:`Shape.without`.

        :param dims: single dimension (str) or collection of dimensions (tuple, list, Shape)
        :return: Shape containing only specified dimensions
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
    def rank(self):
        return len(self.sizes)

    @property
    def spatial_rank(self):
        """
        Fast implementation of Shape.spatial.rank.
        """
        r = 0
        for ty in self.types:
            if ty == SPATIAL_DIM:
                r += 1
        return r

    @property
    def batch_rank(self):
        """
        Fast implementation of Shape.batch.rank.
        """
        r = 0
        for ty in self.types:
            if ty == BATCH_DIM:
                r += 1
        return r

    @property
    def channel_rank(self):
        """
        Fast implementation of Shape.batch.rank.
        """
        r = 0
        for ty in self.types:
            if ty == CHANNEL_DIM:
                r += 1
        return r

    def to_batch(self, dims: tuple or list or None = None) -> Shape:
        """
        Returns a shape like this Shape but with `dims` being of type `batch`.

        Leaves this Shape object untouched.

        :param dims: sequence of dimension names to convert or None to convert all dimensions
        :return: new Shape object
        """
        if dims is None:
            return Shape(self.sizes, self.names, [BATCH_DIM] * self.rank)
        else:
            return Shape(self.sizes, self.names, [BATCH_DIM if dim in dims else self.types[i] for i, dim in enumerate(self.names)])

    @property
    def well_defined(self):
        return None not in self.sizes

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
    def volume(self):
        """
        Returns the total number of values contained in a tensor of this shape.
        This is the product of all dimension sizes.
        """
        if None in self.sizes:
            return None
        if self.rank == 0:
            return 1
        return math.prod(self.sizes)

    @property
    def is_empty(self):
        return len(self.sizes) == 0

    def order(self, sequence, default=None):
        """
        If sequence is a dict with dimension names as keys, orders its values according to this shape.

        Otherwise, the sequence is returned unchanged.

        :param sequence: sequence or dict to be ordered
        :type sequence: dict or list or tuple
        :param default: default value used for dimensions not contained in sequence
        :return: ordered sequence of values
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
        """
        Builds a sequence containing all multi-indices within a tensor of this shape.
        """
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


def parse_dim_names(obj, count: int) -> tuple:
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


def shape(**dims: int) -> Shape:
    """
    Creates a Shape from the dimension names and their respective sizes.

    Dimension types are inferred from the names according to the following rules:

    * single letter -> spatial dimension
    * starts with 'vector' -> channel dimension
    * else -> batch dimension

    :param dims: names -> size
    :return: Shape
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


def infer_shape(shape, dim_names=None, batch_dims=None, spatial_dims=None, channel_dims=None):
    if isinstance(shape, Shape):
        return shape
    shape = tuple(shape)
    if len(shape) == 0:
        return EMPTY_SHAPE
    # --- Infer dim types ---
    dims = _infer_dim_group_counts(len(shape), constraints=[batch_dims, spatial_dims, channel_dims])
    if dims is None:  # could not infer
        channel_dims = 1
        dims = _infer_dim_group_counts(len(shape), constraints=[batch_dims, spatial_dims, channel_dims])
        if dims is None:
            batch_dims = 1
            dims = _infer_dim_group_counts(len(shape), constraints=[batch_dims, spatial_dims, channel_dims])
    assert dims is not None, "Could not infer shape from '%s' given constraints batch_dims=%s, spatial_dims=%s, channel_dims=%s" % (shape, batch_dims, spatial_dims, channel_dims)
    batch_dims, spatial_dims, channel_dims = dims
    # --- Construct shape ---
    from phi import geom
    if dim_names is not None:
        dim_names = parse_dim_names(dim_names, len(shape))
    if dim_names is None or None in dim_names:
        set_dim_names = dim_names
        dim_names = []
        # --- batch names ---
        if batch_dims == 1:
            dim_names.append('batch')
        else:
            for i in range(batch_dims):
                dim_names.append('batch %d' % (i,))
        # --- spatial names ---
        for i in range(spatial_dims):
            dim_names.append(math.GLOBAL_AXIS_ORDER.axis_name(i, spatial_dims))
        # --- channel names ---
        if channel_dims == 0:
            pass
        elif channel_dims == 1:
            dim_names.append('vector')
        else:
            for i in range(channel_dims):
                dim_names.append('vector%d' % i)
        if set_dim_names is not None:
            for i, set_name in enumerate(set_dim_names):
                if set_name is not None:
                    dim_names[i] = set_name
    types = [_infer_dim_type_from_name(name) for name in dim_names]
    return Shape(sizes=shape, names=dim_names, types=types)


def _infer_dim_group_counts(rank, constraints: list):
    known_sum = sum([dim or 0 for dim in constraints])
    unknown_count = sum([1 if dim is None else 0 for dim in constraints])
    if known_sum == rank:
        return [dim or 0 for dim in constraints]
    if unknown_count == 1:
        return [rank - known_sum if dim is None else dim for dim in constraints]
    return None


def spatial_shape(sizes, names=None):
    """
    If `sizes` is a `Shape`, returns the spatial part of it.

    Otherwise, creates a Shape with the given sizes as spatial dimensions.
    The sizes are assumed to be ordered according to the GLOBAL_AXIS_ORDER and the dimensions are named accordingly.

    :param sizes: list of integers or Shape
    :return: Shape containing only spatial dimensions
    """
    if isinstance(sizes, Shape):
        return sizes.spatial.reorder(names) if names else sizes.spatial
    elif isinstance(sizes, dict):
        return Shape(sizes.values(), sizes.keys(), (SPATIAL_DIM,) * len(sizes))
    else:
        return infer_shape(sizes, batch_dims=0, channel_dims=0, dim_names=names)


def channel_shape(sizes: Shape or list or tuple or dict) -> Shape:
    """
    Creates a `Shape` with all dimensions of type `channel`.

    The behavior depends on the type of `sizes`:

    * Shape: Returns a new Shape containing only the channel dimensions.
    * dict str -> int: Keys are interpreted as names and values as sizes.
    * tuple/list of int: Names are generated automatically.

    :param sizes: Shape, dict: name -> size, sequence of int
    :return: new Shape
    """
    if isinstance(sizes, Shape):
        return sizes.channel
    elif isinstance(sizes, dict):
        return Shape(sizes.values(), sizes.keys(), [CHANNEL_DIM] * len(sizes))
    else:
        return infer_shape(sizes, batch_dims=0, spatial_dims=0)


def batch_shape(obj):
    if obj is None:
        return EMPTY_SHAPE
    elif isinstance(obj, Shape):
        return obj.batch
    elif isinstance(obj, int):
        return Shape((obj,), ('batch',), (BATCH_DIM,))
    else:
        return infer_shape(obj, spatial_dims=0, channel_dims=0)


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


def combine_stack(dim: Shape, *shapes: Shape):
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
                    sizes[names.index(name)] = None
    return dim & Shape(sizes, names, types)

