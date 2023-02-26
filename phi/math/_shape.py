import re
import warnings
from numbers import Number
from typing import Tuple, Callable, List, Union, Any

from phi import math


BATCH_DIM = 'batch'
SPATIAL_DIM = 'spatial'
CHANNEL_DIM = 'channel'
INSTANCE_DIM = 'înstance'
DUAL_DIM = 'dual'

TYPE_ABBR = {SPATIAL_DIM: "ˢ", CHANNEL_DIM: "ᶜ", INSTANCE_DIM: "ⁱ", BATCH_DIM: "ᵇ", DUAL_DIM: "ᵈ", None: "⁻"}  # ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ

DEBUG_CHECKS = False


def enable_debug_checks():
    """
    Once called, additional type checks are enabled.
    This may result in a noticeable drop in performance.
    """
    global DEBUG_CHECKS
    DEBUG_CHECKS = True


class Shape:
    """
    Shapes enumerate dimensions, each consisting of a name, size and type.

    There are five types of dimensions: `batch`, `dual`, `spatial`, `channel`, and `instance`.
    """

    def __init__(self, sizes: tuple, names: tuple, types: tuple, item_names: tuple):
        """
        To construct a `Shape`, use `batch`, `dual`, `spatial`, `channel` or `instance`, depending on the desired dimension type.
        To create a shape with multiple types, use `merge_shapes()`, `concat_shapes()` or the syntax `shape1 & shape2`.

        The `__init__` constructor is for internal use only.
        """
        if len(sizes) > 0 and any(s is not None and not isinstance(s, int) for s in sizes):
            from ._tensors import Tensor
            sizes = tuple([s if isinstance(s, Tensor) or s is None else int(s) for s in sizes])  # TODO replace this by an assert
        self.sizes: tuple = sizes
        """
        Ordered dimension sizes as `tuple`.
        The size of a dimension can be an `int` or a `Tensor` for [non-uniform shapes](https://tum-pbs.github.io/PhiFlow/Math.html#non-uniform-tensors).
        
        See Also:
            `Shape.get_size()`, `Shape.size`, `Shape.shape`.
        """
        self.names: Tuple[str] = names
        """
        Ordered dimension names as `tuple[str]`.
        
        See Also:
            `Shape.name`.
        """
        self.types: Tuple[str] = types  # undocumented, may be private
        self.item_names: Tuple[str or 'Shape'] = (None,) * len(sizes) if item_names is None else item_names  # undocumented
        if DEBUG_CHECKS:
            assert len(sizes) == len(names) == len(types) == len(item_names), f"sizes={sizes}, names={names}, types={types}, item_names={item_names}"
            assert all(isinstance(n, str) for n in names), f"All names must be of type string but got {names}"
            assert isinstance(self.item_names, tuple)
            assert all([items is None or isinstance(items, tuple) for items in self.item_names])
            assert all([items is None or all([isinstance(n, str) for n in items]) for items in self.item_names])
            from ._tensors import Tensor
            for name, size in zip(names, sizes):
                if size is not None and isinstance(size, Tensor):
                    assert size.rank > 0
                    # for dim in size.shape.names:
                    #     assert dim in self.names, f"Dimension {name} varies along {dim} but {dim} is not part of the Shape {self}"

    def _to_dict(self, include_sizes=True):
        result = dict(names=self.names, types=self.types, item_names=self.item_names)
        if include_sizes:
            if not all([isinstance(s, int)] for s in self.sizes):
                raise NotImplementedError()
            result['sizes'] = self.sizes
        return result

    @staticmethod
    def _from_dict(dict_: dict):
        names = tuple(dict_['names'])
        sizes = tuple(dict_['sizes']) if 'sizes' in dict_ else (None,) * len(names)
        item_names = tuple([None if n is None else tuple(n) for n in dict_['item_names']])
        return Shape(sizes, names, tuple(dict_['types']), item_names)

    @property
    def _named_sizes(self):
        return zip(self.names, self.sizes)

    @property
    def _dimensions(self):
        return zip(self.sizes, self.names, self.types, self.item_names)

    @property
    def untyped_dict(self):
        """
        Returns:
            `dict` containing dimension names as keys.
                The values are either the item names as `tuple` if available, otherwise the size.
        """
        return {name: self.get_item_names(i) or self.get_size(i) for i, name in enumerate(self.names)}

    def __len__(self):
        return len(self.sizes)

    def __contains__(self, item):
        if isinstance(item, (str, tuple, list)):
            dims = parse_dim_order(item)
            return all(dim in self.names for dim in dims)
        elif isinstance(item, Shape):
            return all([d in self.names for d in item.names])
        else:
            raise ValueError(item)

    def isdisjoint(self, other: 'Shape' or tuple or list or str):
        """ Shapes are disjoint if all dimension names of one shape do not occur in the other shape. """
        other = parse_dim_order(other)
        return not any(dim in self.names for dim in other)

    def __iter__(self):
        return iter(self[i] for i in range(self.rank))

    def index(self, dim: str or 'Shape' or None) -> int:
        """
        Finds the index of the dimension within this `Shape`.

        See Also:
            `Shape.indices()`.

        Args:
            dim: Dimension name or single-dimension `Shape`.

        Returns:
            Index as `int`.
        """
        if dim is None:
            return None
        elif isinstance(dim, str):
            if dim not in self.names:
                raise ValueError(f"Shape {self} has no dimension '{dim}'")
            return self.names.index(dim)
        elif isinstance(dim, Shape):
            assert dim.rank == 1, f"index() requires a single dimension as input but got {dim}. Use indices() for multiple dimensions."
            return self.names.index(dim.name)
        else:
            raise ValueError(f"index() requires a single dimension as input but got {dim}")

    def indices(self, dims: tuple or list or 'Shape') -> Tuple[int]:
        """
        Finds the indices of the given dimensions within this `Shape`.

        See Also:
            `Shape.index()`.

        Args:
            dims: Sequence of dimensions as `tuple`, `list` or `Shape`.

        Returns:
            Indices as `tuple[int]`.
        """
        if isinstance(dims, (list, tuple, set)):
            return tuple([self.index(n) for n in dims])
        elif isinstance(dims, Shape):
            return tuple([self.index(n) for n in dims.names])
        else:
            raise ValueError(f"indices() requires a sequence of dimensions but got {dims}")

    def get_size(self, dim: str or 'Shape' or int, default=None):
        """
        See Also:
            `Shape.get_sizes()`, `Shape.size`

        Args:
            dim: Dimension, either as name `str` or single-dimension `Shape` or index `int`.
            default: (Optional) If the dim does not exist, return this value instead of raising an error.

        Returns:
            Size associated with `dim` as `int` or `Tensor`.
        """
        if isinstance(dim, int):
            assert default is None, "Cannot use a default value when passing an int for dim"
            return self.sizes[dim]
        if isinstance(dim, Shape):
            assert dim.rank == 1, f"get_size() requires a single dimension but got {dim}. Use indices() to get multiple sizes."
            dim = dim.name
        if isinstance(dim, str):
            if dim not in self.names:
                if default is None:
                    raise KeyError(f"get_size() failed because '{dim}' is not part of Shape {self} and no default value was provided")
                else:
                    return default
            return self.sizes[self.names.index(dim)]
        else:
            raise ValueError(f"get_size() requires a single dimension but got {dim}. Use indices() to get multiple sizes.")

    def get_sizes(self, dims: tuple or list or 'Shape') -> tuple:
        """
        See Also:
            `Shape.get_size()`

        Args:
            dims: Dimensions as `tuple`, `list` or `Shape`.

        Returns:
            `tuple`
        """
        assert isinstance(dims, (tuple, list, Shape)), f"get_sizes() requires a sequence of dimensions but got {dims}"
        return tuple([self.get_size(dim) for dim in dims])

    def get_type(self, dim: str or 'Shape') -> str:
        # undocumented, use get_dim_type() instead.
        if isinstance(dim, str):
            return self.types[self.names.index(dim)]
        elif isinstance(dim, Shape):
            assert dim.rank == 1, f"Shape.get_type() only accepts single-dimension Shapes but got {dim}"
            return self.types[self.names.index(dim.name)]
        else:
            raise ValueError(dim)

    def get_dim_type(self, dim: str or 'Shape') -> Callable:
        """
        Args:
            dim: Dimension, either as name `str` or single-dimension `Shape`.

        Returns:
            Dimension type, one of `batch`, `spatial`, `instance`, `channel`.
        """
        return {BATCH_DIM: batch, SPATIAL_DIM: spatial, INSTANCE_DIM: instance, CHANNEL_DIM: channel}[self.get_type(dim)]

    def get_types(self, dims: tuple or list or 'Shape') -> tuple:
        # undocumented, do not use
        if isinstance(dims, (tuple, list)):
            return tuple(self.get_type(n) for n in dims)
        elif isinstance(dims, Shape):
            return tuple(self.get_type(n) for n in dims.names)
        else:
            raise ValueError(dims)

    def get_item_names(self, dim: str or 'Shape' or int, fallback_spatial=False) -> tuple or None:
        """
        Args:
            fallback_spatial: If `True` and no item names are defined for `dim` and `dim` is a channel dimension, the spatial dimension names are interpreted as item names along `dim` in the order they are listed in this `Shape`.
            dim: Dimension, either as `int` index, `str` name or single-dimension `Shape`.

        Returns:
            Item names as `tuple` or `None` if not defined.
        """
        if isinstance(dim, int):
            result = self.item_names[dim]
        elif isinstance(dim, str):
            result = self.item_names[self.index(dim)]
        elif isinstance(dim, Shape):
            assert dim.rank == 1, f"Shape.get_type() only accepts single-dimension Shapes but got {dim}"
            result = self.item_names[self.names.index(dim.name)]
        else:
            raise ValueError(dim)
        if result is not None:
            return result
        elif fallback_spatial and self.spatial_rank == self.get_size(dim) and self.get_type(dim) == CHANNEL_DIM:
            return self.spatial.names
        else:
            return None

    def flipped(self, dims: List[str] or Tuple[str]):
        item_names = list(self.item_names)
        for dim in dims:
            if dim in self.names:
                dim_i_n = self.get_item_names(dim)
                if dim_i_n is not None:
                    item_names[self.index(dim)] = tuple(reversed(dim_i_n))
        return Shape(self.sizes, self.names, self.types, tuple(item_names))

    def __getitem__(self, selection):
        if isinstance(selection, int):
            return Shape((self.sizes[selection],), (self.names[selection],), (self.types[selection],), (self.item_names[selection],))
        elif isinstance(selection, slice):
            return Shape(self.sizes[selection], self.names[selection], self.types[selection], self.item_names[selection])
        elif isinstance(selection, str):
            if ',' in selection:
                selection = [self.index(s.strip()) for s in selection.split(',')]
            else:
                selection = self.index(selection)
            return self[selection]
        elif isinstance(selection, (tuple, list)):
            selection = [self.index(s) if isinstance(s, str) else s for s in selection]
            return Shape(tuple([self.sizes[i] for i in selection]), tuple([self.names[i] for i in selection]), tuple([self.types[i] for i in selection]), tuple([self.item_names[i] for i in selection]))
        raise AssertionError("Can only access shape elements as shape[int] or shape[slice]")

    @property
    def reversed(self):
        return Shape(tuple(reversed(self.sizes)), tuple(reversed(self.names)), tuple(reversed(self.types)), tuple(reversed(self.item_names)))

    @property
    def batch(self) -> 'Shape':
        """
        Filters this shape, returning only the batch dimensions as a new `Shape` object.

        See also:
            `Shape.batch`, `Shape.spatial`, `Shape.instance`, `Shape.channel`, `Shape.dual`, `Shape.non_batch`, `Shape.non_spatial`, `Shape.non_instance`, `Shape.non_channel`, `Shape.non_dual`.

        Returns:
            New `Shape` object
        """
        return self[[i for i, t in enumerate(self.types) if t == BATCH_DIM]]

    @property
    def non_batch(self) -> 'Shape':
        """
        Filters this shape, returning only the non-batch dimensions as a new `Shape` object.

        See also:
            `Shape.batch`, `Shape.spatial`, `Shape.instance`, `Shape.channel`, `Shape.dual`, `Shape.non_batch`, `Shape.non_spatial`, `Shape.non_instance`, `Shape.non_channel`, `Shape.non_dual`.

        Returns:
            New `Shape` object
        """
        return self[[i for i, t in enumerate(self.types) if t != BATCH_DIM]]

    @property
    def spatial(self) -> 'Shape':
        """
        Filters this shape, returning only the spatial dimensions as a new `Shape` object.

        See also:
            `Shape.batch`, `Shape.spatial`, `Shape.instance`, `Shape.channel`, `Shape.dual`, `Shape.non_batch`, `Shape.non_spatial`, `Shape.non_instance`, `Shape.non_channel`, `Shape.non_dual`.

        Returns:
            New `Shape` object
        """
        return self[[i for i, t in enumerate(self.types) if t == SPATIAL_DIM]]

    @property
    def non_spatial(self) -> 'Shape':
        """
        Filters this shape, returning only the non-spatial dimensions as a new `Shape` object.

        See also:
            `Shape.batch`, `Shape.spatial`, `Shape.instance`, `Shape.channel`, `Shape.dual`, `Shape.non_batch`, `Shape.non_spatial`, `Shape.non_instance`, `Shape.non_channel`, `Shape.non_dual`.

        Returns:
            New `Shape` object
        """
        return self[[i for i, t in enumerate(self.types) if t != SPATIAL_DIM]]

    @property
    def instance(self) -> 'Shape':
        """
        Filters this shape, returning only the instance dimensions as a new `Shape` object.

        See also:
            `Shape.batch`, `Shape.spatial`, `Shape.instance`, `Shape.channel`, `Shape.dual`, `Shape.non_batch`, `Shape.non_spatial`, `Shape.non_instance`, `Shape.non_channel`, `Shape.non_dual`.

        Returns:
            New `Shape` object
        """
        return self[[i for i, t in enumerate(self.types) if t == INSTANCE_DIM]]

    @property
    def non_instance(self) -> 'Shape':
        """
        Filters this shape, returning only the non-instance dimensions as a new `Shape` object.

        See also:
            `Shape.batch`, `Shape.spatial`, `Shape.instance`, `Shape.channel`, `Shape.dual`, `Shape.non_batch`, `Shape.non_spatial`, `Shape.non_instance`, `Shape.non_channel`, `Shape.non_dual`.

        Returns:
            New `Shape` object
        """
        return self[[i for i, t in enumerate(self.types) if t != INSTANCE_DIM]]

    @property
    def channel(self) -> 'Shape':
        """
        Filters this shape, returning only the channel dimensions as a new `Shape` object.

        See also:
            `Shape.batch`, `Shape.spatial`, `Shape.instance`, `Shape.channel`, `Shape.dual`, `Shape.non_batch`, `Shape.non_spatial`, `Shape.non_instance`, `Shape.non_channel`, `Shape.non_dual`.

        Returns:
            New `Shape` object
        """
        return self[[i for i, t in enumerate(self.types) if t == CHANNEL_DIM]]

    @property
    def non_channel(self) -> 'Shape':
        """
        Filters this shape, returning only the non-channel dimensions as a new `Shape` object.

        See also:
            `Shape.batch`, `Shape.spatial`, `Shape.instance`, `Shape.channel`, `Shape.dual`, `Shape.non_batch`, `Shape.non_spatial`, `Shape.non_instance`, `Shape.non_channel`, `Shape.non_dual`.

        Returns:
            New `Shape` object
        """
        return self[[i for i, t in enumerate(self.types) if t != CHANNEL_DIM]]

    @property
    def dual(self) -> 'Shape':
        """
        Filters this shape, returning only the dual dimensions as a new `Shape` object.

        See also:
            `Shape.batch`, `Shape.spatial`, `Shape.instance`, `Shape.channel`, `Shape.dual`, `Shape.non_batch`, `Shape.non_spatial`, `Shape.non_instance`, `Shape.non_channel`, `Shape.non_dual`.

        Returns:
            New `Shape` object
        """
        return self[[i for i, t in enumerate(self.types) if t == DUAL_DIM]]

    @property
    def non_dual(self) -> 'Shape':
        """
        Filters this shape, returning only the non-dual dimensions as a new `Shape` object.

        See also:
            `Shape.batch`, `Shape.spatial`, `Shape.instance`, `Shape.channel`, `Shape.dual`, `Shape.non_batch`, `Shape.non_spatial`, `Shape.non_instance`, `Shape.non_channel`, `Shape.non_dual`.

        Returns:
            New `Shape` object
        """
        return self[[i for i, t in enumerate(self.types) if t != DUAL_DIM]]

    @property
    def non_singleton(self) -> 'Shape':
        """
        Filters this shape, returning only non-singleton dimensions as a new `Shape` object.
        Dimensions are singleton if their size is exactly `1`.

        Returns:
            New `Shape` object
        """
        return self[[i for i, s in enumerate(self.sizes) if not _size_equal(s, 1)]]

    def unstack(self, dim='dims') -> Tuple['Shape']:
        """
        Slices this `Shape` along a dimension.
        The dimension listing the sizes of the shape is referred to as `'dims'`.

        Non-uniform tensor shapes may be unstacked along other dimensions as well, see
        https://tum-pbs.github.io/PhiFlow/Math.html#non-uniform-tensors

        Args:
            dim: dimension to unstack

        Returns:
            slices of this shape
        """
        if dim == 'dims':
            return tuple(Shape((self.sizes[i],), (self.names[i],), (self.types[i],), (self.item_names[i],)) for i in range(self.rank))
        if dim not in self:
            return tuple([self])
        else:
            from ._tensors import Tensor
            inner = self.without(dim)
            sizes = []
            dim_size = self.get_size(dim)
            for size in inner.sizes:
                if isinstance(size, Tensor) and dim in size.shape:
                    sizes.append(size.unstack(dim))
                    dim_size = size.shape.get_size(dim)
                else:
                    sizes.append(size)
            assert isinstance(dim_size, int)
            shapes = tuple(Shape(tuple([int(size[i]) if isinstance(size, tuple) else size for size in sizes]), inner.names, inner.types, inner.item_names) for i in range(dim_size))
            return shapes

    @property
    def name(self) -> str:
        """
        Only for Shapes containing exactly one single dimension.
        Returns the name of the dimension.

        See Also:
            `Shape.names`.
        """
        assert self.rank == 1, f"Shape.name is only defined for shapes of rank 1. shape={self}"
        return self.names[0]

    @property
    def size(self) -> int:
        """
        Only for Shapes containing exactly one single dimension.
        Returns the size of the dimension.

        See Also:
            `Shape.sizes`, `Shape.get_size()`.
        """
        assert self.rank == 1, "Shape.size is only defined for shapes of rank 1."
        return self.sizes[0]

    @property
    def type(self) -> int:
        """
        Only for Shapes containing exactly one single dimension.
        Returns the type of the dimension.

        See Also:
            `Shape.get_type()`.
        """
        assert self.rank == 1, "Shape.type is only defined for shapes of rank 1."
        return self.types[0]

    def __int__(self):
        assert self.rank == 1, "int(Shape) is only defined for shapes of rank 1."
        return self.sizes[0]

    def mask(self, names: tuple or list or set or 'Shape'):
        """
        Returns a binary sequence corresponding to the names of this Shape.
        A value of 1 means that a dimension of this Shape is contained in `names`.

        Args:
          names: instance of dimension
          names: tuple or list or set: 

        Returns:
          binary sequence

        """
        if isinstance(names, str):
            names = [names]
        elif isinstance(names, Shape):
            names = names.names
        mask = [1 if name in names else 0 for name in self.names]
        return tuple(mask)

    def __repr__(self):
        def size_repr(size, items):
            if items is not None:
                if len(items) <= 4:
                    return ",".join(items)
                else:
                    return f"{size}:{items[0]}..{items[-1]}"
            else:
                return size

        strings = [f"{name}{TYPE_ABBR.get(dim_type, '?')}={size_repr(size, items)}" for size, name, dim_type, items in self._dimensions]
        return '(' + ', '.join(strings) + ')'

    def __eq__(self, other):
        if not isinstance(other, Shape):
            return False
        if self.names != other.names or self.types != other.types:
            return False
        for size1, size2 in zip(self.sizes, other.sizes):
            equal = size1 == size2
            assert isinstance(equal, (bool, math.Tensor))
            if isinstance(equal, math.Tensor):
                equal = equal.all
            if not equal:
                return False
        for names1, names2 in zip(self.item_names, other.item_names):
            if names1 != names2:
                return False
        return True

    def __ne__(self, other):
        return not self == other

    def __bool__(self):
        return self.rank > 0

    def _reorder(self, names: tuple or list or 'Shape') -> 'Shape':
        assert len(names) == self.rank
        if isinstance(names, Shape):
            names = names.names
        order = [self.index(n) for n in names]
        return self[order]

    def _order_group(self, names: tuple or list or 'Shape') -> list:
        """ Reorders the dimensions of this `Shape` so that `names` are clustered together and occur in the specified order. """
        if isinstance(names, Shape):
            names = names.names
        result = []
        for dim in self.names:
            if dim not in result:
                if dim in names:
                    result.extend(names)
                else:
                    result.append(dim)
        return result

    def __and__(self, other):
        return merge_shapes(self, other)

    def _expand(self, dim: 'Shape', pos=None) -> 'Shape':
        """**Deprecated.** Use `phi.math.merge_shapes()` or `phi.math.concat_shapes()` instead. """
        warnings.warn("Shape.expand() is deprecated. Use merge_shapes() or concat_shapes() instead.", DeprecationWarning)
        if not dim:
            return self
        assert dim.name not in self, f"Cannot expand shape {self} by {dim} because dimension already exists."
        assert isinstance(dim, Shape) and dim.rank == 1, f"Shape.expand() requires a single dimension as a Shape but got {dim}"
        if pos is None:
            same_type_dims = self[[i for i, t in enumerate(self.types) if t == dim.type]]
            if len(same_type_dims) > 0:
                pos = self.index(same_type_dims.names[0])
            else:
                pos = {BATCH_DIM: 0, INSTANCE_DIM: self.batch_rank, SPATIAL_DIM: self.batch.rank + self.instance_rank, CHANNEL_DIM: self.rank + 1}[dim.type]
        elif pos < 0:
            pos += self.rank + 1
        sizes = list(self.sizes)
        names = list(self.names)
        types = list(self.types)
        item_names = list(self.item_names)
        sizes.insert(pos, dim.size)
        names.insert(pos, dim.name)
        types.insert(pos, dim.type)
        item_names.insert(pos, dim.item_names[0])
        return Shape(tuple(sizes), tuple(names), tuple(types), tuple(item_names))

    def without(self, dims: 'DimFilter') -> 'Shape':
        """
        Builds a new shape from this one that is missing all given dimensions.
        Dimensions in `dims` that are not part of this Shape are ignored.
        
        The complementary operation is `Shape.only()`.

        Args:
          dims: Single dimension (str) or instance of dimensions (tuple, list, Shape)
          dims: Dimensions to exclude as `str` or `tuple` or `list` or `Shape`. Dimensions that are not included in this shape are ignored.

        Returns:
          Shape without specified dimensions
        """
        if callable(dims):
            dims = dims(self)
        if isinstance(dims, str):
            dims = parse_dim_order(dims)
        if isinstance(dims, (tuple, list, set)):
            return self[[i for i in range(self.rank) if self.names[i] not in dims]]
        elif isinstance(dims, Shape):
            return self[[i for i in range(self.rank) if self.names[i] not in dims.names]]
        elif dims is None:  # subtract none
            return self
        else:
            raise ValueError(dims)

    def only(self, dims: 'DimFilter', reorder=False):
        """
        Builds a new shape from this one that only contains the given dimensions.
        Dimensions in `dims` that are not part of this Shape are ignored.
        
        The complementary operation is :func:`Shape.without`.

        Args:
          dims: comma-separated dimension names (str) or instance of dimensions (tuple, list, Shape) or filter function.
          reorder: If `False`, keeps the dimension order as defined in this shape.
            If `True`, reorders the dimensions of this shape to match the order of `dims`.

        Returns:
          Shape containing only specified dimensions

        """
        if dims is None:  # keep none
            return EMPTY_SHAPE
        if callable(dims):
            dims = dims(self)
        if isinstance(dims, str):
            dims = parse_dim_order(dims)
        if isinstance(dims, Shape):
            dims = dims.names
        if not isinstance(dims, (tuple, list, set)):
            raise ValueError(dims)
        if reorder:
            return self[[self.names.index(d) for d in dims if d in self.names]]
        else:
            return self[[i for i in range(self.rank) if self.names[i] in dims]]

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
        return sum([1 for ty in self.types if ty == BATCH_DIM])

    @property
    def instance_rank(self) -> int:
        return sum([1 for ty in self.types if ty == INSTANCE_DIM])

    @property
    def spatial_rank(self) -> int:
        """ Number of spatial dimensions """
        return sum([1 for ty in self.types if ty == SPATIAL_DIM])

    @property
    def dual_rank(self) -> int:
        """ Number of spatial dimensions """
        return sum([1 for ty in self.types if ty == DUAL_DIM])

    @property
    def channel_rank(self) -> int:
        """ Number of channel dimensions """
        return sum([1 for ty in self.types if ty == CHANNEL_DIM])

    @property
    def well_defined(self):
        """
        Returns `True` if no dimension size is `None`.

        Shapes with undefined sizes may be used in `phi.math.tensor()`, `phi.math.wrap()`, `phi.math.stack()` or `phi.math.concat()`.

        To create an undefined size, call a constructor function (`batch()`, `spatial()`, `channel()`, `instance()`)
        with positional `str` arguments, e.g. `spatial('x')`.
        """
        for size in self.sizes:
            if size is None:
                return False
        return True

    @property
    def shape(self) -> 'Shape':
        """
        Higher-order `Shape`.
        The returned shape will always contain the channel dimension `dims` with a size equal to the `Shape.rank` of this shape.

        For uniform shapes, `Shape.shape` will only contain the dimension `dims` but the shapes of [non-uniform shapes](https://tum-pbs.github.io/PhiFlow/Math.html#non-uniform-tensors)
        may contain additional dimensions.

        See Also:
            `Shape.is_uniform`.

        Returns:
            `Shape`.
        """
        from phi.math import Tensor
        shape = Shape((self.rank,), ('dims',), (CHANNEL_DIM,), (self.names,))
        for size in self.sizes:
            if isinstance(size, Tensor):
                shape = shape & size.shape
        return shape

    @property
    def is_uniform(self) -> bool:
        """
        A shape is uniform if it all sizes have a single integer value.

        See Also:
            `Shape.is_non_uniform`, `Shape.shape`.
        """
        return not self.is_non_uniform

    @property
    def is_non_uniform(self) -> bool:
        """
        A shape is non-uniform if the size of any dimension varies along another dimension.

        See Also:
            `Shape.is_uniform`, `Shape.shape`.
        """
        from phi.math import Tensor
        for size in self.sizes:
            if isinstance(size, Tensor) and size.rank > 0:
                return True
        return False

    @property
    def non_uniform(self) -> 'Shape':
        """
        Returns only the non-uniform dimensions of this shape, i.e. the dimensions whose size varies along another dimension.
        """
        from phi.math import Tensor
        indices = [i for i, size in enumerate(self.sizes) if isinstance(size, Tensor) and size.rank > 0]
        return self[indices]

    def with_size(self, size: int or None):
        """
        Only for single-dimension shapes.
        Returns a `Shape` representing this dimension but with a different size.

        See Also:
            `Shape.with_sizes()`.

        Args:
            size: Replacement size for this dimension.

        Returns:
            `Shape`
        """
        assert self.rank == 1, "Shape.with_size() is only defined for shapes of rank 1."
        return self.with_sizes([size])

    def with_sizes(self, sizes: tuple or list or 'Shape' or int, keep_item_names=True):
        """
        Returns a new `Shape` matching the dimension names and types of `self` but with different sizes.

        See Also:
            `Shape.with_size()`.

        Args:
            sizes: One of

                * `tuple` / `list` of same length as `self` containing replacement sizes.
                * `Shape` of any rank. Replaces sizes for dimensions shared by `sizes` and `self`.

            keep_item_names: If `False`, forgets all item names.
                If `True`, keeps item names where the size does not change.

        Returns:
            `Shape` with same names and types as `self`.
        """
        if isinstance(sizes, int):
            sizes = [sizes] * len(self.sizes)
        if isinstance(sizes, Shape):
            item_names = [sizes.get_item_names(dim) if dim in sizes else self.get_item_names(dim) for dim in self.names]
            sizes = [sizes.get_size(dim) if dim in sizes else s for dim, s in self._named_sizes]
            return Shape(tuple(sizes), self.names, self.types, tuple(item_names))
        else:
            assert len(sizes) == len(self.sizes), f"Cannot create shape from {self} with sizes {sizes}"
            sizes_ = []
            item_names = []
            for i, obj in enumerate(sizes):
                new_size, new_item_names = Shape._size_and_item_names_from_obj(obj, self.sizes[i], self.item_names[i], keep_item_names)
                sizes_.append(new_size)
                item_names.append(new_item_names)
            return Shape(tuple(sizes_), self.names, self.types, tuple(item_names))

    @staticmethod
    def _size_and_item_names_from_obj(obj, prev_size, prev_item_names, keep_item_names=True):
        if isinstance(obj, str):
            obj = [s.strip() for s in obj.split(',')]
        if isinstance(obj, (tuple, list)):
            return len(obj), tuple(obj)
        elif isinstance(obj, Number):
            return obj, prev_item_names if keep_item_names and (prev_size is None or _size_equal(obj, prev_size)) else None
        elif isinstance(obj, math.Tensor) or obj is None:
            return obj, None
        else:
            raise ValueError(f"sizes can only contain int, str or Tensor but got {type(obj)}")

    def without_sizes(self):
        """
        Returns:
            `Shape` with all sizes undefined (`None`)
        """
        return Shape((None,) * self.rank, self.names, self.types, (None,) * self.rank)

    def _replace_single_size(self, dim: str, size: int, keep_item_names: bool = False):
        new_sizes = list(self.sizes)
        new_sizes[self.index(dim)] = size
        return self.with_sizes(new_sizes, keep_item_names=keep_item_names)

    def with_dim_size(self, dim: str or 'Shape', size: int or 'math.Tensor' or str or tuple or list, keep_item_names=True):
        """
        Returns a new `Shape` that has a different size for `dim`.

        Args:
            dim: Dimension for which to replace the size, `Shape` or `str`.
            size: New size, `int` or `Tensor`

        Returns:
            `Shape` with same names and types as `self`.
        """
        if isinstance(dim, Shape):
            dim = dim.name
        assert isinstance(dim, str)
        new_size, new_item_names = Shape._size_and_item_names_from_obj(size, self.get_size(dim), self.get_item_names(dim), keep_item_names)
        return self.replace(dim, Shape((new_size,), (dim,), (self.get_type(dim),), (new_item_names,)))

    def _with_names(self, names: str or tuple or list):
        if isinstance(names, str):
            names = parse_dim_names(names, self.rank)
            names = [n if n is not None else o for n, o in zip(names, self.names)]
        return Shape(self.sizes, tuple(names), self.types, self.item_names)

    def _replace_names_and_types(self,
                                 dims: 'Shape' or str or tuple or list,
                                 new: 'Shape' or str or tuple or list) -> 'Shape':
        """
        Returns a copy of `self` with `dims` replaced by `new`.
        Dimensions that are not present in `self` are ignored.

        The dimension order is preserved.

        Args:
            dims: Dimensions to replace.
            new: New dimensions, must have same length as `dims`.
                If a `Shape` is given, replaces the dimension types and item names as well.

        Returns:
            `Shape` with same rank and dimension order as `self`.
        """
        dims = parse_dim_order(dims)
        sizes = [math.rename_dims(s, dims, new) if isinstance(s, math.Tensor) else s for s in self.sizes]
        new = parse_dim_order(new) if isinstance(new, str) else new
        names = list(self.names)
        types = list(self.types)
        item_names = list(self.item_names)
        for old_name, new_dim in zip(dims, new):
            if old_name in self:
                if isinstance(new_dim, Shape):
                    names[self.index(old_name)] = new_dim.name
                    types[self.index(old_name)] = new_dim.type
                    item_names[self.index(old_name)] = new_dim.item_names[0]
                else:
                    names[self.index(old_name)] = new_dim
        return Shape(tuple(sizes), tuple(names), tuple(types), tuple(item_names))

    def replace(self, dims: 'Shape' or str or tuple or list, new: 'Shape') -> 'Shape':
        """
        Returns a copy of `self` with `dims` replaced by `new`.
        Dimensions that are not present in `self` are ignored.

        The dimension order is preserved.

        Args:
            dims: Dimensions to replace.
            new: New dimensions, must have same length as `dims`.
                If a `Shape` is given, replaces the dimension types and item names as well.

        Returns:
            `Shape` with same rank and dimension order as `self`.
        """
        dims = parse_dim_order(dims)
        assert isinstance(new, Shape), f"new must be a Shape but got {new}"
        names = list(self.names)
        sizes = list(self.sizes)
        types = list(self.types)
        item_names = list(self.item_names)
        if len(new) > len(dims):  # Put all in one spot
            assert len(dims) == 1, "Cannot replace 2+ dims by more replacements"
            index = self.index(dims[0])
            return concat_shapes(self[:index], new, self[index+1:])
        for old_name, new_dim in zip(dims, new):
            if old_name in self:
                names[self.index(old_name)] = new_dim.name
                types[self.index(old_name)] = new_dim.type
                item_names[self.index(old_name)] = new_dim.item_names[0]
                sizes[self.index(old_name)] = new_dim.size
        replaced = Shape(tuple(sizes), tuple(names), tuple(types), tuple(item_names))
        if len(new) == len(dims):
            return replaced
        to_remove = dims[-(len(dims) - len(new)):]
        return replaced.without(to_remove)

    def _with_types(self, types: 'Shape' or str):
        """
        Only for internal use.
        Note: This method does not rename dimensions to comply with type requirements (e.g. ~ for dual dims).
        """
        if isinstance(types, Shape):
            return Shape(self.sizes, self.names, tuple([types.get_type(name) if name in types else self_type for name, self_type in zip(self.names, self.types)]), self.item_names)
        elif isinstance(types, str):
            return Shape(self.sizes, self.names, (types,) * self.rank, self.item_names)
        else:
            raise ValueError(types)

    def _with_item_names(self, item_names: tuple):
        return Shape(self.sizes, self.names, self.types, item_names)

    def _with_item_name(self, dim: str, item_name: tuple):
        if dim not in self:
            return self
        item_names = list(self.item_names)
        item_names[self.index(dim)] = item_name
        return Shape(self.sizes, self.names, self.types, tuple(item_names))

    def _perm(self, names: Tuple[str]):
        assert len(set(names)) == len(names), f"No duplicates allowed but got {names}"
        assert len(names) >= len(self.names), f"Cannot find permutation for {self} given {names} because names {set(self.names) - set(names)} are missing"
        assert len(names) <= len(self.names), f"Cannot find permutation for {self} given {names} because too many names were passed: {names}"
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
        from phi.math import Tensor
        for dim, size in self._named_sizes:
            if isinstance(size, Tensor) and size.rank > 0:
                non_uniform_dim = size.shape.names[0]
                shapes = self.unstack(non_uniform_dim)
                return sum(s.volume for s in shapes)
        result = 1
        for size in self.sizes:
            if size is None:
                return None
            result *= size
        return int(result)

    @property
    def is_empty(self) -> bool:
        """ True if this shape has no dimensions. Equivalent to `Shape.rank` `== 0`. """
        return len(self.sizes) == 0

    def after_pad(self, widths: dict) -> 'Shape':
        sizes = list(self.sizes)
        item_names = list(self.item_names)
        for dim, (lo, up) in widths.items():
            sizes[self.index(dim)] += lo + up
            item_names[self.index(dim)] = None
        return Shape(tuple(sizes), self.names, self.types, tuple(item_names))

    def prepare_gather(self, dim: str, selection):
        if isinstance(selection, Shape):
            selection = selection.name if selection.rank == 1 else selection.names
        if isinstance(selection, str) and ',' in selection:
            selection = parse_dim_order(selection)
        if isinstance(selection, str):  # single item name
            item_names = self.get_item_names(dim, fallback_spatial=True)
            assert item_names is not None, f"No item names defined for dim '{dim}' in tensor {self.shape} and dimension size does not match spatial rank."
            assert selection in item_names, f"Accessing tensor.{dim}['{selection}'] failed. Item names are {item_names}."
            selection = item_names.index(selection)
        if isinstance(selection, (tuple, list)):
            selection = list(selection)
            if any([isinstance(s, str) for s in selection]):
                item_names = self.get_item_names(dim, fallback_spatial=True)
                for i, s in enumerate(selection):
                    if isinstance(s, str):
                        assert item_names is not None, f"Accessing tensor.{dim}['{s}'] failed because no item names are present on tensor {self.shape}"
                        assert s in item_names, f"Accessing tensor.{dim}['{s}'] failed. Item names are {item_names}."
                        selection[i] = item_names.index(s)
            if not selection:  # empty
                selection = slice(0, 0)
        return selection

    def after_gather(self, selection: dict) -> 'Shape':
        result = self
        for sel_dim, selection in selection.items():
            if sel_dim not in self.names:
                continue
            selection = self.prepare_gather(sel_dim, selection)
            if isinstance(selection, int):
                if result.is_uniform:
                    result = result.without(sel_dim)
                else:
                    from phi.math import Tensor
                    gathered_sizes = [(s[{sel_dim: selection}] if isinstance(s, Tensor) else s) for s in result.sizes]
                    gathered_sizes = [(int(s) if isinstance(s, Tensor) and s.rank == 0 else s) for s in gathered_sizes]
                    result = result.with_sizes(gathered_sizes, keep_item_names=True).without(sel_dim)
            elif isinstance(selection, slice):
                assert isinstance(selection.step, int) or selection.step is None, f"slice step must be an int or None but got {type(selection.step).__name__}"
                assert isinstance(selection.start, int) or selection.start is None, f"slice start must be an int or None but got {type(selection.start).__name__}"
                assert isinstance(selection.stop, int) or selection.stop is None, f"slice stop must be an int or None but got {type(selection.stop).__name__}"
                step = selection.step or 1
                start = selection.start if isinstance(selection.start, int) else (0 if step > 0 else self.get_size(sel_dim)-1)
                stop = selection.stop if isinstance(selection.stop, int) else (self.get_size(sel_dim) if step > 0 else -1)
                if stop < 0 and step > 0:
                    stop += self.get_size(sel_dim)
                    assert stop >= 0
                if start < 0 and step > 0:
                    start += self.get_size(sel_dim)
                    assert start >= 0
                new_size = math.to_int64(math.ceil(math.wrap((stop - start) / step)))
                if new_size.rank == 0:
                    new_size = int(new_size)  # NumPy array not allowed because not hashable
                result = result._replace_single_size(sel_dim, new_size, keep_item_names=True)
                if step < 0:
                    result = result.flipped([sel_dim])
                if self.get_item_names(sel_dim) is not None:
                    result = result._with_item_name(sel_dim, tuple(self.get_item_names(sel_dim)[selection]))
            elif isinstance(selection, (tuple, list)):
                result = result._replace_single_size(sel_dim, len(selection))
                if self.get_item_names(sel_dim) is not None:
                    result = result._with_item_name(sel_dim, tuple([self.get_item_names(sel_dim)[i] for i in selection]))
            else:
                raise NotImplementedError(f"{type(selection)} not supported. Only (int, slice) allowed.")
        return result

    def meshgrid(self, names=False):
        """
        Builds a sequence containing all multi-indices within a tensor of this shape.
        All indices are returned as `dict` mapping dimension names to `int` indices.

        The corresponding values can be retrieved from Tensors and other Sliceables using `tensor[index]`.

        This function currently only supports uniform tensors.

        Args:
            names: If `True`, replace indices by their item names if available.

        Returns:
            `dict` iterator.
        """
        assert self.is_uniform, f"Shape.meshgrid() is currently not supported for non-uniform tensors, {self}"
        indices = [0] * self.rank
        while True:
            if names:
                yield {dim: (names[index] if names is not None else index) for dim, index, names in zip(self.names, indices, self.item_names)}
            else:
                yield {dim: index for dim, index in zip(self.names, indices)}
            for i in range(self.rank-1, -1, -1):
                indices[i] = (indices[i] + 1) % self.sizes[i]
                if indices[i] != 0:
                    break
            else:
                return

    def first_index(self, names=False):
        return next(iter(self.meshgrid(names=names)))

    def are_adjacent(self, dims: str or tuple or list or set or 'Shape'):
        indices = self.indices(dims)
        return (max(indices) - min(indices)) == len(dims) - 1

    def __add__(self, other):
        return self._op2(other, lambda s, o: s + o, 0)

    def __radd__(self, other):
        return self._op2(other, lambda s, o: o + s, 0)

    def __sub__(self, other):
        return self._op2(other, lambda s, o: s - o, 0)

    def __rsub__(self, other):
        return self._op2(other, lambda s, o: o - s, 0)

    def __mul__(self, other):
        return self._op2(other, lambda s, o: s * o, 1)

    def __rmul__(self, other):
        return self._op2(other, lambda s, o: o * s, 1)

    def _op2(self, other, fun, default: int):
        if isinstance(other, int):
            return Shape(tuple([fun(s, other) for s in self.sizes]), self.names, self.types, (None,) * self.rank)
        elif isinstance(other, Shape):
            merged = self.without_sizes() & other.without_sizes()
            sizes = ()
            for dim in merged.names:
                self_val = self.get_size(dim) if dim in self else default
                other_val = other.get_size(dim) if dim in other else default
                sizes += (fun(self_val, other_val),)
            return merged.with_sizes(sizes)
        else:
            return NotImplemented

    def __hash__(self):
        return hash(self.names)


EMPTY_SHAPE = Shape((), (), (), ())
""" Empty shape, `()` """

DimFilter = Union[str, tuple, list, set, Shape, Callable]
try:
    DimFilter.__doc__ = """Dimension filters can be used with `Shape.only()` and `Shype.without()`, making them the standard tool for specifying sets of dimensions.
    
    The following types can be used as dimension filters:
    
    * `Shape` instances
    * `tuple` or `list` objects containing dimension names as `str`
    * Single `str` listing comma-separated dimension names
    * Any function `filter(Shape) -> Shape`, such as `math.batch()`, `math.non_batch()`, `math.spatial()`, etc.
    """  # docstring must be set explicitly
except AttributeError:  # on older Python versions, this is not possible
    pass


class IncompatibleShapes(Exception):
    """
    Raised when the shape of a tensor does not match the other arguments.
    """
    def __init__(self, message, *shapes: Shape):
        Exception.__init__(self, message)
        self.shapes = shapes


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


def parse_dim_order(order: str or tuple or list or Shape or None, check_rank: int = None) -> tuple or None:
    if order is None:
        if check_rank is not None:
            assert check_rank <= 1, "When calling Tensor.native() or Tensor.numpy(), the dimension order must be specified for Tensors with more than one dimension. The listed default dimension order can vary depending on the chosen backend. Consider using math.reshaped_native(Tensor) instead."
        return None
    elif isinstance(order, Shape):
        return order.names
    if isinstance(order, list):
        return tuple(order)
    elif isinstance(order, tuple):
        return order
    elif isinstance(order, str):
        parts = order.split(',')
        parts = [p.strip() for p in parts if p]
        return tuple(parts)
    raise ValueError(order)


def _construct_shape(dim_type: str, prefix: str, *args, **dims):
    sizes = ()
    names = []
    item_names = ()
    for arg in args:
        parts = [s.strip() for s in arg.split(',')]
        for name in parts:
            assert name not in names, f"Duplicate dimension name {name}"
            sizes += (None,)
            names.append(name)
            item_names += (None,)
    for name, size in dims.items():
        assert name not in names, f"Duplicate dimension name {name}"
        if isinstance(size, str):
            items = tuple([i.strip() for i in size.split(',')])
            size = len(items)
        elif isinstance(size, (tuple, list)):
            items = tuple(size)
            size = len(items)
        elif isinstance(size, Shape):
            items = size.names
            size = size.rank
        elif size is None or isinstance(size, int):
            # keep size
            items = None
        else:
            items = None
            from ._tensors import Tensor
            if isinstance(size, Tensor):
                size = int(size) if size.shape.volume == 1 else size
            else:
                try:
                    size = int(size)
                except ValueError:
                    raise ValueError(f"Cannot construct dimension from {type(size).__name__}. Only int, tuple, list, str or Shape allowed. Got {size}")
        names.append(name)
        sizes += (size,)
        item_names += (items,)
    names = tuple(_apply_prefix(name, prefix) for name in names)
    return math.Shape(sizes, names, (dim_type,) * len(sizes), item_names)


def _apply_prefix(name: str, prefix: str):
    match = re.search("\\w", name)
    assert match, f"Dimension name must contain at least one letter or underscore but got '{name}'"
    proper_name_index = match.start()
    return prefix + name[proper_name_index:]


def shape(obj) -> Shape:
    """
    If `obj` is a `Tensor` or `phi.math.magic.Shaped`, returns its shape.
    If `obj` is a `Shape`, returns `obj`.

    This function can be passed as a `dim` argument to an operation to specify that it should act upon all dimensions.

    Args:
        obj: `Tensor` or `Shape` or `Shaped`

    Returns:
        `Shape`
    """
    from phi.math.magic import PhiTreeNode
    if isinstance(obj, Shape):
        return obj
    elif hasattr(obj, '__shape__'):
        return obj.__shape__()
    elif hasattr(obj, 'shape') and isinstance(obj.shape, Shape):
        return obj.shape
    elif isinstance(obj, (int, float, complex, bool)):
        return EMPTY_SHAPE
    elif isinstance(obj, (tuple, list)) and all(isinstance(item, (int, float, complex, bool)) for item in obj):
        return channel('vector')
    elif isinstance(obj, (Number, bool)):
        return EMPTY_SHAPE
    elif isinstance(obj, (tuple, list)) and all(isinstance(item, PhiTreeNode) for item in obj):
        return merge_shapes(*obj, allow_varying_sizes=True)
    elif isinstance(obj, PhiTreeNode):
        from phi.math._magic_ops import all_attributes
        return merge_shapes(*[getattr(obj, a) for a in all_attributes(obj, assert_any=True)], allow_varying_sizes=True)
    else:
        from .backend import choose_backend, NoBackendFound
        try:
            backend = choose_backend(obj)
            shape_tuple = backend.staticshape(obj)
            if len(shape_tuple) == 0:
                return EMPTY_SHAPE
            elif len(shape_tuple) == 1:
                return channel('vector')
            else:
                raise ValueError(f"Cannot auto-complete shape of {backend} tensor with shape {shape_tuple}. Only 0D and 1D tensors have a Φ-Flow shape by default.")
        except NoBackendFound:
            raise ValueError(f'shape() requires Shaped or Shape argument but got {type(obj)}')


def spatial(*args, **dims: int or str or tuple or list or Shape) -> Shape:
    """
    Returns the spatial dimensions of an existing `Shape` or creates a new `Shape` with only spatial dimensions.

    Usage for filtering spatial dimensions:
    >>> spatial_dims = spatial(shape)
    >>> spatial_dims = spatial(tensor)

    Usage for creating a `Shape` with only spatial dimensions:
    >>> spatial_shape = spatial('undef', x=2, y=3)
    (x=2, y=3, undef=None)

    Here, the dimension `undef` is created with an undefined size of `None`.
    Undefined sizes are automatically filled in by `tensor`, `wrap`, `stack` and `concat`.

    To create a shape with multiple types, use `merge_shapes()`, `concat_shapes()` or the syntax `shape1 & shape2`.

    See Also:
        `channel`, `batch`, `instance`

    Args:
        *args: Either

            * `Shape` or `Tensor` to filter or
            * Names of dimensions with undefined sizes as `str`.

        **dims: Dimension sizes and names. Must be empty when used as a filter operation.

    Returns:
        `Shape` containing only dimensions of type spatial.
    """
    from .magic import Shaped
    if all(isinstance(arg, str) for arg in args) or dims:
        return _construct_shape(SPATIAL_DIM, '', *args, **dims)
    elif len(args) == 1 and isinstance(args[0], Shape):
        return args[0].spatial
    elif len(args) == 1 and isinstance(args[0], Shaped):
        return shape(args[0]).spatial
    else:
        raise AssertionError(f"spatial() must be called either as a selector spatial(Shape) or spatial(Tensor) or as a constructor spatial(*names, **dims). Got *args={args}, **dims={dims}")


def channel(*args, **dims: int or str or tuple or list or Shape) -> Shape:
    """
    Returns the channel dimensions of an existing `Shape` or creates a new `Shape` with only channel dimensions.

    Usage for filtering channel dimensions:
    >>> channel_dims = channel(shape)
    >>> channel_dims = channel(tensor)

    Usage for creating a `Shape` with only channel dimensions:
    >>> channel_shape = channel('undef', vector=2)
    (vector=2, undef=None)

    Here, the dimension `undef` is created with an undefined size of `None`.
    Undefined sizes are automatically filled in by `tensor`, `wrap`, `stack` and `concat`.

    To create a shape with multiple types, use `merge_shapes()`, `concat_shapes()` or the syntax `shape1 & shape2`.

    See Also:
        `spatial`, `batch`, `instance`

    Args:
        *args: Either

            * `Shape` or `Tensor` to filter or
            * Names of dimensions with undefined sizes as `str`.

        **dims: Dimension sizes and names. Must be empty when used as a filter operation.

    Returns:
        `Shape` containing only dimensions of type channel.
    """
    from .magic import Shaped
    if all(isinstance(arg, str) for arg in args) or dims:
        return _construct_shape(CHANNEL_DIM, '', *args, **dims)
    elif len(args) == 1 and isinstance(args[0], Shape):
        return args[0].channel
    elif len(args) == 1 and isinstance(args[0], Shaped):
        return shape(args[0]).channel
    else:
        raise AssertionError(f"channel() must be called either as a selector channel(Shape) or channel(Tensor) or as a constructor channel(*names, **dims). Got *args={args}, **dims={dims}")


def batch(*args, **dims: int or str or tuple or list or Shape) -> Shape:
    """
    Returns the batch dimensions of an existing `Shape` or creates a new `Shape` with only batch dimensions.

    Usage for filtering batch dimensions:
    >>> batch_dims = batch(shape)
    >>> batch_dims = batch(tensor)

    Usage for creating a `Shape` with only batch dimensions:
    >>> batch_shape = batch('undef', batch=2)
    (batch=2, undef=None)

    Here, the dimension `undef` is created with an undefined size of `None`.
    Undefined sizes are automatically filled in by `tensor`, `wrap`, `stack` and `concat`.

    To create a shape with multiple types, use `merge_shapes()`, `concat_shapes()` or the syntax `shape1 & shape2`.

    See Also:
        `channel`, `spatial`, `instance`

    Args:
        *args: Either

            * `Shape` or `Tensor` to filter or
            * Names of dimensions with undefined sizes as `str`.

        **dims: Dimension sizes and names. Must be empty when used as a filter operation.

    Returns:
        `Shape` containing only dimensions of type batch.
    """
    from .magic import Shaped
    if all(isinstance(arg, str) for arg in args) or dims:
        return _construct_shape(BATCH_DIM, '', *args, **dims)
    elif len(args) == 1 and isinstance(args[0], Shape):
        return args[0].batch
    elif len(args) == 1 and isinstance(args[0], Shaped):
        return shape(args[0]).batch
    else:
        raise AssertionError(f"batch() must be called either as a selector batch(Shape) or batch(Tensor) or as a constructor batch(*names, **dims). Got *args={args}, **dims={dims}")


def instance(*args, **dims: int or str or tuple or list or Shape) -> Shape:
    """
    Returns the instance dimensions of an existing `Shape` or creates a new `Shape` with only instance dimensions.

    Usage for filtering instance dimensions:
    >>> instance_dims = instance(shape)
    >>> instance_dims = instance(tensor)

    Usage for creating a `Shape` with only instance dimensions:
    >>> instance_shape = instance('undef', points=2)
    (points=2, undef=None)

    Here, the dimension `undef` is created with an undefined size of `None`.
    Undefined sizes are automatically filled in by `tensor`, `wrap`, `stack` and `concat`.

    To create a shape with multiple types, use `merge_shapes()`, `concat_shapes()` or the syntax `shape1 & shape2`.

    See Also:
        `channel`, `batch`, `spatial`

    Args:
        *args: Either

            * `Shape` or `Tensor` to filter or
            * Names of dimensions with undefined sizes as `str`.

        **dims: Dimension sizes and names. Must be empty when used as a filter operation.

    Returns:
        `Shape` containing only dimensions of type instance.
    """
    from .magic import Shaped
    if all(isinstance(arg, str) for arg in args) or dims:
        return _construct_shape(INSTANCE_DIM, '', *args, **dims)
    elif len(args) == 1 and isinstance(args[0], Shape):
        return args[0].instance
    elif len(args) == 1 and isinstance(args[0], Shaped):
        return shape(args[0]).instance
    else:
        raise AssertionError(f"instance() must be called either as a selector instance(Shape) or instance(Tensor) or as a constructor instance(*names, **dims). Got *args={args}, **dims={dims}")


def dual(*args, **dims: int or str or tuple or list or Shape) -> Shape:
    """
    Returns the dual dimensions of an existing `Shape` or creates a new `Shape` with only dual dimensions.

    Dual dimensions are assigned the prefix `~` to distinguish them from regular dimensions.
    This way, a regular and dual dimension of the same name can exist in one `Shape`.

    Dual dimensions represent the input space and are typically only present on matrices or higher-order matrices.
    Dual dimensions behave like batch dimensions in regular operations, if supported.
    During matrix multiplication, they are matched against their regular counterparts by name (ignoring the `~` prefix).

    Usage for filtering dual dimensions:

    >>> dual_dims = dual(shape)
    >>> dual_dims = dual(tensor)

    Usage for creating a `Shape` with only dual dimensions:

    >>> dual('undef', points=2)
    (~undefᵈ=None, ~pointsᵈ=2)

    Here, the dimension `undef` is created with an undefined size of `None`.
    Undefined sizes are automatically filled in by `tensor`, `wrap`, `stack` and `concat`.

    To create a shape with multiple types, use `merge_shapes()`, `concat_shapes()` or the syntax `shape1 & shape2`.

    See Also:
        `channel`, `batch`, `spatial`

    Args:
        *args: Either

            * `Shape` or `Tensor` to filter or
            * Names of dimensions with undefined sizes as `str`.

        **dims: Dimension sizes and names. Must be empty when used as a filter operation.

    Returns:
        `Shape` containing only dimensions of type dual.
    """
    from .magic import Shaped
    if all(isinstance(arg, str) for arg in args) or dims:
        return _construct_shape(DUAL_DIM, '~', *args, **dims)
    elif len(args) == 1 and isinstance(args[0], Shape):
        return args[0].dual
    elif len(args) == 1 and isinstance(args[0], Shaped):
        return shape(args[0]).dual
    else:
        raise AssertionError(f"dual() must be called either as a selector dual(Shape) or dual(Tensor) or as a constructor dual(*names, **dims). Got *args={args}, **dims={dims}")


def merge_shapes(*objs: Shape or Any, order=(batch, dual, instance, spatial, channel), allow_varying_sizes=False):
    """
    Combines `shapes` into a single `Shape`, grouping dimensions by type.
    If dimensions with equal names are present in multiple shapes, their types and sizes must match.

    The shorthand `shape1 & shape2` merges shapes with `check_exact=[spatial]`.

    See Also:
        `concat_shapes()`.

    Args:
        *objs: `Shape` or `Shaped` objects to combine.
        order: Dimension type order as `tuple` of type filters (`channel`, `batch`, `spatial` or `instance`). Dimensions are grouped by type while merging.

    Returns:
        Merged `Shape`

    Raises:
        IncompatibleShapes if the shapes are not compatible
    """
    if not objs:
        return EMPTY_SHAPE
    shapes = [obj if isinstance(obj, Shape) else shape(obj) for obj in objs]
    merged = []
    for dim_type in order:
        type_group = dim_type(shapes[0])
        for sh in shapes[1:]:
            sh = dim_type(sh)
            for dim in sh:
                if dim not in type_group:
                    type_group = type_group._expand(dim, pos=-1)
                else:  # check size match
                    sizes_match = _size_equal(dim.size, type_group.get_size(dim.name))
                    if allow_varying_sizes:
                        if not sizes_match:
                            type_group = type_group.with_dim_size(dim, None)
                    else:
                        if not sizes_match:
                            raise IncompatibleShapes(f"Cannot merge shapes {shapes} because dimension '{dim.name}' exists with different sizes.", *shapes)
                        names1 = type_group.get_item_names(dim)
                        names2 = sh.get_item_names(dim)
                        if names1 is not None and names2 is not None and len(names1) > 1:
                            if names1 != names2:
                                if set(names1) == set(names2):
                                    raise IncompatibleShapes(f"Inconsistent component order: '{','.join(names1)}' vs '{','.join(names2)}' in dimension '{dim.name}'. Failed to merge shapes {shapes}", *shapes)
                                else:
                                    raise IncompatibleShapes(f"Cannot merge shapes {shapes} because dimension '{dim.name}' exists with different item names.", *shapes)
                        elif names1 is None and names2 is not None:
                            type_group = type_group._with_item_name(dim, tuple(names2))
        merged.append(type_group)
    return concat_shapes(*merged)


def non_batch(obj) -> Shape:
    """
    Returns the non-batch dimensions of an object.

    Args:
        obj: `Shape` or object with a valid `shape` property.

    Returns:
        `Shape`
    """
    from .magic import Shaped
    if isinstance(obj, Shape):
        return obj.non_batch
    elif isinstance(obj, Shaped):
        return shape(obj).non_batch
    else:
        raise AssertionError(f"non_batch() must be called either on a Shape or an object with a 'shape' property but got {obj}")


def non_spatial(obj) -> Shape:
    """
    Returns the non-spatial dimensions of an object.

    Args:
        obj: `Shape` or object with a valid `shape` property.

    Returns:
        `Shape`
    """
    from .magic import Shaped
    if isinstance(obj, Shape):
        return obj.non_spatial
    elif isinstance(obj, Shaped):
        return shape(obj).non_spatial
    else:
        raise AssertionError(f"non_spatial() must be called either on a Shape or an object with a 'shape' property but got {obj}")


def non_instance(obj) -> Shape:
    """
    Returns the non-instance dimensions of an object.

    Args:
        obj: `Shape` or object with a valid `shape` property.

    Returns:
        `Shape`
    """
    from .magic import Shaped
    if isinstance(obj, Shape):
        return obj.non_instance
    elif isinstance(obj, Shaped):
        return shape(obj).non_instance
    else:
        raise AssertionError(f"non_instance() must be called either on a Shape or an object with a 'shape' property but got {obj}")


def non_channel(obj) -> Shape:
    """
    Returns the non-channel dimensions of an object.

    Args:
        obj: `Shape` or object with a valid `shape` property.

    Returns:
        `Shape`
    """
    from .magic import Shaped
    if isinstance(obj, Shape):
        return obj.non_channel
    elif isinstance(obj, Shaped):
        return shape(obj).non_channel
    else:
        raise AssertionError(f"non_channel() must be called either on a Shape or an object with a 'shape' property but got {obj}")


def non_dual(obj) -> Shape:
    """
    Returns the non-dual dimensions of an object.

    Args:
        obj: `Shape` or object with a valid `shape` property.

    Returns:
        `Shape`
    """
    from .magic import Shaped
    if isinstance(obj, Shape):
        return obj.non_dual
    elif isinstance(obj, Shaped):
        return shape(obj).non_dual
    else:
        raise AssertionError(f"non_dual() must be called either on a Shape or an object with a 'shape' property but got {obj}")



def _size_equal(s1, s2):
    if s1 is None:
        return s2 is None
    if isinstance(s1, int):
        return isinstance(s2, int) and s2 == s1
    else:
        return math.close(s1, s2)


def concat_shapes(*shapes: Shape or Any) -> Shape:
    """
    Creates a `Shape` listing the dimensions of all `shapes` in the given order.

    See Also:
        `merge_shapes()`.

    Args:
        *shapes: Shapes to concatenate. No two shapes must contain a dimension with the same name.

    Returns:
        Combined `Shape`.
    """
    shapes = [obj if isinstance(obj, Shape) else shape(obj) for obj in shapes]
    names = sum([s.names for s in shapes], ())
    if len(set(names)) != len(names):
        raise IncompatibleShapes(f"Cannot concatenate shapes {list(shapes)}. Duplicate dimension names are not allowed.")
    sizes = sum([s.sizes for s in shapes], ())
    types = sum([s.types for s in shapes], ())
    item_names = sum([s.item_names for s in shapes], ())
    return Shape(sizes, names, types, item_names)


def shape_stack(stack_dim: Shape, *shapes: Shape):
    """ Returns the shape of a tensor created by stacking tensors with `shapes`. """
    names = list(stack_dim.names)
    types = list(stack_dim.types)
    item_names = list(stack_dim.item_names)
    for other in shapes:
        for size, name, type, items in other._dimensions:
            if name not in names:
                if type in types:
                    index = len(types) - types[::-1].index(type)
                elif type == BATCH_DIM:
                    index = 0
                elif type == DUAL_DIM:
                    index = min([len(names), *[i for i in range(len(names)) if types[i] == DUAL_DIM]])
                elif type == CHANNEL_DIM:
                    index = len(names)
                elif type == SPATIAL_DIM:
                    index = min([len(names), *[i for i in range(len(names)) if types[i] == CHANNEL_DIM]])
                elif type == INSTANCE_DIM:
                    index = min([len(names), *[i for i in range(len(names)) if types[i] == INSTANCE_DIM]])
                else:
                    raise ValueError(type)
                names.insert(index, name)
                types.insert(index, type)
                item_names.insert(index, items)
            else:
                index = names.index(name)
                if items != item_names[index]:
                    if item_names[index] is None:
                        item_names[index] = items
                    else:
                        warnings.warn(f"Stacking shapes with incompatible item names will result in item names being lost. Got {item_names[index]} and {items}", RuntimeWarning)
                        item_names[index] = None
    sizes = []
    for name in names:
        if name == stack_dim.name:
            size = len(shapes)
        else:
            dim_sizes = [(shape.get_size(name) if name in shape else 1) for shape in shapes]
            if all([math.close(s, dim_sizes[0]) for s in dim_sizes[1:]]):
                size = dim_sizes[0]
            else:
                from ._magic_ops import stack
                from ._tensors import wrap
                dim_sizes = [wrap(d) for d in dim_sizes]
                size = stack(dim_sizes, stack_dim)
        sizes.append(size)
    return Shape(tuple(sizes), tuple(names), tuple(types), tuple(item_names))


def vector_add(*shapes: Shape):
    if not shapes:
        return EMPTY_SHAPE
    names = shapes[0].names
    types = shapes[0].types
    item_names = shapes[0].item_names
    for shape in shapes[1:]:
        for name in shape.names:
            if name not in names:
                names += (name,)
                types += (shape.get_type(name),)
                item_names += (shape.get_item_names(name),)
    sizes = [sum(sh.get_size(dim) if dim in sh else 0 for sh in shapes) for dim in names]
    return Shape(tuple(sizes), names, types, item_names)
