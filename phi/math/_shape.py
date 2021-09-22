import warnings
from typing import Tuple

from phi import math


BATCH_DIM = 'batch'
SPATIAL_DIM = 'spatial'
CHANNEL_DIM = 'channel'
INSTANCE_DIM = 'înstance'
TYPE_ABBR = {SPATIAL_DIM: "ˢ", CHANNEL_DIM: "ᶜ", INSTANCE_DIM: "ⁱ", BATCH_DIM: "ᵇ", None: "⁻"}  # ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ


class Shape:
    """
    Shapes enumerate dimensions, each consisting of a name, size and type.

    There are four types of dimensions: `batch`, `spatial`, `channel`, and `instance`.
    """

    def __init__(self, sizes: tuple or list, names: tuple or list, types: tuple or list):
        """
        To construct a `Shape`, use `batch`, `spatial`, `channel` or `instance`, depending on the desired dimension type.
        To create a shape with multiple types, use `merge_shapes()`, `concat_shapes()` or the syntax `shape1 & shape2`.

        The `__init__` constructor is for internal use only.
        """
        assert len(sizes) == len(names) == len(types), f"sizes={sizes} ({len(sizes)}), names={names} ({len(names)}), types={types} ({len(types)})"
        if len(sizes) > 0:
            from ._tensors import Tensor
            sizes = tuple([s if isinstance(s, Tensor) or s is None else int(s) for s in sizes])
        else:
            sizes = ()
        self.sizes: tuple = sizes
        """
        Ordered dimension sizes as `tuple`.
        The size of a dimension can be an `int` or a `Tensor` for [non-uniform shapes](https://tum-pbs.github.io/PhiFlow/Math.html#non-uniform-tensors).
        
        See Also:
            `Shape.get_size()`, `Shape.size`, `Shape.shape`.
        """
        self.names: Tuple[str] = tuple(names)
        """
        Ordered dimension names as `tuple[str]`.
        
        See Also:
            `Shape.name`.
        """
        assert all(isinstance(n, str) for n in names), f"All names must be of type string but got {names}"
        self.types: Tuple[str] = tuple(types)  # undocumented, may be private

    @property
    def _named_sizes(self):
        return zip(self.names, self.sizes)

    @property
    def _dimensions(self):
        return zip(self.sizes, self.names, self.types)

    def __len__(self):
        return len(self.sizes)

    def __contains__(self, item):
        if isinstance(item, str):
            return item in self.names
        elif isinstance(item, Shape):
            return all([d in self.names for d in item.names])
        else:
            raise ValueError(item)

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
        if isinstance(dims, (list, tuple)):
            return tuple(self.index(n) for n in dims)
        elif isinstance(dims, Shape):
            return tuple(self.index(n) for n in dims.names)
        else:
            raise ValueError(f"indices() requires a sequence of dimensions but got {dims}")

    def get_size(self, dim: str or 'Shape'):
        """
        See Also:
            `Shape.get_sizes()`, `Shape.size`

        Args:
            dim: Dimension, either as name `str` or single-dimension `Shape`.

        Returns:
            Size associated with `dim` as `int` or `Tensor`.
        """
        if isinstance(dim, str):
            return self.sizes[self.names.index(dim)]
        elif isinstance(dim, Shape):
            assert dim.rank == 1, f"get_size() requires a single dimension but got {dim}. Use indices() to get multiple sizes."
            return self.sizes[self.names.index(dim.name)]
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
        """
        See Also:
            `Shape.get_types()`, `Shape.type`.

        Args:
            dim: Dimension, either as name `str` or single-dimension `Shape`.

        Returns:
            Dimension type as `str`.
        """
        if isinstance(dim, str):
            return self.types[self.names.index(dim)]
        elif isinstance(dim, Shape):
            assert dim.rank == 1, f"Shape.get_type() only accepts single-dimension Shapes but got {dim}"
            return self.types[self.names.index(dim.name)]
        else:
            raise ValueError(dim)

    def get_types(self, dims: tuple or list or 'Shape') -> tuple:
        """
        See Also:
            `Shape.get_type()`

        Args:
            dims: Dimensions as `tuple`, `list` or `Shape`.

        Returns:
            `tuple`
        """
        if isinstance(dims, (tuple, list)):
            return tuple(self.get_type(n) for n in dims)
        elif isinstance(dims, Shape):
            return tuple(self.get_type(n) for n in dims.names)
        else:
            raise ValueError(dims)

    def __getitem__(self, selection):
        if isinstance(selection, int):
            return Shape([self.sizes[selection]], [self.names[selection]], [self.types[selection]])
        elif isinstance(selection, slice):
            return Shape(self.sizes[selection], self.names[selection], self.types[selection])
        elif isinstance(selection, str):
            index = self.index(selection)
            return Shape([self.sizes[index]], [self.names[index]], [self.types[index]])
        elif isinstance(selection, (tuple, list)):
            return Shape([self.sizes[i] for i in selection], [self.names[i] for i in selection], [self.types[i] for i in selection])
        raise AssertionError("Can only access shape elements as shape[int] or shape[slice]")

    @property
    def batch(self) -> 'Shape':
        """
        Filters this shape, returning only the batch dimensions as a new `Shape` object.

        See also:
            `Shape.batch`, `Shape.spatial`, `Shape.instance`, `Shape.channel`, `Shape.non_batch`, `Shape.non_spatial`, `Shape.non_instance`, `Shape.non_channel`.

        Returns:
            New `Shape` object
        """
        return self[[i for i, t in enumerate(self.types) if t == BATCH_DIM]]

    @property
    def non_batch(self) -> 'Shape':
        """
        Filters this shape, returning only the non-batch dimensions as a new `Shape` object.

        See also:
            `Shape.batch`, `Shape.spatial`, `Shape.instance`, `Shape.channel`, `Shape.non_batch`, `Shape.non_spatial`, `Shape.non_instance`, `Shape.non_channel`.

        Returns:
            New `Shape` object
        """
        return self[[i for i, t in enumerate(self.types) if t != BATCH_DIM]]

    @property
    def spatial(self) -> 'Shape':
        """
        Filters this shape, returning only the spatial dimensions as a new `Shape` object.

        See also:
            `Shape.batch`, `Shape.spatial`, `Shape.instance`, `Shape.channel`, `Shape.non_batch`, `Shape.non_spatial`, `Shape.non_instance`, `Shape.non_channel`.

        Returns:
            New `Shape` object
        """
        return self[[i for i, t in enumerate(self.types) if t == SPATIAL_DIM]]

    @property
    def non_spatial(self) -> 'Shape':
        """
        Filters this shape, returning only the non-spatial dimensions as a new `Shape` object.

        See also:
            `Shape.batch`, `Shape.spatial`, `Shape.instance`, `Shape.channel`, `Shape.non_batch`, `Shape.non_spatial`, `Shape.non_instance`, `Shape.non_channel`.

        Returns:
            New `Shape` object
        """
        return self[[i for i, t in enumerate(self.types) if t != SPATIAL_DIM]]

    @property
    def instance(self) -> 'Shape':
        """
        Filters this shape, returning only the instance dimensions as a new `Shape` object.

        See also:
            `Shape.batch`, `Shape.spatial`, `Shape.instance`, `Shape.channel`, `Shape.non_batch`, `Shape.non_spatial`, `Shape.non_instance`, `Shape.non_channel`.

        Returns:
            New `Shape` object
        """
        return self[[i for i, t in enumerate(self.types) if t == INSTANCE_DIM]]

    @property
    def non_instance(self) -> 'Shape':
        """
        Filters this shape, returning only the non-instance dimensions as a new `Shape` object.

        See also:
            `Shape.batch`, `Shape.spatial`, `Shape.instance`, `Shape.channel`, `Shape.non_batch`, `Shape.non_spatial`, `Shape.non_instance`, `Shape.non_channel`.

        Returns:
            New `Shape` object
        """
        return self[[i for i, t in enumerate(self.types) if t != INSTANCE_DIM]]

    @property
    def channel(self) -> 'Shape':
        """
        Filters this shape, returning only the channel dimensions as a new `Shape` object.

        See also:
            `Shape.batch`, `Shape.spatial`, `Shape.instance`, `Shape.channel`, `Shape.non_batch`, `Shape.non_spatial`, `Shape.non_instance`, `Shape.non_channel`.

        Returns:
            New `Shape` object
        """
        return self[[i for i, t in enumerate(self.types) if t == CHANNEL_DIM]]

    @property
    def non_channel(self) -> 'Shape':
        """
        Filters this shape, returning only the non-channel dimensions as a new `Shape` object.

        See also:
            `Shape.batch`, `Shape.spatial`, `Shape.instance`, `Shape.channel`, `Shape.non_batch`, `Shape.non_spatial`, `Shape.non_instance`, `Shape.non_channel`.

        Returns:
            New `Shape` object
        """
        return self[[i for i, t in enumerate(self.types) if t != CHANNEL_DIM]]

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
            return tuple(Shape([self.sizes[i]], [self.names[i]], [self.types[i]]) for i in range(self.rank))
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
            shapes = tuple(Shape([int(size[i]) if isinstance(size, tuple) else size for size in sizes], inner.names, inner.types) for i in range(dim_size))
            return shapes

    @property
    def name(self) -> str:
        """
        Only for Shapes containing exactly one single dimension.
        Returns the name of the dimension.

        See Also:
            `Shape.names`.
        """
        assert self.rank == 1, "Shape.name is only defined for shapes of rank 1."
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
        strings = [f"{name}{TYPE_ABBR.get(dim_type, '?')}={size}" for size, name, dim_type in self._dimensions]
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
        return merge_shapes(self, other, check_exact=[spatial])

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
        sizes.insert(pos, dim.size)
        names.insert(pos, dim.name)
        types.insert(pos, dim.type)
        return Shape(sizes, names, types)

    def without(self, dims: str or tuple or list or 'Shape') -> 'Shape':
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
        if isinstance(dims, str):
            return self[[i for i in range(self.rank) if self.names[i] != dims]]
        if isinstance(dims, (tuple, list)):
            return self[[i for i in range(self.rank) if self.names[i] not in dims]]
        elif isinstance(dims, Shape):
            return self[[i for i in range(self.rank) if self.names[i] not in dims.names]]
        # elif dims is None:  # subtract all
        #     return EMPTY_SHAPE
        else:
            raise ValueError(dims)

    def only(self, dims: str or tuple or list or 'Shape'):
        """
        Builds a new shape from this one that only contains the given dimensions.
        Dimensions in `dims` that are not part of this Shape are ignored.
        
        The complementary operation is :func:`Shape.without`.

        Args:
          dims: single dimension (str) or instance of dimensions (tuple, list, Shape)
          dims: str or tuple or list or Shape: 

        Returns:
          Shape containing only specified dimensions

        """
        if isinstance(dims, str):
            dims = parse_dim_order(dims)
        if isinstance(dims, (tuple, list)):
            return self[[i for i in range(self.rank) if self.names[i] in dims]]
        elif isinstance(dims, Shape):
            return self[[i for i in range(self.rank) if self.names[i] in dims.names]]
        elif dims is None:  # keep all
            return self
        else:
            raise ValueError(dims)

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
    def instance_rank(self) -> int:
        """ Number of instance dimensions """
        r = 0
        for ty in self.types:
            if ty == INSTANCE_DIM:
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
        shape = Shape([self.rank], ['dims'], [CHANNEL_DIM])
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

    def with_size(self, size: int):
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

    def with_sizes(self, sizes: tuple or list or 'Shape'):
        """
        Returns a new `Shape` matching the dimension names and types of `self` but with different sizes.

        See Also:
            `Shape.with_size()`.

        Args:
            sizes: One of

                * `tuple` / `list` of same length as `self` containing replacement sizes.
                * `Shape` of any rank. Replaces sizes for dimensions shared by `sizes` and `self`.

        Returns:
            `Shape` with same names and types as `self`.
        """
        if isinstance(sizes, Shape):
            sizes = [sizes.get_size(dim) if dim in sizes else self.sizes[i] for i, dim in enumerate(self.names)]
            return Shape(sizes, self.names, self.types)
        else:
            assert len(sizes) == len(self.sizes), f"Cannot create shape from {self} with sizes {sizes}"
            return Shape(sizes, self.names, self.types)

    def _replace_single_size(self, dim: str, size: int):
        new_sizes = list(self.sizes)
        new_sizes[self.index(dim)] = size
        return self.with_sizes(new_sizes)

    def _with_names(self, names: str or tuple or list):
        if isinstance(names, str):
            names = parse_dim_names(names, self.rank)
            names = [n if n is not None else o for n, o in zip(names, self.names)]
        return Shape(self.sizes, names, self.types)

    def _replace_names_and_types(self,
                                 dims: 'Shape' or str or tuple or list,
                                 new: 'Shape' or str or tuple or list) -> 'Shape':
        dims = parse_dim_order(dims)
        sizes = [math.rename_dims(s, dims, new) if isinstance(s, math.Tensor) else s for s in self.sizes]
        if isinstance(new, Shape):  # replace names and types
            names = list(self.names)
            types = list(self.types)
            for old_name, new_dim in zip(dims, new):
                names[self.index(old_name)] = new_dim.name
                types[self.index(old_name)] = new_dim.type
            return Shape(sizes, names, types)
        else:  # replace only names
            new = parse_dim_order(new)
            names = list(self.names)
            for old_name, new_name in zip(dims, new):
                names[self.index(old_name)] = new_name
            return Shape(sizes, names, self.types)

    def _with_types(self, types: 'Shape'):
        return Shape(self.sizes, self.names, [types.get_type(name) if name in types else self_type for name, self_type in zip(self.names, self.types)])

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
        for dim, (lo, up) in widths.items():
            sizes[self.index(dim)] += lo + up
        return Shape(sizes, self.names, self.types)

    def after_gather(self, selection: dict) -> 'Shape':
        result = self
        for name, selection in selection.items():
            if name not in self.names:
                continue
            if isinstance(selection, int):
                if result.is_uniform:
                    result = result.without(name)
                else:
                    from phi.math import Tensor
                    gathered_sizes = [(s[{name: selection}] if isinstance(s, Tensor) else s) for s in result.sizes]
                    result = result.with_sizes(gathered_sizes).without(name)
            elif isinstance(selection, slice):
                start = selection.start or 0
                stop = selection.stop or self.get_size(name)
                step = selection.step or 1
                if stop < 0:
                    stop += self.get_size(name)
                    assert stop >= 0
                new_size = math.to_int64(math.ceil(math.wrap((stop - start) / step)))
                if new_size.rank == 0:
                    new_size = int(new_size)  # NumPy array not allowed because not hashable
                result = result._replace_single_size(name, new_size)
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

    def __add__(self, other):
        return self._op2(other, lambda s, o: s + o)

    def __radd__(self, other):
        return self._op2(other, lambda s, o: o + s)

    def __sub__(self, other):
        return self._op2(other, lambda s, o: s - o)

    def __rsub__(self, other):
        return self._op2(other, lambda s, o: o - s)

    def __mul__(self, other):
        return self._op2(other, lambda s, o: s * o)

    def __rmul__(self, other):
        return self._op2(other, lambda s, o: o * s)

    def _op2(self, other, fun):
        if isinstance(other, int):
            return Shape([fun(s, other) for s in self.sizes], self.names, self.types)
        elif isinstance(other, Shape):
            assert self.names == other.names, f"{self.names, other.names}"
            return Shape([fun(s, o) for s, o in zip(self.sizes, other.sizes)], self.names, self.types)
        else:
            return NotImplemented

    def __hash__(self):
        return hash(self.names)


EMPTY_SHAPE = Shape((), (), ())


class IncompatibleShapes(ValueError):
    def __init__(self, message, *shapes: Shape):
        ValueError.__init__(self, message)
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


def spatial(*args, **dims: int):
    """
    Returns the spatial dimensions of an existing `Shape` or creates a new `Shape` with only spatial dimensions.

    Usage for filtering spatial dimensions:
    ```python
    spatial_dims = spatial(shape)
    spatial_dims = spatial(tensor)
    ```

    Usage for creating a `Shape` with only spatial dimensions:
    ```python
    spatial_shape = spatial('undef', x=2, y=3)
    # Out: (x=2, y=3, undef=None)
    ```
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
    from phi.math import Tensor
    if all(isinstance(arg, str) for arg in args) or dims:
        for arg in args:
            parts = [s.strip() for s in arg.split(',')]
            for dim in parts:
                if dim not in dims:
                    dims[dim] = None
        return math.Shape(dims.values(), dims.keys(), [SPATIAL_DIM] * len(dims))
    elif len(args) == 1 and isinstance(args[0], Shape):
        return args[0].spatial
    elif len(args) == 1 and isinstance(args[0], Tensor):
        return args[0].shape.spatial
    else:
        raise AssertionError(f"spatial() must be called either as a selector spatial(Shape) or spatial(Tensor) or as a constructor spatial(*names, **dims). Got *args={args}, **dims={dims}")


def channel(*args, **dims: int):
    """
    Returns the channel dimensions of an existing `Shape` or creates a new `Shape` with only channel dimensions.

    Usage for filtering channel dimensions:
    ```python
    channel_dims = channel(shape)
    channel_dims = channel(tensor)
    ```

    Usage for creating a `Shape` with only channel dimensions:
    ```python
    channel_shape = channel('undef', vector=2)
    # Out: (vector=2, undef=None)
    ```
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
    from ._tensors import Tensor
    if all(isinstance(arg, str) for arg in args) or dims:
        for arg in args:
            parts = [s.strip() for s in arg.split(',')]
            for dim in parts:
                if dim not in dims:
                    dims[dim] = None
        return math.Shape(dims.values(), dims.keys(), [CHANNEL_DIM] * len(dims))
    elif len(args) == 1 and isinstance(args[0], Shape):
        return args[0].channel
    elif len(args) == 1 and isinstance(args[0], Tensor):
        return args[0].shape.channel
    else:
        raise AssertionError(f"channel() must be called either as a selector channel(Shape) or channel(Tensor) or as a constructor channel(*names, **dims). Got *args={args}, **dims={dims}")


def batch(*args, **dims: int):
    """
    Returns the batch dimensions of an existing `Shape` or creates a new `Shape` with only batch dimensions.

    Usage for filtering batch dimensions:
    ```python
    batch_dims = batch(shape)
    batch_dims = batch(tensor)
    ```

    Usage for creating a `Shape` with only batch dimensions:
    ```python
    batch_shape = batch('undef', batch=2)
    # Out: (batch=2, undef=None)
    ```
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
    from phi.math import Tensor
    if all(isinstance(arg, str) for arg in args) or dims:
        for arg in args:
            parts = [s.strip() for s in arg.split(',')]
            for dim in parts:
                if dim not in dims:
                    dims[dim] = None
        return math.Shape(dims.values(), dims.keys(), [BATCH_DIM] * len(dims))
    elif len(args) == 1 and isinstance(args[0], Shape):
        return args[0].batch
    elif len(args) == 1 and isinstance(args[0], Tensor):
        return args[0].shape.batch
    else:
        raise AssertionError(f"batch() must be called either as a selector batch(Shape) or batch(Tensor) or as a constructor batch(*names, **dims). Got *args={args}, **dims={dims}")


def instance(*args, **dims: int):
    """
    Returns the instance dimensions of an existing `Shape` or creates a new `Shape` with only instance dimensions.

    Usage for filtering instance dimensions:
    ```python
    instance_dims = instance(shape)
    instance_dims = instance(tensor)
    ```

    Usage for creating a `Shape` with only instance dimensions:
    ```python
    instance_shape = instance('undef', points=2)
    # Out: (points=2, undef=None)
    ```
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
    from phi.math import Tensor
    if all(isinstance(arg, str) for arg in args) or dims:
        for arg in args:
            parts = [s.strip() for s in arg.split(',')]
            for dim in parts:
                if dim not in dims:
                    dims[dim] = None
        return math.Shape(dims.values(), dims.keys(), [INSTANCE_DIM] * len(dims))
    elif len(args) == 1 and isinstance(args[0], Shape):
        return args[0].instance
    elif len(args) == 1 and isinstance(args[0], Tensor):
        return args[0].shape.instance
    else:
        raise AssertionError(f"instance() must be called either as a selector instance(Shape) or instance(Tensor) or as a constructor instance(*names, **dims). Got *args={args}, **dims={dims}")


def merge_shapes(*shapes: Shape, check_exact: tuple or list = (), order=(batch, instance, spatial, channel)):
    """
    Combines `shapes` into a single `Shape`, grouping dimensions by type.
    If dimensions with equal names are present in multiple shapes, their types and sizes must match.

    The shorthand `shape1 & shape2` merges shapes with `check_exact=[spatial]`.

    See Also:
        `concat_shapes()`.

    Args:
        *shapes: `Shape` objects to combine.
        check_exact: Sequence of type filters, such as `channel`, `batch`, `spatial` or `instance`.
            These types are checked for exact match, i.e. shapes must either contain all dimensions of that type or none.
            The order of the dimensions does not matter.
            For example, when checking `spatial`, the shapes `spatial(x=5)` and `spatial(y=4)` cannot be combined.
            However, `spatial(x=5, y=4)` can be combined with `spatial(y=4, x=5)` and `channel('vector')`.
        order: Dimension type order as `tuple` of type filters (`channel`, `batch`, `spatial` or `instance`). Dimensions are grouped by type while merging.

    Returns:
        Merged `Shape`
    """
    if not shapes:
        return EMPTY_SHAPE
    merged = []
    for dim_type in order:
        check_type_exact = dim_type in check_exact
        group = dim_type(shapes[0])
        for shape in shapes[1:]:
            shape = dim_type(shape)
            if check_type_exact:
                if group.rank == 0:
                    group = shape
                elif shape.rank > 0:  # check exact match
                    if shape.rank != group.rank:
                        raise IncompatibleShapes(f"Failed to combine {shapes} because a different number of {dim_type.__name__} dimensions are present but exact checks are enabled for dimensions of type {dim_type.__name__}. Try declaring all spatial dimensions in one call. Types are {[s.types for s in shapes]}", *shapes)
                    elif set(shape.names) != set(group.names):
                        raise IncompatibleShapes(f"Failed to combine {shapes} because {dim_type.__name__} dimensions do not match but exact checks were enabled for dimensions of type {dim_type.__name__}. Try declaring all spatial dimensions in one call. Types are {[s.types for s in shapes]}", *shapes)
                    elif shape._reorder(group) != group:
                        raise IncompatibleShapes(f"Failed to combine {shapes} because {dim_type.__name__} dimensions do not match but exact checks were enabled for dimensions of type {dim_type.__name__}. Try declaring all spatial dimensions in one call. Types are {[s.types for s in shapes]}", *shapes)
            else:
                for dim in shape:
                    if dim not in group:
                        group = group._expand(dim, pos=-1)
                    else:  # check size match
                        if not _size_equal(dim.size, group.get_size(dim.name)):
                            raise IncompatibleShapes(f"Cannot merge shapes {shapes} because dimension '{dim.name}' exists with different sizes.", *shapes)
        merged.append(group)
    return concat_shapes(*merged)


def _size_equal(s1, s2):
    if isinstance(s1, int):
        return isinstance(s2, int) and s2 == s1
    else:
        return math.close(s1, s2)


def concat_shapes(*shapes: Shape):
    """
    Creates a `Shape` listing the dimensions of all `shapes` in the given order.

    See Also:
        `merge_shapes()`.

    Args:
        *shapes: Shapes to concatenate. No two shapes must contain a dimension with the same name.

    Returns:
        Combined `Shape`.
    """
    names = sum([s.names for s in shapes], ())
    if len(set(names)) != len(names):
        raise IncompatibleShapes(f"Cannot concatenate shapes {list(shapes)}. Duplicate dimension names are not allowed.")
    sizes = sum([s.sizes for s in shapes], ())
    types = sum([s.types for s in shapes], ())
    return Shape(sizes, names, types)


def shape_stack(stack_dim: Shape, *shapes: Shape):
    """ Returns the shape of a tensor created by stacking tensors with `shapes`. """
    names = list(shapes[0].names)
    types = list(shapes[0].types)
    for other in shapes[1:]:
        for size, name, type in other._dimensions:
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
        if all([math.close(s, dim_sizes[0]) for s in dim_sizes[1:]]):
            dim_sizes = dim_sizes[0]
        else:
            from ._ops import stack
            from ._tensors import wrap
            dim_sizes = [wrap(d) for d in dim_sizes]
            dim_sizes = stack(dim_sizes, stack_dim)
        sizes.append(dim_sizes)
    return Shape(sizes, names, types)._expand(stack_dim.with_sizes([len(shapes)]))


def vector_add(*shapes: Shape):
    if not shapes:
        return EMPTY_SHAPE
    names = list(shapes[0].names)
    types = list(shapes[0].types)
    for shape in shapes[1:]:
        for name in shape.names:
            if name not in names:
                names.append(name)
                types.append(shape.get_type(name))
    sizes = [sum(sh.get_size(dim) if dim in sh else 0 for sh in shapes) for dim in names]
    return Shape(sizes, names, types)

