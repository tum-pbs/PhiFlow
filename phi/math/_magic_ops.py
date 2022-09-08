import copy
import warnings
from numbers import Number
from typing import TypeVar, Tuple

from .backend import choose_backend, NoBackendFound
from .backend._dtype import DType
from ._shape import Shape, DimFilter, batch, instance, shape
from .magic import Sliceable, Shaped, Shapable, PhiTreeNode


class MagicNotImplemented(Exception): pass


def unstack(value, dim: DimFilter):
    """
    Un-stacks a `Sliceable` along one or multiple dimensions.

    If multiple dimensions are given, the order of elements will be according to the dimension order in `dim`, i.e. elements along the last dimension will be neighbors in the returned `tuple`.

    Args:
        value: `Tensor` to unstack.
        dim: Dimensions as `Shape` or comma-separated `str` or dimension type, i.e. `channel`, `spatial`, `instance`, `batch`.

    Returns:
        `tuple` of `Tensor` objects.
    """
    assert isinstance(value, Sliceable) and isinstance(value, Shaped)
    dims = shape(value).only(dim)
    assert dims.rank > 0, "unstack() requires at least one dimension"
    if dims.rank == 1:
        if hasattr(value, '__unstack__'):
            result = value.__unstack__(dims.names)
            if result is not NotImplemented:
                assert isinstance(result, tuple), f"__unstack__ must return a tuple but got {type(result)}"
                assert all([isinstance(item, Sliceable) for item in result]), f"__unstack__ must return a tuple of Sliceable objects but not all items were sliceable in {result}"
                return result
        return tuple([value[{dims.name: i}] for i in range(dims.size)])
    else:  # multiple dimensions
        if hasattr(value, '__pack_dims__'):
            packed_dim = batch('_unstack')
            value = value.__pack_dims__(dims.names, packed_dim, pos=None)
            if value is not NotImplemented:
                return unstack(value, packed_dim)
        first_unstacked = unstack(value, dims[0])
        inner_unstacked = [unstack(v, dims.without(dims[0])) for v in first_unstacked]
        return sum(inner_unstacked, ())


def stack(values: tuple or list or dict, dim: Shape, **kwargs):
    """
    Stacks `values` along the new dimension `dim`.

    Stacking tensors is performed lazily, i.e. the memory is allocated only when needed.
    This makes repeated stacking and slicing along the same dimension very efficient.

    Args:
        values: Sequence of `Shapable` objects to be stacked.
            If a `dict`, keys must be of type `str` and are used as item names along `dim`.
        dim: `Shape` with a least one dimension. None of these dimensions can be present with any of the `values`.
            If `dim` is a single-dimension shape, its size is determined from `len(values)` and can be left undefined (`None`).
            If `dim` is a multi-dimension shape, its volume must be equal to `len(values)`.
        **kwargs: Additional keyword arguments required by specific implementations.
            Adding spatial dimensions to fields requires the `bounds: Box` argument specifying the physical extent of the new dimensions.
            Adding batch dimensions must always work without keyword arguments.

    Returns:
        `Tensor` containing `values` stacked along `dim`.
    """
    assert len(values) > 0, f"stack() got empty sequence {values}"
    assert isinstance(dim, Shape)
    if dim.rank == 1:
        assert dim.size == len(values) or dim.size is None, f"stack dim size must match len(values) or be undefined but got {dim} for {len(values)} values"
        if dim.size is None:
            dim = dim.with_size(len(values))
        if isinstance(values, dict):
            dim_item_names = tuple(values.keys())
            values = tuple(values.values())
            dim = dim._with_item_names((dim_item_names,))
        for v in values[1:]:
            assert set(shape(v).names) == set(shape(values[0]).names), f"Stacked values must have the same dimensions but got {shape(values[0])} and {shape(v)}"
        # if any value implements Shapable, use their implementation
        for v in values:
            if hasattr(v, '__stack__'):
                result = v.__stack__(values, dim, **kwargs)
                if result is not NotImplemented:
                    assert isinstance(result, Shapable), "__stack__ must return a Shapable object"
                    return result
        # Fallback: use expand and concat
        for v in values:
            if not hasattr(v, '__stack__') and hasattr(v, '__concat__') and hasattr(v, '__expand__'):
                exp_values = tuple([expand(v, dim.with_size(1), **kwargs) for v in values])
                if len(exp_values) > 8:
                    warnings.warn(f"stack() default implementation is slow on large dimensions ({dim.name}={len(exp_values)}). Please implement __stack__()", RuntimeWarning, stacklevel=2)
                result = v.__concat__(exp_values, dim.name, **kwargs)
                if result is not NotImplemented:
                    assert isinstance(result, Shapable), "__concat__ must return a Shapable object"
                    return result
        # else maybe all values are native scalars
        from ._tensors import wrap
        try:
            values = tuple([wrap(v) for v in values])
        except ValueError:
            raise MagicNotImplemented(f"At least one item in values must be Shapable but got types {[type(v) for v in values]}")
        return values[0].__stack__(values, dim, **kwargs)
    else:
        assert dim.volume == len(values), f"When passing multiple stack dims, their volume must equal len(values) but got {dim} for {len(values)} values"
        if isinstance(values, dict):
            warnings.warn(f"When stacking a dict along multiple dimensions, the key names are discarded. Got keys {tuple(values.keys())}", RuntimeWarning, stacklevel=2)
            values = tuple(values.values())
        # if any value implements Shapable, use stack and unpack_dim
        for v in values:
            if hasattr(v, '__stack__') and hasattr(v, '__unpack_dim__'):
                stack_dim = batch('_stack')
                stacked = v.__stack__(values, stack_dim, **kwargs)
                if stacked is not NotImplemented:
                    assert isinstance(stacked, Shapable), "__stack__ must return a Shapable object"
                    assert hasattr(stacked, '__unpack_dim__'), "If a value supports __unpack_dim__, the result of __stack__ must also support it."
                    reshaped = stacked.__unpack_dim__(stack_dim.name, dim, **kwargs)
                    if kwargs is NotImplemented:
                        warnings.warn("__unpack_dim__ is overridden but returned NotImplemented during multi-dimensional stack. This results in unnecessary stack operations.", RuntimeWarning, stacklevel=2)
                    else:
                        return reshaped
        # Fallback: multi-level stack
        for dim_ in reversed(dim):
            values = [stack(values[i:i + dim_.size], dim_, **kwargs) for i in range(0, len(values), dim_.size)]
        return values[0]


def concat(values: tuple or list, dim: str or Shape, **kwargs):
    """
    Concatenates a sequence of tensors along one dimension.
    The shapes of all values must be equal, except for the size of the concat dimension.

    Args:
        values: Tensors to concatenate
        dim: Concatenation dimension, must be present in all `values`.
            The size along `dim` is determined from `values` and can be set to undefined (`None`).
        **kwargs: Additional keyword arguments required by specific implementations.
            Adding spatial dimensions to fields requires the `bounds: Box` argument specifying the physical extent of the new dimensions.
            Adding batch dimensions must always work without keyword arguments.

    Returns:
        Concatenated `Tensor`
    """
    assert len(values) > 0, f"concat() got empty sequence {values}"
    if isinstance(dim, Shape):
        dim = dim.name
    assert isinstance(dim, str), f"dim must be a str or Shape but got '{dim}' of type {type(dim)}"
    for v in values[1:]:
        assert set(shape(v).names) == set(shape(values[0]).names), f"Concatenated values must have the same dimensions but got {values[0].shape} and {v.shape}"
    # First try __concat__
    for v in values:
        if isinstance(v, Shapable):
            if hasattr(v, '__concat__'):
                result = v.__concat__(values, dim, **kwargs)
                if result is not NotImplemented:
                    assert isinstance(result, Shapable), "__concat__ must return a Shapable object"
                    return result
    # Fallback: slice and __stack__
    try:
        unstacked = sum([unstack(v, dim) for v in values], ())
    except MagicNotImplemented:
        raise MagicNotImplemented(f"concat: No value implemented __concat__ and not all values were Sliceable along {dim}. values = {[type(v) for v in values]}")
    if len(unstacked) > 8:
        warnings.warn(f"concat() default implementation is slow on large dimensions ({dim}={len(unstacked)}). Please implement __concat__()", RuntimeWarning, stacklevel=2)
    dim = shape(values[0])[dim].with_size(None)
    try:
        return stack(unstacked, dim, **kwargs)
    except MagicNotImplemented:
        raise MagicNotImplemented(f"concat: No value implemented __concat__ and slices could not be stacked. values = {[type(v) for v in values]}")



def expand(value, dims: Shape, **kwargs):
    """
    Adds dimensions to a `Tensor` by implicitly repeating the tensor values along the new dimensions.
    If `value` already contains some of the new dimensions, a size and type check is performed instead.

    This function replaces the usual `tile` / `repeat` functions of
    [NumPy](https://numpy.org/doc/stable/reference/generated/numpy.tile.html),
    [PyTorch](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.repeat),
    [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/tile) and
    [Jax](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.tile.html).

    Additionally, it replaces the traditional `unsqueeze` / `expand_dims` functions.

    Args:
        value: `Tensor`
        dims: Dimensions to be added as `Shape`
        **kwargs: Additional keyword arguments required by specific implementations.
            Adding spatial dimensions to fields requires the `bounds: Box` argument specifying the physical extent of the new dimensions.
            Adding batch dimensions must always work without keyword arguments.

    Returns:
        Expanded `Shapable`.
    """
    if hasattr(value, '__expand__'):
        result = value.__expand__(dims, **kwargs)
        if result is not NotImplemented:
            return result
    # Fallback: stack
    if hasattr(value, '__stack__'):
        if dims.volume > 8:
            warnings.warn(f"expand() default implementation is slow on large shapes {dims}. Please implement __expand__()", RuntimeWarning, stacklevel=2)
        for dim in reversed(dims):
            value = stack((value,) * dim.size, dim, **kwargs)
            assert value is not NotImplemented, "Value must implement either __expand__ or __stack__"
        return value
    try:  # value may be a native scalar
        from ._ops import expand_tensor
        from ._tensors import wrap
        value = wrap(value)
    except ValueError:
        raise AssertionError(f"Cannot expand non-shapable object {type(value)}")
    return expand_tensor(value, dims)


def rename_dims(value,
                dims: str or tuple or list or Shape,
                names: str or tuple or list or Shape,
                **kwargs):
    """
    Change the name and optionally the type of some dimensions of `value`.

    Args:
        value: `Shape` or `Tensor` or `Shapable`.
        dims: Existing dimensions of `value`.
        names: Either

            * Sequence of names matching `dims` as `tuple`, `list` or `str`. This replaces only the dimension names but leaves the types untouched.
            * `Shape` matching `dims` to replace names and types.

        **kwargs: Additional keyword arguments required by specific implementations.
            Adding spatial dimensions to fields requires the `bounds: Box` argument specifying the physical extent of the new dimensions.
            Adding batch dimensions must always work without keyword arguments.

    Returns:
        Same type as `value`.
    """
    if isinstance(value, Shape):
        return value._replace_names_and_types(dims, names)
    assert isinstance(value, Shapable) and isinstance(value, Shaped), "value must be a Shape or Shapable."
    dims = shape(value).only(dims)
    names = dims._replace_names_and_types(dims, names)
    if hasattr(value, '__replace_dims__'):
        result = value.__replace_dims__(dims.names, names, **kwargs)
        if result is not NotImplemented:
            return result
    # Fallback: unstack and stack
    if shape(value).only(dims).volume > 8:
        warnings.warn(f"rename_dims() default implementation is slow on large dimensions ({shape(value).only(dims)}). Please implement __replace_dims__()", RuntimeWarning, stacklevel=2)
    for old_name, new_dim in zip(dims.names, names):
        value = stack(unstack(value, old_name), new_dim, **kwargs)
    return value


def pack_dims(value, dims: DimFilter, packed_dim: Shape, pos: int or None = None, **kwargs):
    """
    Compresses multiple dimensions into a single dimension by concatenating the elements.
    Elements along the new dimensions are laid out according to the order of `dims`.
    If the order of `dims` differs from the current dimension order, the tensor is transposed accordingly.
    This function replaces the traditional `reshape` for these cases.

    The type of the new dimension will be equal to the types of `dims`.
    If `dims` have varying types, the new dimension will be a batch dimension.

    If none of `dims` exist on `value`, `packed_dim` will be added only if it is given with a definite size.

    See Also:
        `unpack_dim()`

    Args:
        value: Tensor containing the dimensions `dims`.
        dims: Dimensions to be compressed in the specified order.
        packed_dim: Single-dimension `Shape`.
        pos: Index of new dimension. `None` for automatic, `-1` for last, `0` for first.
        **kwargs: Additional keyword arguments required by specific implementations.
            Adding spatial dimensions to fields requires the `bounds: Box` argument specifying the physical extent of the new dimensions.
            Adding batch dimensions must always work without keyword arguments.

    Returns:
        `Tensor` with compressed shape.
    """
    assert isinstance(value, Shapable) and isinstance(value, Sliceable) and isinstance(value, Shaped), f"value must be Shapable but got {type(value)}"
    dims = shape(value).only(dims)
    if packed_dim in shape(value):
        assert packed_dim in dims, f"Cannot pack dims into new dimension {packed_dim} because it already exists on value {value} and is not packed."
    if len(dims) == 0 or all(dim not in shape(value) for dim in dims):
        return value if packed_dim.size is None else expand(value, packed_dim, **kwargs)  # Inserting size=1 can cause shape errors
    elif len(dims) == 1:
        return rename_dims(value, dims, packed_dim, **kwargs)
    if dims.rank == shape(value).rank and hasattr(value, '__flatten__'):
        result = value.__flatten__(packed_dim, **kwargs)
        if result is not NotImplemented:
            return result
    if hasattr(value, '__pack_dims__'):
        result = value.__pack_dims__(dims.names, packed_dim, pos, **kwargs)
        if result is not NotImplemented:
            return result
    # Fallback: unstack and stack
    if shape(value).only(dims).volume > 8:
        warnings.warn(f"pack_dims() default implementation is slow on large dimensions ({shape(value).only(dims)}). Please implement __pack_dims__()", RuntimeWarning, stacklevel=2)
    return stack(unstack(value, dims), packed_dim, **kwargs)




def unpack_dim(value, dim: str or Shape, unpacked_dims: Shape, **kwargs):
    """
    Decompresses a dimension by unstacking the elements along it.
    This function replaces the traditional `reshape` for these cases.
    The compressed dimension `dim` is assumed to contain elements laid out according to the order of `unpacked_dims`.

    See Also:
        `pack_dims()`

    Args:
        value: `Tensor` for which one dimension should be split.
        dim: Dimension to be decompressed.
        unpacked_dims: `Shape`: Ordered dimensions to replace `dim`, fulfilling `unpacked_dims.volume == shape(self)[dim].rank`.
        **kwargs: Additional keyword arguments required by specific implementations.
            Adding spatial dimensions to fields requires the `bounds: Box` argument specifying the physical extent of the new dimensions.
            Adding batch dimensions must always work without keyword arguments.

    Returns:
        `Tensor` with decompressed shape
    """
    assert isinstance(value, Shapable) and isinstance(value, Sliceable) and isinstance(value, Shaped), f"value must be Shapable but got {type(value)}"
    if isinstance(dim, Shape):
        dim = dim.name
    assert isinstance(dim, str), f"dim must be a str but got {type(dim)}"
    if unpacked_dims.rank == 0:
        return value[{dim: 0}]  # remove dim
    elif unpacked_dims.rank == 1:
        return rename_dims(value, dim, unpacked_dims, **kwargs)
    if hasattr(value, '__unpack_dim__'):
        result = value.__unpack_dim__(dim, unpacked_dims, **kwargs)
        if result is not NotImplemented:
            return result
    # Fallback: unstack and stack
    if shape(value).only(dim).volume > 8:
        warnings.warn(f"pack_dims() default implementation is slow on large dimensions ({shape(value).only(dim)}). Please implement __unpack_dim__()", RuntimeWarning, stacklevel=2)
    unstacked = unstack(value, dim)
    for dim in reversed(unpacked_dims):
        unstacked = [stack(unstacked[i:i+dim.size], dim, **kwargs) for i in range(0, len(unstacked), dim.size)]
    return unstacked[0]



def flatten(value, flat_dim: Shape = instance('flat'), **kwargs):
    """
    Returns a `Tensor` with the same values as `value` but only a single dimension `flat_dim`.
    The order of the values in memory is not changed.

    Args:
        value: `Tensor`
        flat_dim: Dimension name and type as `Shape` object. The size is ignored.
        **kwargs: Additional keyword arguments required by specific implementations.
            Adding spatial dimensions to fields requires the `bounds: Box` argument specifying the physical extent of the new dimensions.
            Adding batch dimensions must always work without keyword arguments.

    Returns:
        `Tensor`
    """
    assert isinstance(flat_dim, Shape) and flat_dim.rank == 1, flat_dim
    assert isinstance(value, Shapable) and isinstance(value, Shaped), f"value must be Shapable but got {type(value)}"
    if hasattr(value, '__flatten__'):
        result = value.__flatten__(flat_dim, **kwargs)
        if result is not NotImplemented:
            return result
    # Fallback: pack_dims
    return pack_dims(value, shape(value), flat_dim, **kwargs)


# PhiTreeNode

PhiTreeNodeType = TypeVar('PhiTreeNodeType')  # Defined in phi.math.magic: tuple, list, dict, custom


def variable_attributes(obj) -> Tuple[str]:
    if hasattr(obj, '__variable_attrs__'):
        return obj.__variable_attrs__()
    elif hasattr(obj, '__value_attrs__'):
        return obj.__value_attrs__()
    else:
        raise ValueError(f"Not PhiTreeNode: {type(obj)}")


def value_attributes(obj):
    assert hasattr(obj, '__value_attrs__'), f"{type(obj)} must implement '__value_attrs__()' to be used with value functions."
    return obj.__value_attrs__()


def variable_values(obj):
    assert hasattr(obj, '__value_attrs__'), f"{type(obj)} must implement '__value_attrs__()' to be used with value functions."
    if hasattr(obj, '__variable_attrs__'):
        values = obj.__value_attrs__()
        variables = obj.__variable_attrs__()
        return [a for a in values if a in variables]
    else:
        return obj.__value_attrs__()


def copy_with(obj: PhiTreeNodeType, **updates) -> PhiTreeNodeType:
    """
    Creates a copy of the given `PhiTreeNode` with updated values as specified in `updates`.

    If `obj` overrides `__with_attrs__`, the copy will be created via that specific implementation.
    Otherwise, the `copy` module and `setattr` will be used.

    Args:
        obj: `PhiTreeNode`
        **updates: Values to be replaced.

    Returns:
        Copy of `obj` with updated values.
    """
    if hasattr(obj, '__with_attrs__'):
        return obj.__with_attrs__(**updates)
    elif isinstance(obj, (Number, bool)):
        return obj
    else:
        cpy = copy.copy(obj)
        for attr, value in updates.items():
            setattr(cpy, attr, value)
        return cpy


# Other Ops

OtherMagicType = TypeVar('OtherMagicType')


def cast(x: OtherMagicType, dtype: DType or type) -> OtherMagicType:
    """
    Casts `x` to a different data type.

    Implementations:

    * NumPy: [`x.astype()`](numpy.ndarray.astype)
    * PyTorch: [`x.to()`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.to)
    * TensorFlow: [`tf.cast`](https://www.tensorflow.org/api_docs/python/tf/cast)
    * Jax: [`jax.numpy.array`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.array.html)

    See Also:
        `to_float`, `to_int32`, `to_int64`, `to_complex`.

    Args:
        x: `Tensor`
        dtype: New data type as `phi.math.DType`, e.g. `DType(int, 16)`.

    Returns:
        `Tensor` with data type `dtype`
    """
    if not isinstance(dtype, DType):
        dtype = DType.as_dtype(dtype)
    if hasattr(x, '__cast__'):
        return x.__cast__(dtype)
    elif isinstance(x, (Number, bool)):
        return dtype.kind(x)
    elif isinstance(x, PhiTreeNode):
        attrs = {key: getattr(x, key) for key in value_attributes(x)}
        new_attrs = {k: cast(v, dtype) for k, v in attrs.items()}
        return copy_with(x, **new_attrs)
    try:
        backend = choose_backend(x)
        return backend.cast(x, dtype)
    except NoBackendFound:
        if dtype.kind == bool:
            return bool(x)
        raise ValueError(f"Cannot cast object of type '{type(x).__name__}'")
