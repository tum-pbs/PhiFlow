import copy
import warnings
from numbers import Number
from typing import TypeVar, Tuple, Set

import dataclasses

from . import channel
from .backend import choose_backend, NoBackendFound
from .backend._dtype import DType
from ._shape import Shape, DimFilter, batch, instance, shape, non_batch, merge_shapes, concat_shapes, spatial, parse_dim_order
from .magic import Sliceable, Shaped, Shapable, PhiTreeNode


class MagicNotImplemented(Exception): pass


def unstack(value, dim: DimFilter):
    """
    Un-stacks a `Sliceable` along one or multiple dimensions.

    If multiple dimensions are given, the order of elements will be according to the dimension order in `dim`, i.e. elements along the last dimension will be neighbors in the returned `tuple`.

    Args:
        value: `phi.math.magic.Shapable`, such as `phi.math.Tensor`
        dim: Dimensions as `Shape` or comma-separated `str` or dimension type, i.e. `channel`, `spatial`, `instance`, `batch`.

    Returns:
        `tuple` of `Tensor` objects.

    Examples:
        >>> unstack(expand(0, spatial(x=5)), 'x')
        (0.0, 0.0, 0.0, 0.0, 0.0)
    """
    assert isinstance(value, Sliceable) and isinstance(value, Shaped), f"Cannot unstack {type(value).__name__}. Must be Sliceable and Shaped, see https://tum-pbs.github.io/PhiFlow/phi/math/magic.html"
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
            value_packed = value.__pack_dims__(dims.names, packed_dim, pos=None)
            if value_packed is not NotImplemented:
                return unstack(value_packed, packed_dim)
        first_unstacked = unstack(value, dims[0])
        inner_unstacked = [unstack(v, dims.without(dims[0])) for v in first_unstacked]
        return sum(inner_unstacked, ())


def stack(values: tuple or list or dict, dim: Shape, expand_values=False, **kwargs):
    """
    Stacks `values` along the new dimension `dim`.
    All values must have the same spatial, instance and channel dimensions. If the dimension sizes vary, the resulting tensor will be non-uniform.
    Batch dimensions will be added as needed.

    Stacking tensors is performed lazily, i.e. the memory is allocated only when needed.
    This makes repeated stacking and slicing along the same dimension very efficient, i.e. jit-compiled functions will not perform these operations.

    Args:
        values: Collection of `phi.math.magic.Shapable`, such as `phi.math.Tensor`
            If a `dict`, keys must be of type `str` and are used as item names along `dim`.
        dim: `Shape` with a least one dimension. None of these dimensions can be present with any of the `values`.
            If `dim` is a single-dimension shape, its size is determined from `len(values)` and can be left undefined (`None`).
            If `dim` is a multi-dimension shape, its volume must be equal to `len(values)`.
        expand_values: If `True`, will first add missing dimensions to all values, not just batch dimensions.
            This allows tensors with different dimensions to be stacked.
            The resulting tensor will have all dimensions that are present in `values`.
        **kwargs: Additional keyword arguments required by specific implementations.
            Adding spatial dimensions to fields requires the `bounds: Box` argument specifying the physical extent of the new dimensions.
            Adding batch dimensions must always work without keyword arguments.

    Returns:
        `Tensor` containing `values` stacked along `dim`.

    Examples:
        >>> stack({'x': 0, 'y': 1}, channel('vector'))
        (x=0, y=1)

        >>> stack([math.zeros(batch(b=2)), math.ones(batch(b=2))], channel(c='x,y'))
        (x=0.000, y=1.000); (x=0.000, y=1.000) (bᵇ=2, cᶜ=x,y)

        >>> stack([vec(x=1, y=0), vec(x=2, y=3.)], batch('b'))
        (x=1.000, y=0.000); (x=2.000, y=3.000) (bᵇ=2, vectorᶜ=x,y)
    """
    assert len(values) > 0, f"stack() got empty sequence {values}"
    assert isinstance(dim, Shape)
    values_ = tuple(values.values()) if isinstance(values, dict) else values
    if not expand_values:
        for v in values_[1:]:
            assert set(non_batch(v).names) == set(non_batch(values_[0]).names), f"Stacked values must have the same non-batch dimensions but got {non_batch(values_[0])} and {non_batch(v)}"
    # --- Add missing dimensions ---
    if expand_values:
        all_dims = merge_shapes(*values_)
        if isinstance(values, dict):
            values = {k: expand(v, all_dims.without(shape(v).non_batch)) for k, v in values.items()}
        else:
            values = [expand(v, all_dims.without(shape(v).non_batch)) for v in values]
    else:
        all_batch_dims = merge_shapes(*[batch(v) for v in values_])
        if isinstance(values, dict):
            values = {k: expand(v, all_batch_dims) for k, v in values.items()}
        else:
            values = [expand(v, all_batch_dims) for v in values]
    if dim.rank == 1:
        assert dim.size == len(values) or dim.size is None, f"stack dim size must match len(values) or be undefined but got {dim} for {len(values)} values"
        if dim.size is None:
            dim = dim.with_size(len(values))
        if isinstance(values, dict):
            dim_item_names = tuple(values.keys())
            values = tuple(values.values())
            dim = dim.with_size(dim_item_names)
        # --- First try __stack__ ---
        for v in values:
            if hasattr(v, '__stack__'):
                result = v.__stack__(values, dim, **kwargs)
                if result is not NotImplemented:
                    assert isinstance(result, Shapable), "__stack__ must return a Shapable object"
                    return result
        # --- Next: try stacking attributes for tree nodes ---
        if all(isinstance(v, PhiTreeNode) for v in values):
            attributes = all_attributes(values[0])
            if attributes and all(all_attributes(v) == attributes for v in values):
                new_attrs = {}
                for a in attributes:
                    assert all(dim not in shape(getattr(v, a)) for v in values), f"Cannot stack attribute {a} because one values contains the stack dimension {dim}."
                    a_values = [getattr(v, a) for v in values]
                    if all(v is a_values[0] for v in a_values[1:]):
                        new_attrs[a] = expand(a_values[0], dim, **kwargs)
                    else:
                        new_attrs[a] = stack(a_values, dim, expand_values=expand_values, **kwargs)
                return copy_with(values[0], **new_attrs)
            else:
                warnings.warn(f"Failed to concat values using value attributes because attributes differ among values {values}")
        # --- Fallback: use expand and concat ---
        for v in values:
            if not hasattr(v, '__stack__') and hasattr(v, '__concat__') and hasattr(v, '__expand__'):
                expanded_values = tuple([expand(v, dim.with_size(1 if dim.item_names[0] is None else dim.item_names[0][i]), **kwargs) for i, v in enumerate(values)])
                if len(expanded_values) > 8:
                    warnings.warn(f"stack() default implementation is slow on large dimensions ({dim.name}={len(expanded_values)}). Please implement __stack__()", RuntimeWarning, stacklevel=2)
                result = v.__concat__(expanded_values, dim.name, **kwargs)
                if result is not NotImplemented:
                    assert isinstance(result, Shapable), "__concat__ must return a Shapable object"
                    return result
        # --- else maybe all values are native scalars ---
        from ._tensors import wrap
        try:
            values = tuple([wrap(v) for v in values])
        except ValueError:
            raise MagicNotImplemented(f"At least one item in values must be Shapable but got types {[type(v) for v in values]}")
        return values[0].__stack__(values, dim, **kwargs)
    else:  # multi-dim stack
        assert dim.volume == len(values), f"When passing multiple stack dims, their volume must equal len(values) but got {dim} for {len(values)} values"
        if isinstance(values, dict):
            warnings.warn(f"When stacking a dict along multiple dimensions, the key names are discarded. Got keys {tuple(values.keys())}", RuntimeWarning, stacklevel=2)
            values = tuple(values.values())
        # --- if any value implements Shapable, use stack and unpack_dim ---
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
        # --- Fallback: multi-level stack ---
        for dim_ in reversed(dim):
            values = [stack(values[i:i + dim_.size], dim_, **kwargs) for i in range(0, len(values), dim_.size)]
        return values[0]


def concat(values: tuple or list, dim: str or Shape, **kwargs):
    """
    Concatenates a sequence of `phi.math.magic.Shapable` objects, e.g. `Tensor`, along one dimension.
    All values must have the same spatial, instance and channel dimensions and their sizes must be equal, except for `dim`.
    Batch dimensions will be added as needed.

    Args:
        values: Tuple or list of `phi.math.magic.Shapable`, such as `phi.math.Tensor`
        dim: Concatenation dimension, must be present in all `values`.
            The size along `dim` is determined from `values` and can be set to undefined (`None`).
        **kwargs: Additional keyword arguments required by specific implementations.
            Adding spatial dimensions to fields requires the `bounds: Box` argument specifying the physical extent of the new dimensions.
            Adding batch dimensions must always work without keyword arguments.

    Returns:
        Concatenated `Tensor`

    Examples:
        >>> concat([math.zeros(batch(b=10)), math.ones(batch(b=10))], 'b')
        (bᵇ=20) 0.500 ± 0.500 (0e+00...1e+00)

        >>> concat([vec(x=1, y=0), vec(z=2.)], 'vector')
        (x=1.000, y=0.000, z=2.000) float64
    """
    assert len(values) > 0, f"concat() got empty sequence {values}"
    if isinstance(dim, Shape):
        dim = dim.name
    assert isinstance(dim, str), f"dim must be a str or Shape but got '{dim}' of type {type(dim)}"
    for v in values:
        assert dim in shape(v), f"dim must be present in the shapes of all values bot got value {type(v).__name__} with shape {shape(v)}"
    for v in values[1:]:
        assert set(non_batch(v).names) == set(non_batch(values[0]).names), f"Concatenated values must have the same non-batch dimensions but got {non_batch(values[0])} and {non_batch(v)}"
    # Add missing batch dimensions
    all_batch_dims = merge_shapes(*[batch(v) for v in values])
    values = [expand(v, all_batch_dims) for v in values]
    # --- First try __concat__ ---
    for v in values:
        if isinstance(v, Shapable):
            if hasattr(v, '__concat__'):
                result = v.__concat__(values, dim, **kwargs)
                if result is not NotImplemented:
                    assert isinstance(result, Shapable), f"__concat__ must return a Shapable object but got {type(result).__name__} from {type(v).__name__} {v}"
                    return result
    # --- Next: try concat attributes for tree nodes ---
    if all(isinstance(v, PhiTreeNode) for v in values):
        attributes = all_attributes(values[0])
        if attributes and all(all_attributes(v) == attributes for v in values):
            new_attrs = {}
            for a in attributes:
                common_shape = merge_shapes(*[shape(getattr(v, a)).without(dim) for v in values])
                a_values = [expand(getattr(v, a), common_shape & shape(v).only(dim)) for v in values]  # expand by dim if missing, and dims of others
                new_attrs[a] = concat(a_values, dim, **kwargs)
            return copy_with(values[0], **new_attrs)
        else:
            warnings.warn(f"Failed to concat values using value attributes because attributes differ among values {values}")
    # --- Fallback: slice and stack ---
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


def expand(value, *dims: Shape, **kwargs):
    """
    Adds dimensions to a `Tensor` or tensor-like object by implicitly repeating the tensor values along the new dimensions.
    If `value` already contains any of the new dimensions, a size and type check is performed for these instead.

    This function replaces the usual `tile` / `repeat` functions of
    [NumPy](https://numpy.org/doc/stable/reference/generated/numpy.tile.html),
    [PyTorch](https://pytorch.org/docs/stable/tensors.html#torch.Tensor.repeat),
    [TensorFlow](https://www.tensorflow.org/api_docs/python/tf/tile) and
    [Jax](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.tile.html).

    Additionally, it replaces the traditional `unsqueeze` / `expand_dims` functions.

    Args:
        value: `phi.math.magic.Shapable`, such as `phi.math.Tensor`
            For tree nodes, expands all value attributes by `dims` or the first variable attribute if no value attributes are set.
        *dims: Dimensions to be added as `Shape`
        **kwargs: Additional keyword arguments required by specific implementations.
            Adding spatial dimensions to fields requires the `bounds: Box` argument specifying the physical extent of the new dimensions.
            Adding batch dimensions must always work without keyword arguments.

    Returns:
        Same type as `value`.
    """
    dims = concat_shapes(*dims)
    merge_shapes(value, dims.only(shape(value)))  # check that existing sizes match
    if not dims.without(shape(value)):  # no new dims to add
        if set(dims) == set(shape(value).only(dims)):  # sizes and item names might differ, though
            return value
    # --- First try __stack__
    if hasattr(value, '__expand__'):
        result = value.__expand__(dims, **kwargs)
        if result is not NotImplemented:
            return result
    # --- Next try Tree Node ---
    if isinstance(value, PhiTreeNode):
        attributes = value_attributes(value) if hasattr(value, '__value_attrs__') else [variable_attributes(value)[0]]
        new_attributes = {a: expand(getattr(value, a), dims, **kwargs) for a in attributes}
        return copy_with(value, **new_attributes)
    # --- Fallback: stack ---
    if hasattr(value, '__stack__'):
        if dims.volume > 8:
            warnings.warn(f"expand() default implementation is slow on large shapes {dims}. Please implement __expand__() for {type(value).__name__} as defined in phi.math.magic", RuntimeWarning, stacklevel=2)
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

    Dimensions that are not present on value will be ignored. The corresponding new dimensions given by `names` will not be added.

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
    elif isinstance(value, (Number, bool)):
        return value
    assert isinstance(value, Shapable) and isinstance(value, Shaped), f"value must be a Shape or Shapable but got {type(value).__name__}"
    dims = parse_dim_order(dims)
    if isinstance(names, str):
        names = parse_dim_order(names)
    assert len(dims) == len(names), f"names and dims must be of equal length but got #dims={len(dims)} and #names={len(names)}"
    existing_dims = shape(value).only(dims, reorder=True)
    if not existing_dims:
        return value
    existing_names = [n for i, n in enumerate(names) if dims[i] in existing_dims]
    existing_names = existing_dims._replace_names_and_types(existing_dims, existing_names)
    # --- First try __replace_dims__ ---
    if hasattr(value, '__replace_dims__'):
        result = value.__replace_dims__(existing_dims.names, existing_names, **kwargs)
        if result is not NotImplemented:
            return result
    # --- Next try Tree Node ---
    if isinstance(value, PhiTreeNode):
        new_attributes = {a: rename_dims(getattr(value, a), existing_dims, existing_names, **kwargs) for a in all_attributes(value)}
        return copy_with(value, **new_attributes)
    # --- Fallback: unstack and stack ---
    if shape(value).only(existing_dims).volume > 8:
        warnings.warn(f"rename_dims() default implementation is slow on large dimensions ({shape(value).only(dims)}). Please implement __replace_dims__() for {type(value).__name__} as defined in phi.math.magic", RuntimeWarning, stacklevel=2)
    for old_name, new_dim in zip(existing_dims.names, existing_names):
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

    If none of `dims` exist on `value`, `packed_dim` will be added only if it is given with a definite size and `value` is not a primitive type.

    See Also:
        `unpack_dim()`

    Args:
        value: `phi.math.magic.Shapable`, such as `phi.math.Tensor`.
        dims: Dimensions to be compressed in the specified order.
        packed_dim: Single-dimension `Shape`.
        pos: Index of new dimension. `None` for automatic, `-1` for last, `0` for first.
        **kwargs: Additional keyword arguments required by specific implementations.
            Adding spatial dimensions to fields requires the `bounds: Box` argument specifying the physical extent of the new dimensions.
            Adding batch dimensions must always work without keyword arguments.

    Returns:
        Same type as `value`.

    Examples:
        >>> pack_dims(math.zeros(spatial(x=4, y=3)), spatial, instance('points'))
        (pointsⁱ=12) const 0.0
    """
    if isinstance(value, (Number, bool)):
        return value
    assert isinstance(value, Shapable) and isinstance(value, Sliceable) and isinstance(value, Shaped), f"value must be Shapable but got {type(value)}"
    dims = shape(value).only(dims, reorder=True)
    if packed_dim in shape(value):
        assert packed_dim in dims, f"Cannot pack dims into new dimension {packed_dim} because it already exists on value {value} and is not packed."
    if len(dims) == 0 or all(dim not in shape(value) for dim in dims):
        return value if packed_dim.size is None else expand(value, packed_dim, **kwargs)  # Inserting size=1 can cause shape errors
    elif len(dims) == 1:
        return rename_dims(value, dims, packed_dim, **kwargs)
    # --- First try __pack_dims__ ---
    if hasattr(value, '__pack_dims__'):
        result = value.__pack_dims__(dims.names, packed_dim, pos, **kwargs)
        if result is not NotImplemented:
            return result
    # --- Next try Tree Node ---
    if isinstance(value, PhiTreeNode):
        new_attributes = {a: pack_dims(getattr(value, a), dims, packed_dim, pos=pos, **kwargs) for a in all_attributes(value)}
        return copy_with(value, **new_attributes)
    # --- Fallback: unstack and stack ---
    if shape(value).only(dims).volume > 8:
        warnings.warn(f"pack_dims() default implementation is slow on large dimensions ({shape(value).only(dims)}). Please implement __pack_dims__() for {type(value).__name__} as defined in phi.math.magic", RuntimeWarning, stacklevel=2)
    return stack(unstack(value, dims), packed_dim, **kwargs)




def unpack_dim(value, dim: str or Shape, *unpacked_dims: Shape, **kwargs):
    """
    Decompresses a dimension by unstacking the elements along it.
    This function replaces the traditional `reshape` for these cases.
    The compressed dimension `dim` is assumed to contain elements laid out according to the order of `unpacked_dims`.

    If `dim` does not exist on `value`, this function will return `value` as-is. This includes primitive types.

    See Also:
        `pack_dims()`

    Args:
        value: `phi.math.magic.Shapable`, such as `Tensor`, for which one dimension should be split.
        dim: Dimension to be decompressed.
        *unpacked_dims: Vararg `Shape`, ordered dimensions to replace `dim`, fulfilling `unpacked_dims.volume == shape(self)[dim].rank`.
        **kwargs: Additional keyword arguments required by specific implementations.
            Adding spatial dimensions to fields requires the `bounds: Box` argument specifying the physical extent of the new dimensions.
            Adding batch dimensions must always work without keyword arguments.

    Returns:
        Same type as `value`.

    Examples:
        >>> unpack_dim(math.zeros(instance(points=12)), 'points', spatial(x=4, y=3))
        (xˢ=4, yˢ=3) const 0.0
    """
    if isinstance(value, (Number, bool)):
        return value
    assert isinstance(value, Shapable) and isinstance(value, Sliceable) and isinstance(value, Shaped), f"value must be Shapable but got {type(value)}"
    if isinstance(dim, Shape):
        dim = dim.name
    assert isinstance(dim, str), f"dim must be a str or Shape but got {type(dim)}"
    if dim not in shape(value):
        return value  # Nothing to do, maybe expand?
    unpacked_dims = concat_shapes(*unpacked_dims)
    if unpacked_dims.rank == 0:
        return value[{dim: 0}]  # remove dim
    elif unpacked_dims.rank == 1:
        return rename_dims(value, dim, unpacked_dims, **kwargs)
    # --- First try __unpack_dim__
    if hasattr(value, '__unpack_dim__'):
        result = value.__unpack_dim__(dim, unpacked_dims, **kwargs)
        if result is not NotImplemented:
            return result
    # --- Next try Tree Node ---
    if isinstance(value, PhiTreeNode) and all_attributes(value):
        new_attributes = {a: unpack_dim(getattr(value, a), dim, unpacked_dims, **kwargs) for a in all_attributes(value)}
        return copy_with(value, **new_attributes)
    # --- Fallback: unstack and stack ---
    if shape(value).only(dim).volume > 8:
        warnings.warn(f"pack_dims() default implementation is slow on large dimensions ({shape(value).only(dim)}). Please implement __unpack_dim__() for {type(value).__name__} as defined in phi.math.magic", RuntimeWarning, stacklevel=2)
    unstacked = unstack(value, dim)
    for dim in reversed(unpacked_dims):
        unstacked = [stack(unstacked[i:i+dim.size], dim, **kwargs) for i in range(0, len(unstacked), dim.size)]
    return unstacked[0]


def flatten(value, flat_dim: Shape = instance('flat'), flatten_batch=False, **kwargs):
    """
    Returns a `Tensor` with the same values as `value` but only a single dimension `flat_dim`.
    The order of the values in memory is not changed.

    Args:
        value: `phi.math.magic.Shapable`, such as `Tensor`.
        flat_dim: Dimension name and type as `Shape` object. The size is ignored.
        flatten_batch: Whether to flatten batch dimensions as well.
            If `False`, batch dimensions are kept, only onn-batch dimensions are flattened.
        **kwargs: Additional keyword arguments required by specific implementations.
            Adding spatial dimensions to fields requires the `bounds: Box` argument specifying the physical extent of the new dimensions.
            Adding batch dimensions must always work without keyword arguments.

    Returns:
        Same type as `value`.

    Examples:
        >>> flatten(math.zeros(spatial(x=4, y=3)))
        (flatⁱ=12) const 0.0
    """
    assert isinstance(flat_dim, Shape) and flat_dim.rank == 1, flat_dim
    assert isinstance(value, Shapable) and isinstance(value, Shaped), f"value must be Shapable but got {type(value)}"
    # --- First try __flatten__ ---
    if hasattr(value, '__flatten__'):
        result = value.__flatten__(flat_dim, flatten_batch, **kwargs)
        if result is not NotImplemented:
            return result
    # There is no tree node implementation for flatten because pack_dims is just as fast
    # --- Fallback: pack_dims ---
    return pack_dims(value, shape(value) if flatten_batch else non_batch(value), flat_dim, **kwargs)


# PhiTreeNode

PhiTreeNodeType = TypeVar('PhiTreeNodeType')  # Defined in phi.math.magic: tuple, list, dict, custom


def variable_attributes(obj) -> Tuple[str]:
    if hasattr(obj, '__variable_attrs__'):
        return obj.__variable_attrs__()
    elif hasattr(obj, '__value_attrs__'):
        return obj.__value_attrs__()
    elif dataclasses.is_dataclass(obj):
        return tuple([f.name for f in dataclasses.fields(obj)])
    else:
        raise ValueError(f"Not a PhiTreeNode: {type(obj).__name__}")


def value_attributes(obj) -> Tuple[str, ...]:
    if hasattr(obj, '__value_attrs__'):
        return obj.__value_attrs__()
    if dataclasses.is_dataclass(obj):
        return tuple([f.name for f in dataclasses.fields(obj)])
    raise ValueError(f"{type(obj).__name__} must implement '__value_attrs__()' or be a dataclass to be used with value functions.")


def variable_values(obj) -> Tuple[str, ...]:
    if hasattr(obj, '__variable_attrs__'):
        values = obj.__value_attrs__()
        variables = obj.__variable_attrs__()
        return tuple([a for a in values if a in variables])
    else:
        return obj.__value_attrs__()  # this takes care of dataclasses as well


def all_attributes(obj, assert_any=False) -> Set[str]:
    if not isinstance(obj, PhiTreeNode):
        raise ValueError(f"Not a PhiTreeNode: {type(obj).__name__}")
    result = set()
    if hasattr(obj, '__variable_attrs__'):
        result.update(obj.__variable_attrs__())
    if hasattr(obj, '__value_attrs__'):
        result.update(obj.__value_attrs__())
    if dataclasses.is_dataclass(obj) and not hasattr(obj, '__variable_attrs__') and not hasattr(obj, '__value_attrs__'):
        result.update([f.name for f in dataclasses.fields(obj)])
    if assert_any:
        assert result, f"{type(obj).__name__} is not a valid tree node because it has no tensor-like attributes."
    return result


def replace(obj: PhiTreeNodeType, **updates) -> PhiTreeNodeType:
    """
    Creates a copy of the given `phi.math.magic.PhiTreeNode` with updated values as specified in `updates`.

    If `obj` overrides `__with_attrs__`, the copy will be created via that specific implementation.
    Otherwise, the `copy` module and `setattr` will be used.

    Args:
        obj: `phi.math.magic.PhiTreeNode`
        **updates: Values to be replaced.

    Returns:
        Copy of `obj` with updated values.
    """
    if hasattr(obj, '__with_attrs__'):
        return obj.__with_attrs__(**updates)
    elif isinstance(obj, (Number, bool)):
        return obj
    elif dataclasses.is_dataclass(obj):
        return dataclasses.replace(obj, **updates)
    else:
        cpy = copy.copy(obj)
        for attr, value in updates.items():
            setattr(cpy, attr, value)
        return cpy


copy_with = replace


# Other Ops

MagicType = TypeVar('MagicType')
OtherMagicType = TypeVar('OtherMagicType')


def cast(x: MagicType, dtype: DType or type) -> OtherMagicType:
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


def bool_to_int(x: MagicType, bits=32):
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, Number):
        return x
    if hasattr(x, 'dtype') and isinstance(x.dtype, DType):
        return cast(x, DType(int, bits)) if x.dtype.kind == bool else x
    elif isinstance(x, PhiTreeNode):
        return tree_map(bool_to_int, x, bits=32)
    try:
        backend = choose_backend(x)
        return backend.cast(x, DType(int, bits)) if backend.dtype(x).kind == bool else x
    except NoBackendFound:
        raise ValueError(f"Cannot cast object of type '{type(x).__name__}'")


def tree_map(f, tree, **f_kwargs):
    from ._tensors import Tensor
    if isinstance(tree, Tensor):
        return f(tree, **f_kwargs)
    if isinstance(tree, list):
        return [tree_map(f, e, **f_kwargs) for e in tree]
    elif isinstance(tree, tuple):
        return tuple([tree_map(f, e, **f_kwargs) for e in tree])
    elif isinstance(tree, dict):
        return {k: tree_map(f, e, **f_kwargs) for k, e in tree.items()}
    elif isinstance(tree, PhiTreeNode):
        attrs = {key: getattr(tree, key) for key in value_attributes(tree)}
        new_attrs = {k: tree_map(f, v, **f_kwargs) for k, v in attrs.items()}
        return copy_with(tree, **new_attrs)
    else:
        return f(tree, **f_kwargs)  # try anyway
