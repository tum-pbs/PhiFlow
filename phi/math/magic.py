"""
Magic methods allow custom classes to be compatible with various functions defined in `phi.math`, analogous to how implementing `__hash__` allows objects to be used with `hash()`.
The magic methods are grouped into purely declarative classes (interfaces) by what functionality they provide.

* `Shaped` objects have a `phi.math.Shape`.
* `Sliceable` objects can be sliced along dimensions.
* `Shapable` objects can additionally be reshaped.
* `PhiTreeNode` objects can be disassembled into tensors.

All of these magic classes declared here define a custom instance checks and should not be used as superclasses.

An object implements one of the types defined here by implementing one or more of the related magic methods.
Instance checks can be performed via `isinstance(obj, <MagicClass>)`.

This is analogous to interfaces defined in the built-in `collections` package, such as `Sized, Iterable, Hashable, Callable`.
To check whether `len(obj)` can be performed, you check `isinstance(obj, Sized)`.
"""
from typing import Tuple, Dict, Any
from ._shape import Shape


class _ShapedType(type):
    def __instancecheck__(self, instance):
        from ._shape import Shape
        return hasattr(instance, '__shape__') or (hasattr(instance, 'shape') and isinstance(instance.shape, Shape))


class Shaped(metaclass=_ShapedType):
    """
    To be considered shaped, an object must either implement the magic method `__shape__()` or have a valid `shape` property.
    In either case, the returned shape must be an instance of `phi.math.Shape`.

    To check whether an object is `Shaped`, use `isinstance(obj, Shaped)`.

    **Usage in `phi.math`:**

    The functions `phi.math.shape` as well as dimension filters, such as `phi.math.spatial` or `phi.math.non_batch` can be called on all shaped objects.

    See Also:
        `Sliceable`, `Shapable`
    """

    def __shape__(self) -> 'Shape':
        """
        Returns the shape of this object.

        Alternatively, the shape can be declared via the property `shape`.

        Returns:
            `phi.math.Shape`
        """
        raise NotImplementedError

    @property
    def shape(self) -> 'Shape':
        """
        Alternative form of `__shape__()`.
        Implement either to be considered `Shaped`.

        Returns:
            `phi.math.Shape`
        """
        raise NotImplementedError


class _SliceableType(type):
    def __instancecheck__(self, instance):
        return hasattr(instance, '__getitem__')


class Sliceable(metaclass=_SliceableType):
    """
    Objects are considered sliceable if they implement `__getitem__` as defined below.

    To enable the slicing syntax `obj.dim[slice]`, implement the `__getattr__` method as defined below.

    **Usage in `phi.math`:**

    In addition to slicing, sliceable objects can be unstacked along one or multiple dimensions using `phi.math.unstack`.

    See Also
        `Shapable`, `Shaped`
    """

    def __getitem__(self, item: Dict[str, Any]) -> 'Sliceable':
        """
        Slice this object along one or multiple existing or non-existing dimensions.

        Args:
            item: `dict` mapping dimension names to the corresponding selections.
                Selections can be slices, indices, tuples, item names, bool tensors, int tensors or other custom types.

        Returns:
            Instance of the same class (or a compatible class) as `self`.
        """
        raise NotImplementedError


class _ShapableType(type):
    def __instancecheck__(self, instance):
        return isinstance(instance, Sliceable) and isinstance(instance, Shaped) and\
               (hasattr(instance, '__stack__') or (hasattr(instance, '__concat__') and hasattr(instance, '__expand__')))


class Shapable(metaclass=_ShapableType):
    """
    Shapable objects can be stacked, concatenated and reshaped.

    To be considered `Shapable`, objects must be `Sliceable` and `Shaped` and implement

    * `__stack__()` or
    * `__concat__()` and `__expand__()`.

    Objects should additionally implement the other magic methods for performance reasons.

    **Usage in `phi.math`:**

    Shapable objects can be used with the following functions in addition to what they inherit from being `Sliceable` and `Shaped`:

    * `phi.math.stack`
    * `phi.math.concat`
    * `phi.math.rename_dims`
    * `phi.math.pack_dims`
    * `phi.math.unpack_dims`
    * `phi.math.expand`

    Additionally, the `phi.math.BoundDim` syntax for dimension renaming and retyping is enabled, e.g. `obj.dim.as_channel('vector')`.
    """

    def __stack__(self, values: tuple, dim: Shape, **kwargs) -> 'Shapable':
        """
        Stack all `values` into a single instance along the new dimension `dim`.

        Args:
            values: `tuple` of `Shapable` objects to be stacked. `self` is included in that list at least once.
            dim: Single-dimension `Shape`. This dimension must not be present with any of the `values`.
                The dimension fulfills the condition `dim.size == len(values)`.
            **kwargs: Additional keyword arguments required by specific implementations.
                Adding spatial dimensions to fields requires the `bounds: Box` argument specifying the physical extent of the new dimensions.

        Returns:
            New instance of `Shapable` representing the stacked slices.
            Its shape includes `dim` in addition to the dimensions present in `values`.
            If such a representation cannot be created because some values in `values` are not supported, returns `NotImplemented`.
        """
        raise NotImplementedError

    def __concat__(self, values: tuple, dim: str, **kwargs) -> 'Shapable':
        """
        Concatenate `values` along `dim`.

        Args:
            values: Values to concatenate. `self` is included in that list at least once.
            dim: Dimension nams as `str`, must be present in all `values`.
            **kwargs: Additional keyword arguments required by specific implementations.
                Adding spatial dimensions to fields requires the `bounds: Box` argument specifying the physical extent of the new dimensions.

        Returns:
            New instance of `Shapable` representing the concatenated values.
            The size of `dim` must be equal to the sum of all `dim` sizes in `values`.
            If such a representation cannot be created because some values in `values` are not supported, returns `NotImplemented`.
        """
        raise NotImplementedError

    def __expand__(self, dims: Shape, **kwargs) -> 'Shapable':
        """
        Adds new dimensions to this object.
        The value of this object is constant along the new dimensions.

        Args:
            dims: Dimensions to add.
                They are guaranteed to not already be present in `shape(self)`.
            **kwargs: Additional keyword arguments required by specific implementations.
                Adding spatial dimensions to fields requires the `bounds: Box` argument specifying the physical extent of the new dimensions.

        Returns:
            New instance of `Shapable`.
        """
        raise NotImplementedError

    def __replace_dims__(self, dims: Tuple[str, ...], new_dims: Shape, **kwargs) -> 'Shapable':
        """
        Exchange existing dimensions.
        This can be used to rename dimensions, change dimension types or change item names.

        Args:
            dims: Dimensions to be replaced.
            new_dims: Replacement dimensions as `Shape` with `rank == len(dims)`.
            **kwargs: Additional keyword arguments required by specific implementations.
                Adding spatial dimensions to fields requires the `bounds: Box` argument specifying the physical extent of the new dimensions.

        Returns:
            New instance of `Shapable`.
        """
        raise NotImplementedError

    def __pack_dims__(self, dims: Tuple[str, ...], packed_dim: Shape, pos: int or None = None, **kwargs) -> 'Shapable':
        """
        Compresses multiple dimensions into a single dimension by concatenating the elements.
        Elements along the new dimensions are laid out according to the order of `dims`.

        The type of the new dimension will be equal to the types of `dims`.
        If `dims` have varying types, the new dimension will be a batch dimension.

        Args:
            dims: Dimensions to be compressed in the specified order.
            packed_dim: Single-dimension `Shape`.
            pos: Index of new dimension. `None` for automatic, `-1` for last, `0` for first.
            **kwargs: Additional keyword arguments required by specific implementations.
                Adding spatial dimensions to fields requires the `bounds: Box` argument specifying the physical extent of the new dimensions.

        Returns:
            New instance of `Shapable`.
        """
        raise NotImplementedError

    def __unpack_dims__(self, dim: str, unpacked_dims: Shape, **kwargs) -> 'Shapable':
        """
        Decompresses a tensor dimension by unstacking the elements along it.
        The compressed dimension `dim` is assumed to contain elements laid out according to the order of `unpacked_dims`.

        Args:
            dim: Dimension to be decompressed.
            unpacked_dims: `Shape`: Ordered dimensions to replace `dim`, fulfilling `unpacked_dims.volume == shape(self)[dim].rank`.
            **kwargs: Additional keyword arguments required by specific implementations.
                Adding spatial dimensions to fields requires the `bounds: Box` argument specifying the physical extent of the new dimensions.

        Returns:
            New instance of `Shapable`.
        """
        raise NotImplementedError



class _PhiTreeNodeType(type):

    def __instancecheck__(self, instance):
        from ._tensors import Tensor, MISSING_TENSOR, NATIVE_TENSOR, Dict
        if isinstance(instance, Tensor):
            return True
        if instance is MISSING_TENSOR or instance is NATIVE_TENSOR:
            return True
        if instance is None or isinstance(instance, Tensor):
            return True
        elif isinstance(instance, (tuple, list)):
            return all(isinstance(item, PhiTreeNode) for item in instance)
        elif isinstance(instance, Dict):
            return True
        elif isinstance(instance, dict):
            return all(isinstance(name, str) for name in instance.keys()) and all(isinstance(val, PhiTreeNode) for val in instance.values())
        else:
            return hasattr(instance, '__variable_attrs__') or hasattr(instance, '__value_attrs__')


class PhiTreeNode(metaclass=_PhiTreeNodeType):
    """
    Φ-tree nodes can be iterated over and disassembled or flattened into elementary objects, such as tensors.
    `phi.math.Tensor` instances as well as PyTree nodes (`tuple`, `list`, `dict` with `str` keys) are Φ-tree nodes.

    For custom classes to be considered Φ-tree nodes, they have to implement one of the following magic methods:

    * `__variable_attrs__()`
    * `__value_attrs__()`

    Additionally, Φ-tree nodes must override `__eq__()` to allow comparison of data-stripped (key) instances.

    To check whether an object is a Φ-tree node, use `isinstance(obj, PhiTreeNode)`.

    **Usage in `phi.math`:**

    Φ-tree nodes can be used as keys, for example in `jit_compile()`.
    They are converted to keys by stripping all variable tensors and replacing them by a placeholder object.
    In key mode, `__eq__()` compares all non-variable properties that might invalidate a trace when changed.

    Disassembly and assembly of Φ-tree nodes uses `phi.math.copy_with` which will call `__with_attrs__` if implemented.
    """

    def __value_attrs__(self) -> Tuple[str]:
        """
        Returns all `Tensor` or `PhiTreeNode` attribute names of `self` that should be transformed by single-operand math operations,
        such as `sin()`, `exp()`.

        Returns:
            `tuple` of `str` attributes.
                Calling `getattr(self, attr)` must return a `Tensor` or `PhiTreeNode` for all returned attributes.
        """
        raise NotImplementedError

    def __variable_attrs__(self) -> Tuple[str]:
        """
        Returns all `Tensor` or `PhiTreeNode` attribute names of `self` whose values are variable.
        Variables denote values that can change from one function call to the next or for which gradients can be recorded.
        If this method is not implemented, all attributes returned by `__value_attrs__()` are considered variable.

        The returned properties are used by the following functions:

        - `jit_compile()`
        - `jit_compile_linear()`
        - `stop_gradient()`
        - `jacobian()`
        - `custom_gradient()`

        Returns:
            `tuple` of `str` attributes.
                Calling `getattr(self, attr)` must return a `Tensor` or `PhiTreeNode` for all returned attributes.
        """
        raise NotImplementedError

    def __with_attrs__(self, **attrs):
        """
        Used by `phi.math.copy_with`.
        Create a copy of this object which has the `Tensor` or `PhiTreeNode` attributes contained in `attrs` replaced.
        If this method is not implemented, tensor attributes are replaced using `setattr()`.

        Args:
            **attrs: `dict` mapping `str` attribute names to `Tensor` or `PhiTreeNode`.

        Returns:
            Altered copy of `self`
        """
        raise NotImplementedError

    def __eq__(self, other):
        raise NotImplementedError



class BoundDim:
    """
    Represents a dimension of a sliceable object to make slicing, renaming and retyping prettier.
    Any instance of `BoundDim` is bound to the sliceable object and is immutable.
    All operations upon the dim affect return a copy of the sliceable object.

    `BoundDim` objects are generally created by and for objects that are both `Sliceable` and `Shaped`.
    These objects should declare the following method to support the `.dim` syntax:

    ```python
    from phi.math.magic import BoundDim

    class MyClass:

        def __getattr__(self, name: str) -> BoundDim:
            return BoundDim(self, name)
    ```

    **Usage**

    * `obj.dim.size` return the dimension size.
    * `obj.dim.item_names` return the dimension item names.
    * `obj.dim.exists` checks whether a dimension is listed in the shape of the bound object.
    * `obj.dim[0]` picks the first element along `dim`. The shape of the result will not contain `dim`.
    * `obj.dim[1:-1]` discards the first and last element along `dim`.
    * `obj.dim.rename('new_name')` renames `dim` to `new_name`.
    * `obj.dim.as_channel()` changes the type of `dim` to *channel*.
    * `obj.dim.unstack()` un-stacks the bound value along `dim`.
    * `for slice in obj.dim` loops over all slices of `dim`.
    """

    def __init__(self, obj, name: str):
        """
        Args:
            obj: Bound object, must be `Sliceable` and `Shaped`.
            name: Dimension name as `str`.
        """
        if name.startswith('_') or ',' in name or ' ' in name:
            raise AttributeError(f"'{type(self)}' object has no attribute '{name}'")
        assert isinstance(obj, Sliceable) and isinstance(obj, Shaped)
        self.obj = obj
        self.name = name

    @property
    def exists(self):
        """ Whether the dimension is listed in the `Shape` of the object. """
        return self.name in self.obj.shape

    def __repr__(self):
        if self.name not in self.obj.shape:
            return f"{type(self.obj).__name__}.{self.name} (non-existent)"
        items = self.item_names
        if items is not None:
            if len(items) <= 4:
                size_repr = ",".join(items)
            else:
                size_repr = f"{self.size}:{items[0]}..{items[-1]}"
        else:
            size_repr = self.size
        from ._shape import TYPE_ABBR
        return f"{type(self.obj).__name__}.{self.name}{TYPE_ABBR.get(self.dim_type, '?')}={size_repr}"

    @property
    def size(self):
        """ Length of this dimension as listed in the `Shape` of the bound object. """
        return self.obj.shape.get_size(self.name) if self.exists else None

    @property
    def dim_type(self):
        return self.obj.shape.get_type(self.name)

    @property
    def _dim_type(self):
        return self.obj.shape.get_type(self.name)

    @property
    def is_batch(self):
        """ Whether the type of this dimension as listed in the `Shape` is *batch*. Only defined for existing dimensions. """
        from ._shape import BATCH_DIM
        return self._dim_type == BATCH_DIM

    @property
    def is_spatial(self):
        """ Whether the type of this dimension as listed in the `Shape` is *spatial*. Only defined for existing dimensions. """
        from ._shape import SPATIAL_DIM
        return self._dim_type == SPATIAL_DIM

    @property
    def is_instance(self):
        """ Whether the type of this dimension as listed in the `Shape` is *instance*. Only defined for existing dimensions. """
        from ._shape import INSTANCE_DIM
        return self._dim_type == INSTANCE_DIM


    @property
    def is_channel(self):
        """ Whether the type of this dimension as listed in the `Shape` is *channel*. Only defined for existing dimensions. """
        from ._shape import CHANNEL_DIM
        return self._dim_type == CHANNEL_DIM

    @property
    def item_names(self):
        return self.obj.shape.get_item_names(self.name)

    @property
    def index(self):
        """ The index of this dimension in the `Shape` of the bound object. """
        return self.obj.shape.index(self.name)

    def __int__(self):
        return self.index

    def __getitem__(self, item):
        return self.obj[{self.name: item}]

    def unstack(self, size: int or None = None) -> tuple:
        """
        Lists the slices along this dimension as a `tuple`.

        Args:
            size: (optional) If given as `int`, this dimension can be unstacked even if it is not present on the object.
                In that case, `size` copies of the object are returned.

        Returns:
            `tuple` of `Sliceable`
        """
        from ._ops import unstack
        if size is None:
            return unstack(self.obj, self.name)
        else:
            if self.exists:
                unstacked = unstack(self.obj, self.name)
                assert len(unstacked) == size, f"Size of dimension {self.name} does not match {size}."
                return unstacked
            else:
                return (self.obj,) * size

    def __iter__(self):
        """ Iterate over slices along this dim """
        if self.exists:
            return iter(self.unstack())
        else:
            return iter([self.obj])

    def __call__(self, *args, **kwargs):
        raise TypeError(f"Method {type(self.obj).__name__}.{self.name}() does not exist.")


__pdoc__ = {}  # Show all magic functions in pdoc3
for cls_name, cls in dict(globals()).items():
    if isinstance(cls, type) and type(cls) != type and not cls_name.startswith('_'):
        for magic_function in dir(cls):
            if magic_function.startswith('__') and magic_function.endswith('__') and not hasattr(object, magic_function) and magic_function != '__weakref__':
                __pdoc__[f'{cls_name}.{magic_function}'] = True
