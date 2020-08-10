from phi.struct import Trait
from phi.backend.dynamic_backend import DYNAMIC_BACKEND as math


class Batched(Trait):
    """
Structs with this trait can tag items with the keyword 'min_rank', representing the number of innate inner dimensions of tensor values of that item.
All further, outer dimensions are assumed to be batch dimensions.

Example:
    @struct.definition(traits=[math.BATCHED])\n
    class MyStruct(struct.Struct):
        @struct.constant(min_rank=1)\n
        def batched_constant(self, c):
            assert math.ndims(c) >= 1  # Will always be fulfilled\n
            return c

For each item with the 'min_rank' keyword, (1) the specified minimum tensor rank is ensured by expanding the dimensions if necessary and (2) batch shape checks are performed during validation.
Also, all additional dimensions, called `batch dimensions` are collected and cross-checked between all items.
The batch dimensions of a valid batched struct can be accessed as `struct.batch_shape`, and `struct.batch_rank` holds the corresponding length.

When a struct with inconsistent batch dimensions is validated, a `ShapeMismatch` error is raised, typically upon struct creation.

The Batched trait also ensures that all values are converted to tensors before the validation function is called.
    """

    def check_argument(self, struct_class, item, keyword, value):
        assert keyword == 'min_rank'
        assert isinstance(value, int) or callable(value), value

    def endow(self, struct):
        struct.batch_shape = None
        struct.batch_rank = None

    def pre_validate_struct(self, struct):
        struct.batch_shape = None
        struct.batch_rank = None

    def pre_validated(self, struct, item, value):
        tensor = math.as_tensor(value)
        min_rank = item.trait_kwargs['min_rank']
        if callable(min_rank):
            min_rank = min_rank(struct)
        shape = math.staticshape(value)
        if len(shape) < min_rank:
            tensor = math.expand_dims(tensor, axis=0, number=min_rank - len(shape))
            shape = math.staticshape(value)
        batch_shape = shape[:-min_rank if min_rank != 0 else None]
        if struct.batch_shape is None:
            struct.batch_shape = batch_shape
        else:
            struct.batch_shape = _combined_shape(batch_shape, struct.batch_shape, item, struct)
        struct.batch_rank = len(struct.batch_shape)
        return tensor


BATCHED = Batched(keywords=['min_rank'])


def _combined_shape(shape1, shape2, prop, obj):
    rank = max(len(shape1), len(shape2))
    resulting_shape = []
    for i in range(1, rank + 1):
        dim1 = shape1[-i] if len(shape1) >= i else 1
        dim2 = shape2[-i] if len(shape2) >= i else 1
        try:
            resulting_shape.append(_combined_dim(dim1, dim2))
        except AssertionError:
            raise ShapeMismatch("Batch dimension %d with value %d of '%s' of %s does not match other properties with value %d. Occured during comparison of batch shapes %s and %s" % (-i, dim1, prop, obj, dim2, shape1, shape2))
    return tuple(resulting_shape[::-1])


def _combined_dim(dim1, dim2):
    if dim1 is None or dim2 is None:
        return None
    if dim1 == 1:
        return dim2
    if dim2 == 1:
        return dim1
    assert dim1 == dim2
    return dim1


class ShapeMismatch(ValueError):
    """
Raised when a shape check fails, i.e. when tensors that require compatible shapes do not match.
It is a subclass of `ValueError` because ValueErrors are often raised in this case.
    """

    def __init__(self, *args):
        ValueError.__init__(self, *args)
