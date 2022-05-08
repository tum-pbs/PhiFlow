from phi import math
from . import GridCell
from ._geom import Geometry
from ..math import Shape
from ..math._shape import parse_dim_names, parse_dim_order
from ..math._tensors import variable_attributes, copy_with


def pack_dims(value: Geometry,
              dims: Shape or tuple or list or str,
              packed_dim: Shape,
              pos: int or None = None) -> Geometry:
    """
    Reshapes dimensions of a geometry.

    See Also:
        `phi.math.pack_dims()`.

    Args:
        value: `Geometry` containing `dims`
        dims: Dimensions to pack into a single dimension.
        packed_dim: The new dimension.
        pos: (Optional) Position of the new dimension.

    Returns:
        `Geometry` with `packed_dim` instead of `dims`.
    """
    if isinstance(value, GridCell):
        value = value.corner_representation()
    dims = parse_dim_order(dims)
    new_attrs = {}
    for a in variable_attributes(value):
        v = getattr(value, a)
        if all(d in v.shape for d in dims):
            new_attrs[a] = math.pack_dims(v, dims, packed_dim, pos=pos)
        else:
            assert all(d not in v.shape for d in dims), f"Cannot pack_dims for attribute {a} of {value}"
    return copy_with(value, **new_attrs)


def concat(geometries: tuple or list,
           dim: Shape,
           sizes: tuple or list or None = None):
    """
    Concatenates multiple geometries of the same type.

    Args:
        geometries: sequence of `phi.geom.Geometry` objects of the same type
        sizes: implicit
        dim: dimension to concatenate

    Returns:
        New `phi.geom.Geometry` object
    """
    if all(isinstance(g, type(geometries[0])) for g in geometries):
        characteristics = [{a: getattr(g, a) for a in variable_attributes(g)} for g in geometries]
        new_attributes = {}
        for c in characteristics[0].keys():
            if any([item[c].shape.volume > 1 for item in characteristics]) or any([not math.close(item[c], characteristics[0][c]) for item in characteristics]):
                for item, size in zip(characteristics, sizes):
                    item[c] = math.expand(item[c], dim.with_size(size))
                concatenated = math.concat([item[c] for item in characteristics], dim)
                new_attributes[c] = concatenated
        return copy_with(geometries[0], **new_attributes)
    else:
        raise NotImplementedError()


def invert(geometry: Geometry):
    """
    Swaps inside and outside.

    Args:
        geometry: `phi.geom.Geometry` to swap

    Returns:
        New `phi.geom.Geometry` object with same surface but swapped normals
    """
    return ~geometry
