from phi import math
from ._geom import Geometry
from ..math import Shape
from ..math._tensors import variable_attributes, copy_with


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
        if sizes is not None:
            characteristics = [{key: math.expand(val, dim) for key, val in c.items()}
                               for c, size in zip(characteristics, sizes)]
        new_attributes = {}
        for key in characteristics[0].keys():
            concatenated = math.concat([c[key] for c in characteristics], dim)
            new_attributes[key] = concatenated
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
