from phi import math
from ._geom import Geometry


def concat(geometries: tuple or list,
                   dim: str,
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
        characteristics = [g.__characteristics__() for g in geometries]
        if sizes is not None:
            characteristics = [{key: math.expand(val, dim, size) for key, val in c.items()}
                               for c, size in zip(characteristics, sizes)]
        new_attributes = {}
        for key in characteristics[0].keys():
            concatenated = math.concat([c[key] for c in characteristics], dim)
            new_attributes[key] = concatenated
        return geometries[0].__with__(**new_attributes)
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
