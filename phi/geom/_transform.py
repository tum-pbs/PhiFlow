from typing import Union

from phiml.math import Tensor

from phiml.math import Tensor
from ._functions import rotate_vector
from ._geom import Geometry, GeometricType


def scale(obj: GeometricType, scale: Union[float, Tensor], pivot: Tensor = None, dim='vector') -> GeometricType:
    """
    Scale a `Geometry` or vector `Tensor` about a pivot point.

    Args:
        obj: `Geometry` to scale.
        scale: Scaling factor.
        pivot: Point that stays fixed under the scaling operation. Defaults to the bounding box center.

    Returns:
        Rotated `Geometry`
    """
    if scale is None:
        return obj
    if isinstance(obj, Geometry):
        if pivot is None:
            pivot = obj.bounding_box().center
        center = pivot + scale * (obj.center - pivot)
        return obj.scaled(scale).at(center)
    elif isinstance(obj, Tensor):
        assert 'vector' in obj.shape, f"vector must have exactly a channel dimension named 'vector'"
        if pivot is None:
            return obj * scale
        raise NotImplementedError
    raise ValueError(obj)


def rotate(obj: GeometricType, rot: Union[float, Tensor, None], invert=False, pivot: Union[Tensor, str] = 'bounds') -> GeometricType:
    """
    Rotate a vector or `Geometry` about the `pivot`.

    Args:
        obj: n-dimensional vector `Tensor` or `Geometry`.
        rot: Euler angle(s) or rotation matrix.
            `None` is interpreted as no rotation.
        invert: Whether to apply the inverse rotation.
        pivot: Either a point (`Tensor`) lying on the rotation axis or one of the following strings: 'bounds', 'individual'.
            Vector tensors are rotated about the origin if `pivot` is not given as a `Tensor`.

    Returns:
        Rotated vector as `Tensor`
    """
    if rot is None:
        return obj
    if isinstance(obj, Geometry):
        if pivot is None:
            pivot = obj.bounding_box().center
        center = pivot + rotate(obj.center - pivot, rot, invert=invert)
        if invert:
            raise NotImplementedError
        return obj.rotated(rot).at(center)
    elif isinstance(obj, Tensor):
        if isinstance(pivot, Tensor):
            return pivot + rotate_vector(obj - pivot, rot, invert=invert)
        else:
            return rotate_vector(obj, rot, invert=invert)
