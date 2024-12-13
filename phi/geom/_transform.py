from typing import Optional, Sequence

from phiml import math
from phiml.math import Tensor, channel, rename_dims, wrap, shape, normalize, dual, stack, length, arange, to_float, Shape

from ._geom import Geometry, GeometricType
from ._functions import cross


def scale(obj: GeometricType, scale: float | Tensor, pivot: Tensor = None, dim='vector') -> GeometricType:
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


def rotate(obj: GeometricType, rot: float | Tensor | None, invert=False, pivot: Tensor | str = 'bounds') -> GeometricType:
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
        center = pivot + rotate(obj.center - pivot, rot)
        return obj.rotated(rot).at(center)
    elif isinstance(obj, Tensor):
        assert 'vector' in obj.shape, f"vector must have exactly a channel dimension named 'vector'"
        matrix = rotation_matrix(rot)
        if invert:
            matrix = rename_dims(matrix, '~vector,vector', matrix.shape['vector'] + matrix.shape['~vector'])
        assert matrix.vector.dual.size == obj.vector.size, f"Rotation matrix from {rot.shape} is {matrix.vector.dual.size}D but vector {obj.shape} is {obj.vector.size}D."
        return math.dot(matrix, '~vector', obj, 'vector')


def rotation_matrix(x: float | math.Tensor | None, matrix_dim=channel('vector'), none_to_unit=False) -> Optional[Tensor]:
    """
    Create a 2D or 3D rotation matrix from the corresponding angle(s).

    Args:
        x:
            2D: scalar angle
            3D: Either vector pointing along the rotation axis with rotation angle as length or Euler angles.
            Euler angles need to be laid out along a `angle` channel dimension with dimension names listing the spatial dimensions.
            E.g. a 90° rotation about the z-axis is represented by `vec('angles', x=0, y=0, z=PI/2)`.
            If a rotation matrix is passed for `angle`, it is returned without modification.
        matrix_dim: Matrix dimension for 2D rotations. In 3D, the channel dimension of angle is used.

    Returns:
        Matrix containing `matrix_dim` in primal and dual form as well as all non-channel dimensions of `x`.
    """
    if x is None and not none_to_unit:
        return None
    elif x is None:
        return to_float(arange(matrix_dim) == arange(matrix_dim.as_dual()))
    if isinstance(x, Tensor) and x.dtype == object:  # possibly None in matrices
        return math.map(rotation_matrix, x, dims=object, matrix_dim=matrix_dim, none_to_unit=none_to_unit)
    if isinstance(x, Tensor) and '~vector' in x.shape and 'vector' in x.shape.channel and x.shape.get_size('~vector') == x.shape.get_size('vector'):
        return x  # already a rotation matrix
    elif 'angle' in shape(x) and shape(x).get_size('angle') == 3:  # 3D Euler angles
        assert channel(x).rank == 1 and channel(x).size == 3, f"x for 3D rotations needs to be a 3-vector but got {x}"
        s1, s2, s3 = math.sin(x).angle  # x, y, z
        c1, c2, c3 = math.cos(x).angle
        matrix_dim = matrix_dim.with_size(shape(x).get_item_names('angle'))
        return wrap([[c3 * c2, c3 * s2 * s1 - s3 * c1, c3 * s2 * c1 + s3 * s1],
                     [s3 * c2, s3 * s2 * s1 + c3 * c1, s3 * s2 * c1 - c3 * s1],
                     [-s2, c2 * s1, c2 * c1]], matrix_dim, matrix_dim.as_dual())  # Rz * Ry * Rx  (1. rotate about X by first angle)
    elif 'vector' in shape(x) and shape(x).get_size('vector') == 3:  # 3D axis + x
        angle = length(x)
        s, c = math.sin(angle), math.cos(angle)
        t = 1 - c
        k1, k2, k3 = normalize(x, epsilon=1e-12).vector
        matrix_dim = matrix_dim.with_size(shape(x).get_item_names('vector'))
        return wrap([[c + k1**2 * t, k1 * k2 * t - k3 * s, k1 * k3 * t + k2 * s],
                     [k2 * k1 * t + k3 * s, c + k2**2 * t, k2 * k3 * t - k1 * s],
                     [k3 * k1 * t - k2 * s, k3 * k2 * t + k1 * s, c + k3**2 * t]], matrix_dim, matrix_dim.as_dual())
    else:  # 2D rotation
        sin = wrap(math.sin(x))
        cos = wrap(math.cos(x))
        return wrap([[cos, -sin], [sin, cos]], matrix_dim, matrix_dim.as_dual())


def rotation_angles(rot: Tensor):
    """
    Compute the scalar x in 2D or the Euler angles in 3D from a given rotation matrix.
    This function returns one valid solution but often, there are multiple solutions.

    Args:
        rot: Rotation matrix as created by `phi.math.rotation_matrix()`.
            Must have exactly one channel and one dual dimension with equally-ordered elements.

    Returns:
        Scalar x in 2D, Euler angles
    """
    assert channel(rot).rank == 1 and dual(rot).rank == 1, f"Rotation matrix must have one channel and one dual dimension but got {rot.shape}"
    if channel(rot).size == 2:
        cos = rot[{channel: 0, dual: 0}]
        sin = rot[{channel: 1, dual: 0}]
        return math.arctan(sin, divide_by=cos)
    elif channel(rot).size == 3:
        a2 = -math.arcsin(rot[{channel: 2, dual: 0}])  # ToDo handle [2, 0] == 1 (i.e. cos_theta == 0)
        cos2 = math.cos(a2)
        a1 = math.arctan(rot[{channel: 2, dual: 1}] / cos2, divide_by=rot[{channel: 2, dual: 2}] / cos2)
        a3 = math.arctan(rot[{channel: 1, dual: 0}] / cos2, divide_by=rot[{channel: 0, dual: 0}] / cos2)
        regular_sol = stack([a1, a2, a3], channel(angle=channel(rot).item_names[0]))
        # --- pole case cos(theta) == 1 ---
        a3_pole = 0  # unconstrained
        bottom_pole = rot[{channel: 2, dual: 0}] < 0
        a2_pole = math.where(bottom_pole, 1.57079632679, -1.57079632679)
        a1_pole = math.where(bottom_pole, math.arctan(rot[{channel: 0, dual: 1}], divide_by=rot[{channel: 0, dual: 2}]), math.arctan(-rot[{channel: 0, dual: 1}], divide_by=-rot[{channel: 0, dual: 2}]))
        pole_sol = stack([a1_pole, a2_pole, a3_pole], channel(regular_sol))
        return math.where(abs(rot[{channel: 2, dual: 0}]) >= 1, pole_sol, regular_sol)
    else:
        raise ValueError(f"")


def rotation_matrix_from_directions(source_dir: Tensor, target_dir: Tensor, vec_dim: str = 'vector', epsilon=None) -> Tensor:
    """
    Computes a rotation matrix A, such that `target_dir = A @ source_dir`

    Args:
        source_dir: Two or three-dimensional vector. `Tensor` with channel dim called 'vector'.
        target_dir: Two or three-dimensional vector. `Tensor` with channel dim called 'vector'.

    Returns:
        Rotation matrix as `Tensor` with 'vector' dim and its dual counterpart.
    """
    if source_dir.vector.size == 3:
        axis, angle = axis_angle_from_directions(source_dir, target_dir, vec_dim, epsilon=epsilon)
        return rotation_matrix_from_axis_and_angle(axis, angle, is_axis_normalized=False, epsilon=epsilon)
    raise NotImplementedError


def axis_angle_from_directions(source_dir: Tensor, target_dir: Tensor, vec_dim: str = 'vector', epsilon=None) -> tuple[Tensor, Tensor]:
    if source_dir.vector.size == 3:
        source_dir = normalize(source_dir, vec_dim, epsilon=epsilon)
        target_dir = normalize(target_dir, vec_dim, epsilon=epsilon)
        axis = cross(source_dir, target_dir)
        lim = 1-epsilon if epsilon is not None else 1
        angle = math.arccos(math.clip(source_dir.vector @ target_dir.vector, -lim, lim))
        return axis, angle
    raise NotImplementedError


def rotation_matrix_from_axis_and_angle(axis: Tensor, angle: float | Tensor, vec_dim='vector', is_axis_normalized=False, epsilon=1e-5) -> Tensor:
    """
    Computes a rotation matrix that rotates by `angle` around `axis`.

    Args:
        axis: 3D vector. `Tensor` with channel dim called 'vector'.
        angle: Rotation angle.
        is_axis_normalized: Whether `axis` has length 1.
        epsilon: Minimum axis length. For shorter axes, the unit matrix is returned.

    Returns:
        Rotation matrix as `Tensor` with 'vector' dim and its dual counterpart.
    """
    if axis.vector.size == 3:  # Rodrigues' rotation formula
        axis = normalize(axis, vec_dim, epsilon=epsilon, allow_zero=False) if not is_axis_normalized else axis
        kx, ky, kz = axis.vector
        s = math.sin(angle)
        c = 1 - math.cos(angle)
        return wrap([
            (1 - c*(ky*ky+kz*kz),    -kz*s + c*(kx*ky),     ky*s + c*(kx*kz)),
            (   kz*s + c*(kx*ky),  1 - c*(kx*kx+kz*kz),     -kx*s + c*(ky * kz)),
            (  -ky*s + c*(kx*kz),    kx*s + c*(ky * kz),  1 - c*(kx*kx+ky*ky)),
        ], axis.shape['vector'], axis.shape['vector'].as_dual())
    raise NotImplementedError


def matching_rotations(*matrices: Optional[Tensor]) -> Sequence[Tensor]:
    """
    Replaces `None` rotations with unit matrices if any of `matrices` is not `None`.
    """
    if all(m is None for m in matrices) or all(m is not None for m in matrices):
        return matrices
    some = [m for m in matrices if m is not None][0]
    unit_matrix = arange(some.shape.primal) == arange(some.shape.dual)
    return [unit_matrix if m is None else m for m in matrices]
