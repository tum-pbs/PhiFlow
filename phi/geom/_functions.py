from typing import Sequence, Union, Optional, Tuple

from phiml import math
from phiml.math import Tensor, channel, Shape, normalize, vec, sqrt, maximum, clip, vec_squared, norm, where, stack, dual, argmin, safe_div, arange, wrap, to_float, rename_dims
from phiml.math._shape import parse_dim_order, DimFilter, shape


# No dependence on Geometry


def vec_normalize(x, epsilon=None, allow_infinite=False, allow_zero=False):
    return normalize(x, 'vector', epsilon, allow_infinite=allow_infinite, allow_zero=allow_zero)


def vec_length(x, eps=None):
    return norm(x, 'vector', eps)


def rotate_vector(x, rot, invert=False):
    assert 'vector' in x.shape, f"vector must have exactly a channel dimension named 'vector'"
    if rot is None:
        return x
    matrix = rotation_matrix(rot, matrix_dim=x.shape['vector'])
    if invert:
        matrix = rename_dims(matrix, '~vector,vector', matrix.shape['vector'] + matrix.shape['~vector'])
    assert matrix.vector.dual.size == x.vector.size, f"Rotation matrix from {rot.shape} is {matrix.vector.dual.size}D but vector {x.shape} is {x.vector.size}D."
    return math.dot(matrix, '~vector', x, 'vector')


def cross(vec1: Tensor, vec2: Tensor) -> Tensor:
    """
    Computes the cross product of two vectors in 2D.

    Args:
        vec1: `Tensor` with a single channel dimension called `'vector'`
        vec2: `Tensor` with a single channel dimension called `'vector'`

    Returns:
        `Tensor`
    """
    vec1 = math.tensor(vec1)
    vec2 = math.tensor(vec2)
    spatial_rank = vec1.vector.size if 'vector' in vec1.shape else vec2.vector.size
    if spatial_rank == 2:  # Curl in 2D
        assert 'vector' in vec2.shape
        if 'vector' in vec1.shape:
            v1_x, v1_y = vec1.vector
            v2_x, v2_y = vec2.vector
            return v1_x * v2_y - v1_y * v2_x
        else:
            v2_x, v2_y = vec2.vector
            return vec1 * stack([-v2_y, v2_x], channel(vec2))
    elif spatial_rank == 3:  # Curl in 3D
        assert 'vector' in vec1.shape and 'vector' in vec2.shape, f"Both vectors must have a 'vector' dimension but got shapes {vec1.shape}, {vec2.shape}"
        v1_x, v1_y, v1_z = vec1.vector
        v2_x, v2_y, v2_z = vec2.vector
        return math.stack([
            v1_y * v2_z - v1_z * v2_y,
            v1_z * v2_x - v1_x * v2_z,
            v1_x * v2_y - v1_y * v2_x,
        ], vec1.shape['vector'])
    else:
        raise AssertionError(f'dims = {spatial_rank}. Vector product not available in > 3 dimensions')


def clip_length(vec: Tensor, min_len=0, max_len=1, vec_dim: DimFilter = 'vector', eps: Union[float, Tensor] = 1e-5):
    """
    Clips the length of a vector to the interval `[min_len, max_len]` while keeping the direction.
    Zero-vectors remain zero-vectors.

    Args:
        vec: `Tensor`
        min_len: Lower clipping threshold.
        max_len: Upper clipping threshold.
        vec_dim: Dimensions to compute the length over. By default, all channel dimensions are used to compute the vector length.
        eps: Minimum vector length. Use to avoid `inf` gradients for zero-length vectors.

    Returns:
        `Tensor` with same shape as `vec`.
    """
    le = math.length(vec, vec_dim, eps)
    new_length = clip(le, min_len, max_len)
    return vec * safe_div(new_length, le)


def normal_from_slope(slope: Tensor, space: Union[str, Shape, Sequence[str]]):
    """
    Computes the normal vector of a line, plane, or hyperplane.

    Args:
        slope: Line Slope (2D), plane slope (3D) or hyperplane slope (4+D).
            Must have one channel dimension listing the vector components.
            The vector must list all but one dimensions of `space`.
        space: Ordered spatial dimensions as comma-separated string, sequence of names or `Shape`

    Returns:
        Normal vector with the channel dimension of `slope` listing all dimensions of `space` in that order.
    """
    assert channel(slope).rank == 1 and all(d in space for d in channel(slope).item_names[0]), f"slope must have a single channel dim listing all but one component of the space {space} but got {slope.shape}"
    space = parse_dim_order(space)
    assert len(space) > 1, f"space must contain at least 2 dimensions"
    up = set(space) - set(channel(slope).item_names[0])
    assert len(up) == 1, f"space must have exactly one more dimension than slope but got slope {channel(slope)} for space {space}"
    up = next(iter(up))
    normal = vec(channel(slope).name, **{d: 1 if d == up else -slope[d] for d in space})
    return vec_normalize(normal, allow_infinite=True)


def y_intersect_2d(slope_y, per_x, x, y):
    """

    Args:
        slope_y: Y component of the slope
        per_x: Slope = slope_y / per_x. This may be used to handle infinite slopes.
        x: x component of any point on the line.
        y: y component of same point on the line.

    Returns:

    """
    m = slope_y / per_x  # we may want to handle per_x == 0 in the future
    b = y - m * x
    dist_from_xy = sqrt(x ** 2 * (1 + m**2))
    return b, dist_from_xy


def plane_sgn_dist(plane_offset: Tensor, plane_normal: Tensor, point: Tensor):
    """
    Args:
        plane_offset: Either any point on the plane or the plane's signed distance from origin.
        plane_normal: Normal vector of plane. This vector is assumed to be normalized.
        point: Query point.

    Returns:
        Signed distance from plane to point.
    """
    if 'vector' in plane_offset.shape:
        plane_offset = plane_offset.vector @ plane_normal.vector
    return plane_normal.vector @ point.vector - plane_offset


def closest_on_triangle(A: Tensor, B: Tensor, C: Tensor, query: Tensor, exact_edges=True) -> Tensor:
    """
    Computes the point inside the triangle spanned by `A,B,C` closest to `query`.

    Args:
        A: One corner of the triangle(s).
        B: Second corner of the triangle(s).
        C: Third corner of the triangle(s).
        query: Query point.
        exact_edges: If `True` computes the exact closest point when the projection of `query` lies outside the triangle.
            If `False`, approximates the closest point in a faster way but may give inaccurate results.
            Points that project inside the triangle are always accurate.

    Returns:
        `Tensor`
    """
    v0 = B - A
    v1 = C - A
    v2 = query - A
    dot00 = v0.vector @ v0.vector
    dot01 = v0.vector @ v1.vector
    dot02 = v0.vector @ v2.vector
    dot11 = v1.vector @ v1.vector
    dot12 = v1.vector @ v2.vector
    denom = dot00 * dot11 - dot01 * dot01  # assume != 0, i.e. triangle is not degenerate (area > 0)
    u = (dot11 * dot02 - dot01 * dot12) / denom
    v = (dot00 * dot12 - dot01 * dot02) / denom
    if exact_edges:
        closest_if_inside = A + u * v0 + v * v1
        is_outside = (u < 0) | (v < 0) | (u + v > 1)
        p1 = stack([A, B, C], dual('_tri_points'))
        p2 = stack([B, C, A], dual('_tri_points'))
        closest_on_edges = closest_on_line(p1, p2, query)
        dist = vec_length(query - closest_on_edges)
        closest_on_edge = closest_on_edges[argmin(dist, '~_tri_points')]
        return where(is_outside, closest_on_edge, closest_if_inside)
    else:
        u = clip(u)
        v = clip(v)
        outside = maximum(0, u + v - 1)
        u -= .5 * outside
        v -= .5 * outside
        return A + u * v0 + v * v1


def closest_on_line(A, B, query):
    v = B - A
    u = query - A
    t = u.vector @ v.vector / vec_squared(v)
    t = clip(t, 0, 1)
    return A + t * v


def closest_points_on_lines(p1, v1, p2, v2, eps=1e-10, can_be_parallel=True):
    """Find the closest points between two infinite lines defined by point and direction."""
    n = cross(v1, v2)
    n_norm = vec_normalize(n)
    diff = p2 - p1
    t1 = cross(v2, n_norm).vector @ diff.vector
    t2 = cross(v1, n_norm).vector @ diff.vector
    c1, c2 = p1 + t1 * v1, p2 + t2 * v2
    if can_be_parallel:
        is_parallel = vec_squared(n) < eps
        t = (p2-p1).vector @ v1.vector  # Project p2-p1 onto v1 to get the closest point on line 1
        c1 = where(is_parallel, p1 + t * v1, c1)
        c2 = where(is_parallel, p2, c2)
    return c1, c2


def distance_line_point(line_offset: Tensor, line_direction: Tensor, point: Tensor, is_direction_normalized=False) -> Tensor:
    to_point = point - line_offset
    c = vec_length(cross(to_point, line_direction))
    if not is_direction_normalized:
        c /= vec_length(line_direction)
    return c


def closest_normal_vector(target: Tensor, normal: Tensor, is_normalized=False, eps=1e-10):
    """Finds a vector orthogonal to `normal` that approximately points along `target`."""
    if not is_normalized:
        normal = vec_normalize(normal)
    target_normal = normal * (target.vector @ normal.vector)
    target_ortho = target - target_normal
    return vec_normalize(target_ortho, eps)


def orthogonal_vector(vector: Tensor):
    if vector.vector.size == 3:
        x_not_parallel = (vector.vector[0] != 0)
        ref = where(x_not_parallel, (0, 0, 1), (1, 0, 0))
        return vec_normalize(cross(vector, ref))
    raise NotImplementedError


def rotation_matrix(x: Union[float, math.Tensor, None], matrix_dim=channel('vector'), none_to_unit=False) -> Optional[Tensor]:
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
        angle = vec_length(x)
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


def axis_angle_from_directions(source_dir: Tensor, target_dir: Tensor, vec_dim: str = 'vector', epsilon=None) -> Tuple[Tensor, Tensor]:
    if source_dir.vector.size == 3:
        source_dir = normalize(source_dir, vec_dim, epsilon=epsilon)
        target_dir = normalize(target_dir, vec_dim, epsilon=epsilon)
        axis = cross(source_dir, target_dir)
        lim = 1-epsilon if epsilon is not None else 1
        angle = math.arccos(math.clip(source_dir.vector @ target_dir.vector, -lim, lim))
        return axis, angle
    raise NotImplementedError


def rotation_matrix_from_axis_and_angle(axis: Tensor, angle: Union[float, Tensor], vec_dim='vector', is_axis_normalized=False, epsilon=1e-5) -> Tensor:
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


def sample_helix(offset: Tensor, axis: Tensor, start: Tensor, end: Tensor, t: Tensor):
    start -= offset
    end -= offset
    # --- Component along axis (h) ---
    h_start = start.vector @ axis.vector
    h_end = end.vector @ axis.vector
    h = t * h_end + (1-t) * h_start
    start = start - h_start * axis
    end = end - h_end * axis
    # --- Radius ---
    r_start = math.norm(start, 'vector')
    r_end = math.norm(end, 'vector')
    r = t * r_end + (1-t) * r_start
    # --- Angle ---
    rot_axis, angle = axis_angle_from_directions(start, end)
    a = angle * t
    # --- Combine ---
    return offset + h * axis + r/r_start * rotate_vector(start, rot_axis * a)


def solve2x2(a, b, c, d, y1, y2):
    denom = (a*d - b*c)
    x1 = (d*y1 - b*y2) / denom
    x2 = (a*y2 - c*y1) / denom
    return x1, x2
