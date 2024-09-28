from typing import Sequence, Union

from phiml.math import Tensor, channel, Shape, vec_normalize, vec, sqrt, maximum, clip, vec_squared, vec_length, where, stack, dual, argmin
from phiml.math._shape import parse_dim_order

# No dependence on Geometry


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
