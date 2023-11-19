from typing import Sequence, Union

from phiml.math import Tensor, channel, Shape, vec_normalize, vec, sqrt
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
