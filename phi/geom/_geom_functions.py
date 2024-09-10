import warnings
from typing import Tuple, Optional

from phiml import math
from phiml.math import Tensor

from ._geom import Geometry


def line_trace(geo: Geometry, origin: Tensor, direction: Tensor, max_iter=64, epsilon=None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    """
    Trace a line until it hits the surface of `geo`.
    The surface can be hit either from the outside or the inside.

    Args:
        geo: `Geometry` that implements `approximate_closest_surface`.
        origin: Line start location.
        direction: Unit vector pointing in the line direction.
        max_iter: Maximum number of steps per line.
        epsilon: Surface distance tolerance.

    Returns:
        hit: Whether a surface intersection was found for the line.
        distance: Distance between the line and the surface.
        position: Hit location or point until which the line was traced.
        normal: Surface normal at hit location
        hit_index: Geometry face index at hit location
    """
    if epsilon is None:
        epsilon = 1e-4 * geo.bounding_box().size.min
    walked = 0
    has_hit = False
    initial_sdf = None
    for i in range(max_iter):
        sgn_dist, delta, normal, _, face_index = geo.approximate_closest_surface(origin + walked * direction)
        initial_sdf = sgn_dist if initial_sdf is None else initial_sdf
        has_crossed = sgn_dist * initial_sdf < 0
        intersection = (normal.vector @ delta.vector) / (normal.vector @ direction.vector)
        intersection = math.where(math.is_nan(intersection), math.INF, intersection)
        max_walk = math.minimum(abs(sgn_dist), math.where(intersection < -epsilon, math.INF, intersection))
        max_walk = math.where(has_hit, 0, max_walk)
        walked += max_walk * .9
        has_hit = (abs(sgn_dist) <= epsilon) | has_crossed
        if math.all(has_hit):
            break
    else:
        warnings.warn(f"thickness reached maximum iterations {max_iter}", RuntimeWarning, stacklevel=2)
    return has_hit, walked, origin + walked * direction, normal, face_index
