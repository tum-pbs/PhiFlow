import warnings
from typing import Tuple, Optional

from phiml import math
from phiml.math import Tensor, stack, instance, wrap

from ._geom import Geometry


def line_trace(geo: Geometry, origin: Tensor, direction: Tensor, side='both', tolerance=None, max_iter=64, step_size=.9, max_line_length=None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Optional[Tensor]]:
    """
    Trace a line until it hits the surface of `geo`.
    The surface can be hit either from the outside or the inside.

    Args:
        geo: `Geometry` that implements `approximate_closest_surface`.
        origin: Line start location.
        direction: Unit vector pointing in the line direction.
        side: 'outside' or 'inside' or 'both'.
        tolerance: Surface distance tolerance.
        max_iter: Maximum number of steps per line.
        step_size: Step size factor. This can be set to `1` if the signed distance values of `geo` are exact.
            For inexact SDFs, smaller step sizes prevent skipping over surfaces.

    Returns:
        hit: Whether a surface intersection was found for the line.
        distance: Distance between the line and the surface.
        position: Hit location or point until which the line was traced.
        normal: Surface normal at hit location
        hit_index: Geometry face index at hit location
    """
    assert side in ['outside', 'inside', 'both'], f"{side} is not a valid side"
    if tolerance is None:
        tolerance = 1e-4 * geo.bounding_box().size.min
    walked = 0
    has_hit = False
    initial_sdf = None
    last_sdf = None
    has_crossed = wrap(False)
    for i in range(max_iter):
        sgn_dist, delta, normal, _, face_index = geo.approximate_closest_surface(origin + walked * direction)
        initial_sdf = sgn_dist if initial_sdf is None else initial_sdf
        normal_dot_direction = normal.vector @ direction.vector
        intersection = (normal.vector @ delta.vector) / normal_dot_direction
        intersection = math.where(math.is_nan(intersection), math.INF, intersection)
        if side == 'both':
            can_hit_surface = True
        elif side == 'outside':
            can_hit_surface = normal_dot_direction <= 0
        else:
            can_hit_surface = normal_dot_direction >= 0
        intersection = math.where(intersection < math.where(can_hit_surface, -tolerance, 0), math.INF, intersection)  # surface behind us
        if last_sdf is not None:
            if side == 'both':
                has_crossed = (sgn_dist * last_sdf < 0) | (abs(sgn_dist) <= tolerance)
            elif side == 'outside':
                has_crossed = (last_sdf > tolerance) & (sgn_dist <= tolerance)
                has_crossed |= (last_sdf > 0) & (sgn_dist <= 0)
            else:
                has_crossed = (last_sdf < -tolerance) & (sgn_dist >= -tolerance)
                has_crossed |= (last_sdf < 0) & (sgn_dist >= 0)
        has_hit |= has_crossed
        max_walk = math.minimum(abs(sgn_dist), intersection)
        max_walk = math.where(can_hit_surface, step_size * max_walk, max_walk + tolerance)  # jump over surface if we can't hit it
        max_walk = math.where(has_hit, 0, max_walk)
        walked += max_walk
        is_done = has_hit.all if max_line_length is None else (has_hit | walked > max_line_length).all
        if is_done:
            break
        last_sdf = sgn_dist
        # trj.append(walked)
        # if i == 15:
        #     from phi.vis import show
        #     trj = stack(trj, instance('trj'))
        #     show(geo, trj, overlay='args')
    else:
        warnings.warn(f"thickness reached maximum iterations {max_iter}", RuntimeWarning, stacklevel=2)
    return has_hit, walked, origin + walked * direction, normal, face_index
