import warnings
from typing import Tuple, Optional, Union

from phiml import math
from phiml.math import Tensor, stack, instance, wrap, shape
from . import Cylinder

from ._geom import Geometry


def length(obj: Union[Geometry, Tensor], epsilon=None) -> Tensor:
    """
    Returns the length of a vector `Tensor` or geometric object with a length-like property.

    Args:
        obj: `Tensor` with 'vector' dim or `Geometry` with a length-like property.
        epsilon: Minimum valid vector length. Use to avoid `inf` gradients for zero-length vectors.
            Lengths shorter than `eps` are set to 0.

    Returns:
        Length as `Tensor`
    """
    if isinstance(obj, Tensor):
        assert 'vector' in obj.shape, f"length() requires 'vector' dim but got {type(obj)} with shape {shape(obj)}."
        return math.norm(obj, 'vector', epsilon)
    elif isinstance(obj, Cylinder):
        return obj.depth
    raise ValueError(obj)


def squared_length(obj: Union[Geometry, Tensor]) -> Tensor:
    """
    Returns the squared length of a vector `Tensor` or geometric object with a length-like property.

    Args:
        obj: `Tensor` with 'vector' dim or `Geometry` with a length-like property.

    Returns:
        Squared length as `Tensor`
    """
    if isinstance(obj, Tensor):
        assert 'vector' in obj.shape, f"squared_length() requires 'vector' dim but got {type(obj)} with shape {shape(obj)}."
        return math.squared_norm(obj, 'vector')
    elif isinstance(obj, Cylinder):
        return obj.depth ** 2
    raise ValueError(obj)


def normalize(obj: Tensor, epsilon=1e-5, allow_infinite=False, allow_zero=True):
    """
    Normalize a vector `Tensor` along the 'vector' dim.

    Args:
        obj: `Tensor` with 'vector' dim.
        epsilon: (Optional) Zero-length threshold. Vectors shorter than this length yield the unit vector (1, 0, 0, ...).
            If not specified, the zero-vector yields `NaN` as it cannot be normalized.
        allow_infinite: Allow infinite components in vectors. These vectors will then only points towards the infinite components.
        allow_zero: Whether to return zero vectors for inputs smaller `epsilon` instead of a unit vector.

    Returns:
        `Tensor` of the same shape as `obj`.
    """
    assert 'vector' in obj.shape, f"normalize() requires 'vector' dim but got {type(obj)} with shape {shape(obj)}."
    return math.normalize(obj, 'vector', epsilon, allow_infinite=allow_infinite, allow_zero=allow_zero)


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
