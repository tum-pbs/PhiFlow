"""
Functions for simulating incompressible fluids, both grid-based and particle-based.

The main function for incompressible fluids (Eulerian as well as FLIP / PIC) is `make_incompressible()` which removes the divergence of a velocity field.
"""
from typing import Tuple

from phi import math, field
from phi.math import wrap, channel
from phi.field import SoftGeometryMask, AngularVelocity, Grid, divergence, spatial_gradient, where, CenteredGrid, PointCloud
from phi.geom import union, Geometry
from ..field._embed import FieldEmbedding
from ..field._grid import GridType
from ..math import extrapolation, NUMPY, batch, shape, non_channel, expand
from ..math._magic_ops import copy_with
from ..math.extrapolation import combine_sides, Extrapolation


class Obstacle:
    """
    An obstacle defines boundary conditions inside a geometry.
    It can also have a linear and angular velocity.
    """

    def __init__(self, geometry, velocity=0, angular_velocity=0):
        """
        Args:
            geometry: Physical shape and size of the obstacle.
            velocity: Linear velocity vector of the obstacle.
            angular_velocity: Rotation speed of the obstacle. Scalar value in 2D, vector in 3D.
        """
        self.geometry = geometry
        self.velocity = wrap(velocity, channel(geometry)) if isinstance(velocity, (tuple, list)) else velocity
        self.angular_velocity = angular_velocity
        self.shape = shape(geometry) & non_channel(self.velocity) & non_channel(angular_velocity)

    @property
    def is_stationary(self):
        """ Test whether the obstacle is completely still. """
        return isinstance(self.velocity, (int, float)) and self.velocity == 0 and isinstance(self.angular_velocity, (int, float)) and self.angular_velocity == 0

    def copied_with(self, **kwargs):
        geometry, velocity, angular_velocity = self.geometry, self.velocity, self.angular_velocity
        if 'geometry' in kwargs:
            geometry = kwargs['geometry']
        if 'velocity' in kwargs:
            velocity = kwargs['velocity']
        if 'angular_velocity' in kwargs:
            angular_velocity = kwargs['angular_velocity']
        return Obstacle(geometry, velocity, angular_velocity)


def make_incompressible(velocity: GridType,
                        obstacles: tuple or list = (),
                        solve=math.Solve('auto', 1e-5, 1e-5, gradient_solve=math.Solve('auto', 1e-5, 1e-5)),
                        active: CenteredGrid = None) -> Tuple[GridType, CenteredGrid]:
    """
    Projects the given velocity field by solving for the pressure and subtracting its spatial_gradient.
    
    This method is similar to :func:`field.divergence_free()` but differs in how the boundary conditions are specified.

    Args:
        velocity: Vector field sampled on a grid
        obstacles: List of Obstacles to specify boundary conditions inside the domain (Default value = ())
        solve: Parameters for the pressure solve as.
        active: (Optional) Mask for which cells the pressure should be solved.
            If given, the velocity may take `NaN` values where it does not contribute to the pressure.
            Also, the total divergence will never be subtracted if active is given, even if all values are 1.

    Returns:
        velocity: divergence-free velocity of type `type(velocity)`
        pressure: solved pressure field, `CenteredGrid`
    """
    assert isinstance(obstacles, (tuple, list)), f"obstacles must be a tuple or list but got {type(obstacles)}"
    obstacles = [Obstacle(o) if isinstance(o, Geometry) else o for o in obstacles]
    for obstacle in obstacles:
        assert obstacle.geometry.vector.item_names == velocity.vector.item_names, f"Obstacles must live in the same physical space as the velocity field {velocity.vector.item_names} but got {type(obstacle.geometry).__name__} obstacle with order {obstacle.geometry.vector.item_names}"
    input_velocity = velocity
    # --- Create masks ---
    accessible_extrapolation = _accessible_extrapolation(input_velocity.extrapolation)
    with NUMPY:
        accessible = CenteredGrid(~union([obs.geometry for obs in obstacles]), accessible_extrapolation, velocity.bounds, velocity.resolution)
        hard_bcs = field.stagger(accessible, math.minimum, input_velocity.extrapolation, type=type(velocity))
    all_active = active is None
    if active is None:
        active = accessible.with_extrapolation(extrapolation.NONE)
    else:
        active *= accessible  # no pressure inside obstacles
    # --- Linear solve ---
    velocity = apply_boundary_conditions(velocity, obstacles)
    div = divergence(velocity) * active
    if not all_active:  # NaN in velocity allowed
        div = field.where(field.is_finite(div), div, 0)
    if not input_velocity.extrapolation.is_flexible and all_active:
        assert solve.preprocess_y is None, "fluid.make_incompressible() does not support custom preprocessing"
        solve = copy_with(solve, preprocess_y=_balance_divergence, preprocess_y_args=(active,))
    if solve.x0 is None:
        pressure_extrapolation = _pressure_extrapolation(input_velocity.extrapolation)
        solve = copy_with(solve, x0=CenteredGrid(0, pressure_extrapolation, div.bounds, div.resolution))
    if batch(math.merge_shapes(*obstacles)).without(batch(solve.x0)):  # The initial pressure guess must contain all batch dimensions
        solve = copy_with(solve, x0=expand(solve.x0, batch(math.merge_shapes(*obstacles))))
    pressure = math.solve_linear(masked_laplace, f_args=[hard_bcs, active], y=div, solve=solve)
    # --- Subtract grad p ---
    grad_pressure = field.spatial_gradient(pressure, input_velocity.extrapolation, type=type(velocity)) * hard_bcs
    velocity = velocity - grad_pressure
    return velocity, pressure


@math.jit_compile_linear  # jit compilation is required for boundary conditions that add a constant offset solving Ax + b = y
def masked_laplace(pressure: CenteredGrid, hard_bcs: Grid, active: CenteredGrid) -> CenteredGrid:
    """
    Computes the laplace of `pressure` in the presence of obstacles.

    Args:
        pressure: Pressure field.
        hard_bcs: Mask encoding which cells are connected to each other.
            One between fluid cells, zero inside and at the boundary of obstacles.
            This should be of the same type as the velocity, i.e. `StaggeredGrid` or `CenteredGrid`.
        active: Mask indicating for which cells the pressure value is valid.
            Linear solves will only determine the pressure for these cells.
            This is generally zero inside obstacles and in non-simulated regions.

    Returns:
        `CenteredGrid`
    """
    grad = spatial_gradient(pressure, hard_bcs.extrapolation, type=type(hard_bcs))
    valid_grad = grad * hard_bcs
    div = divergence(valid_grad)
    laplace = where(active, div, pressure)
    return laplace


def _balance_divergence(div, active):
    return div - active * (field.mean(div) / field.mean(active))


def apply_boundary_conditions(velocity: Grid or PointCloud, obstacles: tuple or list):
    """
    Enforces velocities boundary conditions on a velocity grid.
    Cells inside obstacles will get their velocity from the obstacle movement.
    Cells outside far away will be unaffected.

    Args:
      velocity: Velocity `Grid`.
      obstacles: Obstacles as `tuple` or `list`

    Returns:
        Velocity of same type as `velocity`
    """
    # velocity = field.bake_extrapolation(velocity)  # TODO we should bake only for divergence but keep correct extrapolation for velocity. However, obstacles should override extrapolation.
    for obstacle in obstacles:
        if isinstance(obstacle, Geometry):
            obstacle = Obstacle(obstacle)
        assert isinstance(obstacle, Obstacle)
        obs_mask = SoftGeometryMask(obstacle.geometry, balance=1) @ velocity
        if obstacle.is_stationary:
            velocity = (1 - obs_mask) * velocity
        else:
            angular_velocity = AngularVelocity(location=obstacle.geometry.center, strength=obstacle.angular_velocity, falloff=None) @ velocity
            velocity = (1 - obs_mask) * velocity + obs_mask * (angular_velocity + obstacle.velocity)
    return velocity


def boundary_push(particles: PointCloud, obstacles: tuple or list, offset: float = 0.5) -> PointCloud:
    """
    Enforces boundary conditions by correcting possible errors of the advection step and shifting particles out of
    obstacles or back into the domain.

    Args:
        particles: PointCloud holding particle positions as elements
        obstacles: List of `Obstacle` or `Geometry` objects where any particles inside should get shifted outwards
        offset: Minimum distance between particles and domain boundary / obstacle surface after particles have been shifted.

    Returns:
        PointCloud where all particles are inside the domain / outside of obstacles.
    """
    pos = particles.elements.center
    for obj in obstacles:
        geometry = obj.geometry if isinstance(obj, Obstacle) else obj
        assert isinstance(geometry, Geometry), f"obstacles must be a list of Obstacle or Geometry objects but got {type(obj)}"
        pos = geometry.push(pos, shift_amount=offset)
    return particles.with_elements(particles.elements @ pos)


def _pressure_extrapolation(vext: Extrapolation):
    if vext == extrapolation.PERIODIC:
        return extrapolation.PERIODIC
    elif vext == extrapolation.BOUNDARY:
        return extrapolation.ZERO
    elif isinstance(vext, extrapolation.ConstantExtrapolation):
        return extrapolation.BOUNDARY
    elif isinstance(vext, extrapolation._MixedExtrapolation):
        return combine_sides(**{dim: (_pressure_extrapolation(lo), _pressure_extrapolation(hi)) for dim, (lo, hi) in vext.ext.items()})
    else:
        raise ValueError(f"Unsupported extrapolation: {type(vext)}")


def _accessible_extrapolation(vext: Extrapolation):
    """ Determine whether outside cells are accessible based on the velocity extrapolation. """
    if vext == extrapolation.PERIODIC:
        return extrapolation.PERIODIC
    elif vext == extrapolation.BOUNDARY:
        return extrapolation.ONE
    elif isinstance(vext, extrapolation.ConstantExtrapolation):
        return extrapolation.ZERO
    elif isinstance(vext, FieldEmbedding):
        return extrapolation.ONE
    elif isinstance(vext, extrapolation._MixedExtrapolation):
        return combine_sides(**{dim: (_accessible_extrapolation(lo), _accessible_extrapolation(hi)) for dim, (lo, hi) in vext.ext.items()})
    elif isinstance(vext, extrapolation._NormalTangentialExtrapolation):
        return _accessible_extrapolation(vext.normal)
    else:
        raise ValueError(f"Unsupported extrapolation: {type(vext)}")
