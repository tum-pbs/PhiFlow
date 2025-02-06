"""
Functions for simulating incompressible fluids, both grid-based and particle-based.

The main function for incompressible fluids (Eulerian as well as FLIP / PIC) is `make_incompressible()` which removes the divergence of a velocity field.
"""
import warnings
from typing import Tuple, Callable, Union, List, Optional

from phi import math, field
from phi.math import wrap, channel, Solve
from phi.field import AngularVelocity, Grid, divergence, spatial_gradient, where, CenteredGrid, PointCloud, Field, resample
from phi.geom import union, Geometry
from phiml.math import Tensor
from ..field._embed import FieldEmbedding
from ..field._grid import StaggeredGrid
from ..math import extrapolation, NUMPY, batch, shape, non_channel, expand
from phiml.math._magic_ops import copy_with
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
        """ Test whether the obstacle is completely still, i.e. not moving or rotating. """
        return not self.is_moving and not self.is_rotating

    @property
    def is_rotating(self):
        """
        Checks whether this obstacle might be rotating.
        This also evaluates to `True` if the angular velocity is unknown at this time.
        """
        return not math.always_close(self.angular_velocity, 0)

    @property
    def is_moving(self):
        """
        Checks whether this obstacle might be moving.
        This also evaluates to `True` if the velocity is unknown at this time.
        """
        return not math.always_close(self.velocity, 0)

    def copied_with(self, **kwargs):
        warnings.warn("Obstacle.copied_with is deprecated. Use math.copy_with instead.", DeprecationWarning, stacklevel=2)
        return math.copy_with(self, **kwargs)

    def __variable_attrs__(self) -> Tuple[str, ...]:
        return 'geometry', 'velocity', 'angular_velocity'

    def with_geometry(self, geometry):
        return Obstacle(geometry, self.velocity, self.angular_velocity)

    def shifted(self, delta: Tensor):
        return self.with_geometry(self.geometry.shifted(delta))

    def at(self, position: Tensor):
        return self.with_geometry(self.geometry.at(position))

    def rotated(self, angle: Union[float, Tensor]):
        return self.with_geometry(self.geometry.rotated(angle))

    def __eq__(self, other):
        if not isinstance(other, Obstacle):
            return False
        return self.geometry == other.geometry and self.velocity == other.velocity and self.angular_velocity == other.angular_velocity


def _get_obstacles_for(obstacles, space: Field) -> List[Obstacle]:
    obstacles = [obstacles] if isinstance(obstacles, (Obstacle, Geometry)) else obstacles
    assert isinstance(obstacles, (tuple, list)), f"obstacles must be an Obstacle or Geometry or a tuple/list thereof but got {type(obstacles)}"
    obstacles = [Obstacle(o) if isinstance(o, Geometry) else o for o in obstacles]
    for obstacle in obstacles:
        assert obstacle.geometry.vector.item_names == space.vector.item_names, f"Obstacles must live in the same physical space as the velocity field {space.vector.item_names} but got {type(obstacle.geometry).__name__} obstacle with order {obstacle.geometry.vector.item_names}"
    return obstacles


def make_incompressible(velocity: Field,
                        obstacles: Obstacle or Geometry or tuple or list = (),
                        solve: Solve = Solve(),
                        active: CenteredGrid = None,
                        order: int = 2,
                        correct_skew=False,
                        wide_stencil: bool = None) -> Tuple[Field, Field]:
    """
    Projects the given velocity field by solving for the pressure and subtracting its spatial_gradient.

    This method is similar to :func:`field.divergence_free()` but differs in how the boundary conditions are specified.

    Args:
        velocity: Vector field sampled on a grid.
        obstacles: `Obstacle` or `phi.geom.Geometry` or tuple/list thereof to specify boundary conditions inside the domain.
        solve: `Solve` object specifying method and tolerances for the implicit pressure solve.
        active: (Optional) Mask for which cells the pressure should be solved.
            If given, the velocity may take `NaN` values where it does not contribute to the pressure.
            Also, the total divergence will never be subtracted if active is given, even if all values are 1.
        order: spatial order for derivative computations.
            For Higher-order schemes, the laplace operation is not conducted with a stencil exactly corresponding to the one used in divergence calculations but a smaller one instead.
            While this disrupts the formal correctness of the method it only induces insignificant errors and yields considerable performance gains.
            supported: explicit 2/4th order - implicit 6th order (obstacles are only supported with explicit 2nd order)

    Returns:
        velocity: divergence-free velocity of type `type(velocity)`
        pressure: solved pressure field, `CenteredGrid`
    """
    assert not correct_skew
    obstacles = _get_obstacles_for(obstacles, velocity)
    assert order <= 2 or len(obstacles) == 0, f"obstacles are not supported with higher order schemes"
    assert not velocity.is_mesh or not obstacles, f"Meshes don't support obstacle masks. Apply the obstacle when building the mesh instead."
    input_velocity = velocity
    # --- Obstacles ---
    all_active = active is None
    hard_bcs = None
    if obstacles:
        accessible_boundary = _accessible_extrapolation(input_velocity.extrapolation)
        with NUMPY:
            accessible = Field(velocity.geometry, ~union([obs.geometry for obs in obstacles]), accessible_boundary)
            # accessible = CenteredGrid(~union([obs.geometry for obs in obstacles]), accessible_boundary, velocity.bounds, velocity.resolution)
            hard_bcs = field.stagger(accessible, math.minimum, velocity.boundary, at=velocity.sampled_at, dims=velocity.vector.item_names)
        active = accessible.with_extrapolation(extrapolation.NONE) if active is None else active * accessible  # no pressure inside obstacles
        velocity = apply_boundary_conditions(velocity, obstacles)
    div = divergence(velocity, order=order)
    if active is not None:
        div *= active  # inactive cells must solvable
    assert not channel(div), f"Divergence must not have any channel dimensions. This is likely caused by an improper velocity field v={input_velocity}"
    # --- Linear solve for pressure ---
    if not all_active:  # NaN in velocity allowed
        div = field.where(field.is_finite(div), div, 0)
    if not input_velocity.extrapolation.is_flexible and all_active:
        solve = solve.with_preprocessing(_balance_divergence, active)
        if solve.rank_deficiency is None:
            solve = copy_with(solve, rank_deficiency=1)
    if solve.x0 is None:
        pressure_extrapolation = _pressure_extrapolation(input_velocity.extrapolation)
        solve = copy_with(solve, x0=Field(div.geometry, 0, pressure_extrapolation))  # convert=False
    if (batch(math.merge_shapes(*obstacles)) & batch(velocity)).without(batch(solve.x0.values)):  # The initial pressure guess must contain all batch dimensions
        solve = copy_with(solve, x0=solve.x0.with_values(expand(solve.x0.values, batch(math.merge_shapes(*obstacles)) & batch(velocity))))
    if wide_stencil is None:
        wide_stencil = not velocity.is_staggered
    pressure = math.solve_linear(masked_laplace, div, solve, velocity.boundary, hard_bcs, active, wide_stencil=wide_stencil, order=order, implicit=None, upwind=None, correct_skew=correct_skew)
    # --- Subtract grad p ---
    grad_pressure = field.spatial_gradient(pressure, input_velocity.extrapolation, at=velocity.sampled_at, order=order, scheme='green-gauss')
    if hard_bcs is not None:
        grad_pressure *= hard_bcs
    velocity = (velocity - grad_pressure).with_extrapolation(input_velocity.extrapolation)
    return velocity, pressure


@math.jit_compile_linear(forget_traces=True)
def masked_laplace(pressure: Field,
                   v_boundary: Extrapolation,
                   hard_bcs: Field,
                   active: Field,
                   wide_stencil=False,
                   order=2,
                   implicit: Solve = None,
                   upwind=None,
                   correct_skew=False) -> CenteredGrid:
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
        order: Spatial order of accuracy.
            Higher orders entail larger stencils and more computation time but result in more accurate results assuming a large enough resolution.
            Supported: 2 explicit, 4 explicit, 6 implicit (inherited from `phi.field.laplace()`).

    Returns:
        `CenteredGrid`
    """
    if pressure.is_mesh:
        return field.laplace(pressure, gradient=None, order=order, upwind=upwind, correct_skew=correct_skew, wide_stencil=wide_stencil)
    if order > 2 and not wide_stencil:
        return field.laplace(pressure, order=order)
    assert pressure.is_grid and pressure.is_centered, f"Only mesh and centered grid supported for pressure"
    grad = spatial_gradient(pressure, v_boundary, at='center' if wide_stencil else 'face')
    valid_grad = grad * hard_bcs if hard_bcs is not None else grad
    valid_grad = valid_grad.with_boundary(extrapolation.remove_constant_offset(valid_grad.extrapolation))
    div = divergence(valid_grad)
    return where(active, div, pressure) if active is not None else div


def _balance_divergence(div: Field, active: Optional[Field]) -> Field:
    if active is not None:
        return div - active * (field.mean(div) / field.mean(active))
    else:
        return div - field.mean(div)


def apply_boundary_conditions(velocity: Grid or PointCloud, obstacles: Obstacle or Geometry or tuple or list):
    """
    Enforces velocities boundary conditions on a velocity grid.
    Cells inside obstacles will get their velocity from the obstacle movement.
    Cells outside far away will be unaffected.

    Args:
      velocity: Velocity `Grid`.
        obstacles: `Obstacle` or `phi.geom.Geometry` or tuple/list thereof to specify boundary conditions inside the domain.

    Returns:
        Velocity of same type as `velocity`
    """
    obstacles = _get_obstacles_for(obstacles, velocity)
    # velocity = field.bake_extrapolation(velocity)  # TODO we should bake only for divergence but keep correct extrapolation for velocity. However, obstacles should override extrapolation.
    for obstacle in obstacles:
        if isinstance(obstacle, Geometry):
            obstacle = Obstacle(obstacle)
        assert isinstance(obstacle, Obstacle)
        obs_mask = resample(obstacle.geometry, velocity, soft=True, balance=1)
        if obstacle.is_stationary:
            velocity = field.safe_mul(1 - obs_mask, velocity)
        else:
            if obstacle.is_rotating:
                angular_velocity = resample(AngularVelocity(location=obstacle.geometry.center, strength=obstacle.angular_velocity, falloff=None), to=velocity)
            else:
                angular_velocity = 0
            velocity = field.safe_mul(1 - obs_mask, velocity) + field.safe_mul(obs_mask, angular_velocity + obstacle.velocity)
    return velocity


def boundary_push(particles: PointCloud, obstacles: tuple or list, separation: float = 0.5) -> PointCloud:
    """
    Enforces boundary conditions by correcting possible errors of the advection step and shifting particles out of
    obstacles or back into the domain.

    Args:
        particles: PointCloud holding particle positions as elements
        obstacles: List of `Obstacle` or `Geometry` objects where any particles inside should get shifted outwards
        separation: Minimum distance between particles and domain boundary / obstacle surface after particles have been shifted.

    Returns:
        PointCloud where all particles are inside the domain / outside of obstacles.
    """
    pos = particles.geometry.center
    for obj in obstacles:
        geometry = obj.geometry if isinstance(obj, Obstacle) else obj
        assert isinstance(geometry, Geometry), f"obstacles must be a list of Obstacle or Geometry objects but got {type(obj)}"
        pos = geometry.push(pos, shift_amount=separation)
    return particles.with_elements(particles.geometry.at(pos))


def _pressure_extrapolation(vext: Extrapolation):
    if vext == extrapolation.PERIODIC:
        return extrapolation.PERIODIC
    elif vext == extrapolation.BOUNDARY:
        return extrapolation.ZERO
    elif isinstance(vext, extrapolation.ConstantExtrapolation):
        return extrapolation.BOUNDARY
    elif isinstance(vext, FieldEmbedding):
        return extrapolation.BOUNDARY
    else:
        return extrapolation.map(_pressure_extrapolation, vext)


def _accessible_extrapolation(vext: Extrapolation):
    """ Determine whether outside cells are accessible based on the velocity extrapolation. """
    vext = extrapolation.get_normal(vext)
    if vext == extrapolation.PERIODIC:
        return extrapolation.PERIODIC
    elif vext == extrapolation.BOUNDARY:
        return extrapolation.ONE
    elif isinstance(vext, extrapolation.ConstantExtrapolation):
        return extrapolation.ZERO
    elif isinstance(vext, FieldEmbedding):
        return extrapolation.ONE
    return extrapolation.map(_accessible_extrapolation, vext)


def incompressible_rk4(pde: Callable, velocity: Field, pressure: Field, dt, pressure_order=4, pressure_solve=Solve('CG'), **pde_aux_kwargs):
    """
    Implements the 4th-order Runge-Kutta time advancement scheme for incompressible vector fields.
    This approach is inspired by [Kampanis et. al., 2006](https://www.sciencedirect.com/science/article/pii/S0021999105005061) and incorporates the pressure treatment into the time step.

    Args:
        pde: Momentum equation. Function that computes all PDE terms not related to pressure, e.g. diffusion, advection, external forces.
        velocity: Velocity grid at time `t`.
        pressure: Pressure at time `t`.
        dt: Time increment to integrate.
        pressure_order: spatial order for derivative computations.
            For Higher-order schemes, the laplace operation is not conducted with a stencil exactly corresponding to the one used in divergence calculations but a smaller one instead.
            While this disrupts the formal correctness of the method it only induces insignificant errors and yields considerable performance gains.
            supported: explicit 2/4th order - implicit 6th order (obstacles are only supported with explicit 2nd order)
        pressure_solve: `Solve` object specifying method and tolerances for the implicit pressure solve.
        **pde_aux_kwargs: Auxiliary arguments for `pde`. These are considered constant over time.

    Returns:
        velocity: Velocity at time `t+dt`, same type as `velocity`.
        pressure: Pressure grid at time `t+dt`, `CenteredGrid`.
    """
    v1, p1 = velocity, pressure
    # PDE at current point
    rhs1 = pde(v1, **pde_aux_kwargs) - p1.gradient(at=v1.sampled_at, order=pressure_order)
    v2_old = velocity + (dt / 2) * rhs1
    v2, delta_p = make_incompressible(v2_old, solve=pressure_solve, order=pressure_order)
    p2 = p1 + delta_p / dt
    # PDE at half-point
    rhs2 = pde(v2, **pde_aux_kwargs) - p2.gradient(at=v1.sampled_at, order=pressure_order)
    v3_old = velocity + (dt / 2) * rhs2
    v3, delta_p = make_incompressible(v3_old, solve=pressure_solve, order=pressure_order)
    p3 = p2 + delta_p / dt
    # PDE at corrected half-point
    rhs3 = pde(v3, **pde_aux_kwargs) - p3.gradient(at=v1.sampled_at, order=pressure_order)
    v4_old = velocity + dt * rhs2
    v4, delta_p = make_incompressible(v4_old, solve=pressure_solve, order=pressure_order)
    p4 = p3 + delta_p / dt
    # PDE at RK4 point
    rhs4 = pde(v4, **pde_aux_kwargs) - p4.gradient(at=v1.sampled_at, order=pressure_order)
    v_p1_old = velocity + (dt / 6) * (rhs1 + 2 * rhs2 + 2 * rhs3 + rhs4)
    p_p1_old = (1 / 6) * (p1 + 2 * p2 + 2 * p3 + p4)
    v_p1, delta_p = make_incompressible(v_p1_old, solve=pressure_solve, order=pressure_order)
    p_p1 = p_p1_old + delta_p / dt
    return v_p1, p_p1
