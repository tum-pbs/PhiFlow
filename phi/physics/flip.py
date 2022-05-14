"""
Functions for running fluid implicit particle (FLIP) and particle-in-cell (PIC) simulations.
"""
import warnings

from phi import math, field

from phi.math._tensors import copy_with
from .fluid import Obstacle, _pressure_extrapolation
from phi.field import StaggeredGrid, PointCloud, Grid, finite_fill, CenteredGrid
from phi.geom import union, Sphere, Box, Geometry
from phi.math.extrapolation import Extrapolation, ZERO


def make_incompressible(velocity: StaggeredGrid,
                        particles: PointCloud,
                        obstacles: tuple or list or StaggeredGrid = (),
                        solve=math.Solve('auto', 1e-5, 0, gradient_solve=math.Solve('auto', 1e-5, 1e-5))):
    """
    Projects the given velocity field by solving for the pressure and subtracting its spatial_gradient.

    Args:
        velocity: Current velocity field as StaggeredGrid
        domain: Domain object
        particles: `PointCloud` holding the current positions of the particles
        obstacles: Sequence of `phi.physics.Obstacle` objects or binary StaggeredGrid marking through-flow cell faces
        solve: Parameters for the pressure solve_linear

    Returns:
      velocity: divergence-free velocity of type `type(velocity)`
      pressure: solved pressure field, `CenteredGrid`
      iterations: Number of iterations required to solve_linear for the pressure
      divergence: divergence field of input velocity, `CenteredGrid`
      occupation_mask: StaggeredGrid
    """
    points = particles.with_values(math.tensor(1., convert=True))
    active = CenteredGrid(points, velocity.extrapolation.spatial_gradient(), velocity.bounds, velocity.resolution)  # occupied centered
    occupied_staggered = StaggeredGrid(points, velocity.extrapolation, velocity.bounds, velocity.resolution)

    if isinstance(obstacles, StaggeredGrid):
        hard_bcs = obstacles
    else:
        hard_bcs = accessible_mask(union(*[obstacle.geometry for obstacle in obstacles]), velocity.extrapolation, velocity.bounds, velocity.resolution, type=StaggeredGrid)

    # --- Extrapolation is needed to exclude border divergence from the `occupied_centered` mask and thus
    # from the pressure solve_linear. If particles are randomly distributed, the `occupied_centered` mask
    # could sometimes include the divergence at the borders (due to single particles right at the edge
    # which temporarily deform the `occupied_centered` mask when moving into a new cell). This would then
    # get compensated by the pressure. This is unwanted for falling liquids and therefore prevented by this
    # extrapolation. ---
    velocity_field = finite_fill(velocity / occupied_staggered)
    velocity_field *= hard_bcs  # Enforces boundary conditions after extrapolation
    div = field.divergence(velocity_field) * active  # Multiplication with `occupied_centered` excludes border divergence from pressure solve_linear
    from phi import vis; vis.show(velocity_field, div, active)

    @field.jit_compile_linear
    def matrix_eq(p):
        return field.where(active, field.divergence(field.spatial_gradient(p, type=StaggeredGrid) * hard_bcs), p)

    if solve.x0 is None:
        pressure_extrapolation = _pressure_extrapolation(velocity.extrapolation)
        solve = copy_with(solve, x0=CenteredGrid(0, pressure_extrapolation, div.bounds, div.resolution))
    pressure = field.solve_linear(matrix_eq, div, solve)

    def pressure_backward(_p, _p_, dp):
        return dp * active.values,

    add_mask_in_gradient = math.custom_gradient(lambda p: p, pressure_backward)
    pressure = pressure.with_values(add_mask_in_gradient(pressure.values))

    gradp = field.spatial_gradient(pressure, type=type(velocity_field)) * hard_bcs
    return velocity_field - gradp, pressure, occupied_staggered


def map_velocity_to_particles(previous_particle_velocity: PointCloud,
                              velocity_grid: Grid,
                              previous_velocity_grid: Grid = None,
                              viscosity: float = 0.) -> PointCloud:
    """
    Maps result of velocity projection on grid back to particles.
    Provides option to choose between FLIP (particle velocities are updated by the change between projected and initial grid velocities)
    and PIC (particle velocities are replaced by the the projected velocities)
    method depending on the value of the `initial_v_field`.
    
    Args:
        previous_particle_velocity: PointCloud with particle positions as elements and their corresponding velocities as values
        velocity_grid: Divergence-free velocity grid
        occupation_mask: Binary Grid (same type as `velocity_grid`) indicating which cells hold particles
        previous_velocity_grid: Velocity field before projection and force update
        viscosity: If previous_velocity_grid is None, the particle-in-cell method (PIC) is applied.
            Otherwise this is the ratio between FLIP and PIC (0. for pure FLIP)

    Returns:
        PointCloud with particle positions as elements and updated particle velocities as values.
    """
    velocities = math.zeros_like(previous_particle_velocity.values)
    if viscosity > 0:
        # --- PIC ---
        velocities += viscosity * (velocity_grid @ previous_particle_velocity).values
    if viscosity < 1:
        # --- FLIP ---
        v_change_field = velocity_grid - previous_velocity_grid
        v_change = (v_change_field @ previous_particle_velocity).values
        velocities += (1 - viscosity) * (previous_particle_velocity.values + v_change)
    return previous_particle_velocity.with_values(velocities)


def respect_boundaries(particles: PointCloud, obstacles: tuple or list, offset: float = 0.5) -> PointCloud:
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
    if particles._bounds is not None:
        pos = (~particles.bounds).push(pos, shift_amount=offset)
    return particles.with_elements(particles.elements @ pos)


def accessible_mask(not_accessible: tuple or list, extrapolation, bounds, resolution, type: type = CenteredGrid) -> CenteredGrid or StaggeredGrid:
    """
    Unifies domain and Obstacle or Geometry objects into a binary StaggeredGrid mask which can be used
    to enforce boundary conditions.

    Args:
        not_accessible: blocked region(s) of space specified by geometries
        type: class of Grid to create, must be either CenteredGrid or StaggeredGrid
        extrapolation: (optional) grid extrapolation, defaults to Domain.boundaries['accessible']

    Returns:
        Binary mask indicating valid fields w.r.t. the boundary conditions.
    """
    warnings.warn("flip.accessible_mask() is deprecated. Use CenteredGrid(~union(not_accessible)) for centered grids and field.stagger(mask, math.minimum, extrapolation) for staggered grids.", DeprecationWarning)
    assert type in (CenteredGrid, StaggeredGrid)
    mask = CenteredGrid(~union(not_accessible), extrapolation, bounds, resolution)
    return field.stagger(mask, math.minimum, extrapolation) if type is StaggeredGrid else mask
