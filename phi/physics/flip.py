"""
Functions for running fluid implicit particle (FLIP) and particle-in-cell (PIC) simulations.
"""
from phi import math, field

from phi.math._tensors import copy_with
from ._boundaries import Domain, Obstacle
from phi.field import StaggeredGrid, PointCloud, Grid, extrapolate_valid
from phi.geom import union, Sphere


def make_incompressible(velocity: StaggeredGrid,
                        domain: Domain,
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
    occupied_centered = points @ domain.scalar_grid()
    occupied_staggered = points @ domain.staggered_grid()

    if isinstance(obstacles, StaggeredGrid):
        accessible = obstacles
    else:
        accessible = domain.accessible_mask(union(*[obstacle.geometry for obstacle in obstacles]), type=StaggeredGrid)

    # --- Extrapolation is needed to exclude border divergence from the `occupied_centered` mask and thus
    # from the pressure solve_linear. If particles are randomly distributed, the `occupied_centered` mask
    # could sometimes include the divergence at the borders (due to single particles right at the edge
    # which temporarily deform the `occupied_centered` mask when moving into a new cell). This would then
    # get compensated by the pressure. This is unwanted for falling liquids and therefore prevented by this
    # extrapolation. ---
    velocity_field, _ = extrapolate_valid(velocity * occupied_staggered, occupied_staggered, 1)
    velocity_field *= accessible  # Enforces boundary conditions after extrapolation
    div = field.divergence(velocity_field) * occupied_centered  # Multiplication with `occupied_centered` excludes border divergence from pressure solve_linear

    @field.jit_compile_linear
    def matrix_eq(p):
        return field.where(occupied_centered, field.divergence(field.spatial_gradient(p, type=StaggeredGrid) * accessible), p)

    if solve.x0 is None:
        solve = copy_with(solve, x0=domain.scalar_grid())
    pressure = field.solve_linear(matrix_eq, div, solve)

    def pressure_backward(_p, _p_, dp):
        return dp * occupied_centered.values,

    add_mask_in_gradient = math.custom_gradient(lambda p: p, pressure_backward)
    pressure = pressure.with_values(add_mask_in_gradient(pressure.values))

    gradp = field.spatial_gradient(pressure, type=type(velocity_field)) * accessible
    return velocity_field - gradp, pressure, occupied_staggered


def map_velocity_to_particles(previous_particle_velocity: PointCloud,
                              velocity_grid: Grid,
                              occupation_mask: Grid,
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
    viscosity = min(max(0., viscosity), 1.)
    if previous_velocity_grid is None:
        viscosity = 1.
    velocities = math.zeros_like(previous_particle_velocity.values)
    if viscosity > 0.:
        # --- PIC ---
        velocity_grid, _ = extrapolate_valid(velocity_grid, occupation_mask)
        velocities += viscosity * (velocity_grid @ previous_particle_velocity).values
    if viscosity < 1.:
        # --- FLIP ---
        v_change_field = velocity_grid - previous_velocity_grid
        v_change_field, _ = extrapolate_valid(v_change_field, occupation_mask)
        v_change = (v_change_field @ previous_particle_velocity).values
        velocities += (1 - viscosity) * (previous_particle_velocity.values + v_change)
    return previous_particle_velocity.with_values(velocities)


def respect_boundaries(particles: PointCloud, domain: Domain, not_accessible: list, offset: float = 0.5) -> PointCloud:
    """
    Enforces boundary conditions by correcting possible errors of the advection step and shifting particles out of 
    obstacles or back into the domain.
    
    Args:
        particles: PointCloud holding particle positions as elements
        domain: Domain for which any particles outside should get shifted inwards
        not_accessible: List of Obstacle or Geometry objects where any particles inside should get shifted outwards
        offset: Minimum distance between particles and domain boundary / obstacle surface after particles have been shifted.

    Returns:
        PointCloud where all particles are inside the domain / outside of obstacles.
    """
    new_positions = particles.elements.center
    for obj in not_accessible:
        if isinstance(obj, Obstacle):
            obj = obj.geometry
        new_positions = obj.push(new_positions, shift_amount=offset)
    new_positions = (~domain.bounds).push(new_positions, shift_amount=offset)
    return particles.with_elements(Sphere(new_positions, math.mean(particles.bounds.size) * 0.005))
