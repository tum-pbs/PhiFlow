from phi import math, field
from typing import Tuple
from phi.physics import Domain, Obstacle
from phi.field import StaggeredGrid, CenteredGrid, PointCloud, Grid, extrapolate_valid
from phi.geom import union, Sphere


def make_incompressible(velocity: StaggeredGrid,
                        domain: Domain,
                        particles: PointCloud,
                        obstacles: tuple or list or StaggeredGrid = (),
                        solve_params=math.Solve('CG', 1e-5, 0),
                        pressure_guess: CenteredGrid = None) -> Tuple[StaggeredGrid, CenteredGrid, int, CenteredGrid, StaggeredGrid]:
    """
    Projects the given velocity field by solving for the pressure and subtracting its spatial_gradient.

    Args:
        velocity: Current velocity field as StaggeredGrid
        domain: Domain object
        particles: Pointcloud holding the current positions of the particles
        obstacles: Sequence of `phi.physics.Obstacle` objects or binary StaggeredGrid marking through-flow cell faces
        solve_params (Optional): Parameters for the pressure solve
        pressure_guess (Optional): Initial pressure guess as CenteredGrid

    Returns:
      velocity: divergence-free velocity of type `type(velocity)`
      pressure: solved pressure field, `CenteredGrid`
      iterations: Number of iterations required to solve for the pressure
      divergence: divergence field of input velocity, `CenteredGrid`
      occupation_mask: StaggeredGrid
    """
    points = particles.with_(values=math.tensor(1., convert=True))
    occupied_centered = points >> domain.scalar_grid()
    occupied_staggered = points >> domain.staggered_grid()

    if isinstance(obstacles, StaggeredGrid):
        accessible = obstacles
    else:
        accessible = domain.accessible_mask(union(*[obstacle.geometry for obstacle in obstacles]), type=StaggeredGrid)

    # --- Extrapolation is needed to exclude border divergence from the `occupied_centered` mask and thus
    # from the pressure solve. If particles are randomly distributed, the `occupied_centered` mask
    # could sometimes include the divergence at the borders (due to single particles right at the edge
    # which temporarily deform the `occupied_centered` mask when moving into a new cell) which would then
    # get compensated by the pressure. This is unwanted for falling liquids and therefore prevented by this
    # extrapolation. ---
    velocity_field, _ = extrapolate_valid(velocity * occupied_staggered, occupied_staggered, 1)
    velocity_field *= accessible  # Enforces boundary conditions after extrapolation
    div = field.divergence(velocity_field) * occupied_centered  # Multiplication with `occupied_centered` excludes border divergence from pressure solve

    @field.linear_function
    def matrix_eq(p):
        return field.where(occupied_centered, field.divergence(field.spatial_gradient(p, type=StaggeredGrid) * accessible), p)

    pressure = field.solve(matrix_eq, div, pressure_guess or domain.scalar_grid(), solve_params=solve_params)
    gradp = field.spatial_gradient(pressure, type=type(velocity_field)) * accessible
    return velocity_field - gradp, pressure, solve_params.result.iterations, div, occupied_staggered


def map_velocity_to_particles(previous_particle_velocity: PointCloud, velocity_grid: Grid, occupation_mask: Grid,
                              previous_velocity_grid: Grid = None, viscosity: float = 0.) -> PointCloud:
    """
    Maps result of velocity projection on grid back to particles. Provides option to choose between FLIP (particle velocities are
    updated by the change between projected and initial grid velocities) and PIC (particle velocities are replaced by the the 
    projected velocities) method depending on the value of the `initial_v_field`.
    
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
        velocities += viscosity * velocity_grid.sample_at(previous_particle_velocity.elements.center)
    if viscosity < 1.:
        # --- FLIP ---
        v_change_field = velocity_grid - previous_velocity_grid
        v_change_field, _ = extrapolate_valid(v_change_field, occupation_mask)
        v_change = v_change_field.sample_at(previous_particle_velocity.elements.center)
        velocities += (1 - viscosity) * (previous_particle_velocity.values + v_change)
    return previous_particle_velocity.with_(values=velocities)


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
    return particles.with_(elements=Sphere(new_positions, math.mean(particles.bounds.size) * 0.005))
