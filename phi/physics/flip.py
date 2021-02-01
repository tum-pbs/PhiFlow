from phi import math, field
from typing import List, Tuple
from phi.math import Tensor
from phi.physics import Domain, Obstacle
from phi.field import StaggeredGrid, HardGeometryMask, PointCloud, CenteredGrid, Grid, extrapolate_valid
from phi.geom import union, Sphere


def get_accessible_mask(domain: Domain, obstacles: List[Obstacle]) -> StaggeredGrid:
    """
    Unifies domain and obstacles into a binary StaggeredGrid mask which can be used to enforce
    boundary conditions

    Args:
        domain: Domain object
        obstacles: List of Obstacles

    Returns:
        Binary mask indicating valid fields w.r.t. the boundary conditions.
    """
    accessible = domain.grid(1 - HardGeometryMask(union([obstacle.geometry for obstacle in obstacles])))
    accessible_mask = domain.grid(accessible, extrapolation=domain.boundaries.accessible_extrapolation)
    return field.stagger(accessible_mask, math.minimum, accessible_mask.extrapolation)


def make_incompressible(velocity_field: StaggeredGrid,
                        accessible: StaggeredGrid,
                        cvalid: CenteredGrid,
                        svalid: StaggeredGrid,
                        pressure: CenteredGrid) -> Tuple[StaggeredGrid, CenteredGrid]:
    """
    Projects the given velocity field by solving for the pressure and subtracting its gradient.

    Args:
        velocity_field: Current velocity field as StaggeredGrid
        accessible: Boundary conditions as binary StaggeredGrid
        cvalid: Binary CenteredGrid indicating which cells hold particles
        svalid: Binary StaggeredGrid indicating which cells hold particles
        pressure: Initial pressure guess as CenteredGrid

    Returns:
        Projected velocity field and corresponding pressure field
    """
    velocity_field, _ = extrapolate_valid(velocity_field, svalid, 1)  # extrapolation conserves falling shapes
    velocity_field *= accessible  # Enforces boundary conditions after extrapolation
    div = field.divergence(velocity_field) * cvalid

    def laplace(p):
        # TODO: prefactor of pressure should not have any effect
        return field.where(cvalid, field.divergence(field.gradient(p, type=StaggeredGrid) * accessible), -4 * p)

    converged, pressure, iterations = field.solve(laplace, div, pressure, solve_params=math.LinearSolve(None, 1e-5))
    gradp = field.gradient(pressure, type=type(velocity_field))
    return velocity_field - gradp, pressure


def map_velocity_to_particles(previous_particle_velocity: PointCloud, velocity_grid: Grid, occupation_mask: Grid,
                              previous_velocity_grid: Grid = None) -> PointCloud:
    """
    Maps result of velocity projection on grid back to particles. Provides option to choose between FLIP (particle velocities are
    updated by the change between projected and initial grid velocities) and PIC (particle velocities are replaced by the the 
    projected velocities) method depending on the value of the `initial_v_field`.
    
    Args:
        previous_particle_velocity: PointCloud with particle positions as elements and their corresponding velocities as values
        velocity_grid: Divergence-free velocity grid
        occupation_mask: Binary grid (same type as `velocity_grid`) indicating which cells hold particles
        previous_velocity_grid: Velocity field before projection and force update. If None, the PIC method gets applied, FLIP otherwise

    Returns:
        PointCloud with particle positions as elements and updated particle velocities as values.
    """
    if previous_velocity_grid is not None:
        # --- FLIP ---
        v_change_field = velocity_grid - previous_velocity_grid
        v_change_field, _ = extrapolate_valid(v_change_field, occupation_mask, 1)
        v_change = v_change_field.sample_at(previous_particle_velocity.elements.center)
        return previous_particle_velocity.with_(values=previous_particle_velocity.values + v_change)
    else:
        # --- PIC ---
        v_div_free_field, _ = extrapolate_valid(velocity_grid, occupation_mask, 1)
        v_values = v_div_free_field.sample_at(previous_particle_velocity.elements.center)
        return previous_particle_velocity.with_(values=v_values)


def add_inflow(particles: PointCloud, inflow_points: Tensor, inflow_values: Tensor) -> PointCloud:
    """
    Merges the current particles with inflow particles.

    Args:
        particles: PointCloud with particle positions as elements and their corresponding velocities as values
        inflow_points: Tensor of new point positions which should get added (must hold 'points' dimension)
        inflow_values: Tensor of new points velocities (must hold 'points' dimension)

    Return:
        PointCloud with merged particle positions as elements and their velocities as values.
    """
    new_points = math.tensor(math.concat([particles.points, inflow_points], dim='points'), names=['points', 'vector'])
    new_values = math.tensor(math.concat([particles.values, inflow_values], dim='points'), names=['points', 'vector'])
    return particles.with_(elements=Sphere(new_points, 0), values=new_values)


def respect_boundaries(particles: PointCloud, domain: Domain, obstacles: tuple or list, offset=1.0) -> PointCloud:
    """
    Enforces boundary conditions by correcting possible errors of the advection step and shifting particles out of 
    obstacles or back into the domain.
    
    Args:
        particles: PointCloud holding particle positions as elements
        domain: Domain for which any particles outside should get shifted inside
        obstacles: List of obstacles where any particles inside should get shifted outwards
        offset: Distance between particles and domain boundary / obstacle surface after particles have been shifted.

    Returns:
        PointCloud where all particles are inside the domain / outside of obstacles.
    """
    new_positions = particles.elements.center
    for obstacle in obstacles:
        new_positions = obstacle.geometry.push(new_positions, shift_amount=offset)
    new_positions = (~domain.bounds).push(new_positions, shift_amount=offset)
    return particles.with_(elements=Sphere(new_positions, 0))
