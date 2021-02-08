from phi import math, field
from typing import Tuple
from phi.physics import Domain, Obstacle
from phi.field import StaggeredGrid, CenteredGrid, PointCloud, HardGeometryMask, Grid, extrapolate_valid
from phi.geom import union, Sphere, Geometry


def get_points(domain: Domain, geometries: Geometry or Obstacle or list,
               points_per_cell: int = 8,
               color: str = None,
               initial_velocity: tuple = (0, 0)) -> PointCloud:
    """
    Transforms Geometry or Obstacle objects into a PointCloud.

    Args:
        domain: Domain object
        geometries: Single or multiple Geometry or Obstacle objects
        points_per_cell: Number of points for each cell of `geometries`
        color (Optional): Color of PointCloud
        initial_velocity (Optional): Tuple with x, y velocities

    Returns:
         PointCloud representation of `geometries`.
    """
    if not isinstance(geometries, list):
        geometries = [geometries]
    for ix in range(len(geometries)):
        if isinstance(geometries[ix], Obstacle):
            geometries[ix] = geometries[ix].geometry
    point_mask = domain.grid(HardGeometryMask(union(geometries)))
    initial_points = math.distribute_points(point_mask.values, points_per_cell)
    return domain.points(initial_points, color=color) * initial_velocity


def get_accessible_mask(domain: Domain, not_accessible: Geometry or Obstacle or list) -> StaggeredGrid:
    """
    Unifies domain and Obstacle or Geometry objects into a binary StaggeredGrid mask which can be used
    to enforce boundary conditions.

    Args:
        domain: Domain object
        not_accessible: Obstacle or Geometry objects which are not accessible

    Returns:
        Binary mask indicating valid fields w.r.t. the boundary conditions.
    """
    if not isinstance(not_accessible, list):
        not_accessible = [not_accessible]
    for ix in range(len(not_accessible)):
        if isinstance(not_accessible[ix], Obstacle):
            not_accessible[ix] = not_accessible[ix].geometry
    accessible = domain.grid(1 - HardGeometryMask(union(not_accessible)))
    accessible_mask = domain.grid(accessible, extrapolation=domain.boundaries.accessible_extrapolation)
    return field.stagger(accessible_mask, math.minimum, accessible_mask.extrapolation)


def make_incompressible(velocity_field: StaggeredGrid,
                        accessible: StaggeredGrid,
                        particles: PointCloud = None,
                        domain: Domain = None,
                        pressure: CenteredGrid = None,
                        occupied_centered: CenteredGrid = None,
                        occupied_staggered: StaggeredGrid = None) -> Tuple[StaggeredGrid, CenteredGrid, StaggeredGrid]:
    """
    Projects the given velocity field by solving for the pressure and subtracting its gradient.

    Args:
        velocity_field: Current velocity field as StaggeredGrid
        accessible: Boundary conditions as binary StaggeredGrid
        particles (Optional if occupation masks are provided): Pointcloud holding the current positions of the particles
        domain (Optional if occupation masks are provided): Domain object
        pressure (Optional): Initial pressure guess as CenteredGrid
        occupied_centered (Optional): Binary mask indicating CenteredGrid cells which hold particles
        occupied_staggered (Optional): Binary mask indicating StaggeredGrid cells which hold particles

    Returns:
        Projected velocity field, pressure field and occupation_mask
    """
    if occupied_centered is None or occupied_staggered is None:
        assert particles is not None and domain is not None, 'Particles and Domain are necessary if occupation masks are not provided.'
        points = domain.points(particles.elements.center)  # ensure values == 1
    occupied_centered = occupied_centered or points >> domain.grid()
    occupied_staggered = occupied_staggered or points >> domain.staggered_grid()

    velocity_field, _ = extrapolate_valid(velocity_field, occupied_staggered, 1)  # extrapolation conserves falling shapes
    velocity_field *= accessible  # Enforces boundary conditions after extrapolation
    div = field.divergence(velocity_field) * occupied_centered

    def matrix_eq(p):
        return field.where(occupied_centered, field.divergence(field.gradient(p, type=StaggeredGrid) * accessible), p)

    converged, pressure, iterations = field.solve(matrix_eq, div, pressure or domain.grid(), solve_params=math.LinearSolve(None, 1e-5))
    gradp = field.gradient(pressure, type=type(velocity_field))
    return velocity_field - gradp, pressure, occupied_staggered


def map_velocity_to_particles(previous_particle_velocity: PointCloud, velocity_grid: Grid, occupation_mask: Grid,
                              previous_velocity_grid: Grid = None) -> PointCloud:
    """
    Maps result of velocity projection on grid back to particles. Provides option to choose between FLIP (particle velocities are
    updated by the change between projected and initial grid velocities) and PIC (particle velocities are replaced by the the 
    projected velocities) method depending on the value of the `initial_v_field`.
    
    Args:
        previous_particle_velocity: PointCloud with particle positions as elements and their corresponding velocities as values
        velocity_grid: Divergence-free velocity grid
        occupation_mask: Binary Grid (same type as `velocity_grid`) indicating which cells hold particles
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


def respect_boundaries(particles: PointCloud, domain: Domain, not_accessible: list, offset: float = 1.0) -> PointCloud:
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
