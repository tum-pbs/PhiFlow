from phi import math, field
from phi.math import Tensor
from phi.field import StaggeredGrid, HardGeometryMask, PointCloud
from phi.geom import union, Sphere
from ._effect import Gravity, gravity_tensor


def distribute_points(mask: Tensor, points_per_cell: int = 1, dist: str = 'uniform'):
    indices = math.to_float(math.nonzero(mask, list_dim='points'))
    temp = []
    for _ in range(points_per_cell):
        if dist == 'center':
            temp.append(indices + 0.5)
        elif dist == 'uniform':
            temp.append(indices + (math.random_uniform(indices.shape)))
        else:
            raise NotImplementedError
    return math.concat(temp, dim='points')


def apply_gravity(dt, v_field):
    force = dt * gravity_tensor(Gravity(), v_field.shape.spatial.rank)
    return v_field + force


def get_bcs(domain, obstacles):
    accessible = domain.grid(1 - HardGeometryMask(union([obstacle.geometry for obstacle in obstacles])))
    accessible_mask = domain.grid(accessible, extrapolation=domain.boundaries.accessible_extrapolation)
    return field.stagger(accessible_mask, math.minimum, accessible_mask.extrapolation)


def make_incompressible(v_field, bcs, cmask, smask, pressure):
    v_field, _ = field.extp_sgrid(v_field, smask, 1)  # extrapolation conserves falling shapes
    v_field *= bcs
    div = field.divergence(v_field) * cmask  # cmask prevents falling shape from collapsing

    def laplace(p):
        # TODO: prefactor of pressure should not have any effect, but it has
        return field.where(cmask, field.divergence(field.gradient(p, type=StaggeredGrid) * bcs), -4 * p)

    converged, pressure, iterations = field.solve(laplace, div, pressure, solve_params=math.LinearSolve(None, 1e-5))
    gradp = field.gradient(pressure, type=type(v_field))
    return v_field - gradp, pressure


def map2particle(v_particle, current_v_field, smask, orig_v_field=None):
    if orig_v_field is not None:
        # FLIP
        v_change_field = current_v_field - orig_v_field
        v_change_field, _ = field.extp_sgrid(v_change_field, smask, 1)  # conserves falling shapes (no hard_bcs here!)
        v_change = v_change_field.sample_at(v_particle.elements.center)
        return PointCloud(v_particle.elements, values=v_particle.values + v_change, bounds=v_particle.bounds,
                          add_overlapping=v_particle.add_overlapping, color=v_particle.color)
    else:
        # PIC
        v_div_free_field, _ = field.extp_sgrid(current_v_field, smask, 1)
        v_values = v_div_free_field.sample_at(v_particle.elements.center)
        return PointCloud(v_particle.elements, values=v_values, bounds=v_particle.bounds,
                          add_overlapping=v_particle.add_overlapping, color=v_particle.color)


def add_inflow(particles, inflow_points, inflow_values):
    new_points = math.tensor(math.concat([particles.points, inflow_points], dim='points'), names=['points', 'vector'])
    new_values = math.tensor(math.concat([particles.values, inflow_values], dim='points'), names=['points', 'vector'])
    return PointCloud(Sphere(new_points, 0), add_overlapping=particles.add_overlapping, bounds=particles.bounds,
                      values=new_values, color=particles.color)


def respect_boundaries(domain, obstacles, particles, shift_amount=1):
    points = particles.elements
    for obstacle in obstacles:
        shift = obstacle.geometry.shift_points(points.center, shift_amount=shift_amount)
        points = particles.elements.shifted(shift)
    shift = (~domain.bounds).shift_points(points.center, shift_amount=shift_amount)
    return PointCloud(points.shifted(shift), add_overlapping=particles.add_overlapping,
                      bounds=particles.bounds, values=particles.values, color=particles.color)
