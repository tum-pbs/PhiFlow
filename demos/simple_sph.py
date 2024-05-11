""" This is a re-implementation of NVIDIA warp's SPH example from
https://github.com/NVIDIA/warp/blob/3ed2ceab824b65486c5204d2a7381d37b79fc314/warp/examples/core/example_sph.py

Reference Publication
Matthias Müller, David Charypar, and Markus H. Gross.
"Particle-based fluid simulation for interactive applications."
Symposium on Computer animation. Vol. 2. 2003.
"""
import time

from phi.torch.flow import *
from phi.physics import sph


domain = Box(x=80, y=80, z=80)
smoothing_length = 0.8  # copied from warp, we would usually set desired_neighbors instead.
initial_positions = pack_dims(math.meshgrid(x=25, y=100, z=25), spatial, instance('particles')) * smoothing_length
initial_positions += 0.001 * smoothing_length * math.random_normal(initial_positions.shape)
particles = Sphere(initial_positions, volume=smoothing_length**3, radius_variable=False)
desired_neighbors = sph.expected_neighbors(particles.volume, smoothing_length, 3)

particle_mass = 0.01 * math.mean(particles.volume)
dt = 0.01 * smoothing_length
dynamic_visc = 0.025
damping_coef = -0.95
gravity = vec(x=0, y=0, z=-0.1)
pressure_normalization = -(45.0 * particle_mass) / (PI * smoothing_length**6)
viscous_normalization = (45.0 * dynamic_visc * particle_mass) / (PI * smoothing_length**6)


@jit_compile
def sph_step(v: Field, dt, isotropic_exp=20., base_density=1.):
    graph = sph.neighbor_graph(v.geometry, 'poly6', desired_neighbors=desired_neighbors, compute='kernel', domain=domain)
    # --- compute density and pressure ---
    rho = math.sum(graph.edges['kernel'], dual) * particle_mass  # this includes the density contribution from self
    pressure = isotropic_exp * (rho - base_density)
    nb_rho = rename_dims(rho, instance, dual)
    nb_pressure = rename_dims(pressure, instance, dual)  # warp re-computes this from nb_rho
    distance_gaps = smoothing_length - graph.distances
    # --- pressure force ---
    avg_pressure = (graph.connectivity * pressure + nb_pressure) / (2 * nb_rho)
    pressure_force = -graph.unit_deltas * avg_pressure * distance_gaps ** 2
    # --- viscosity force ---
    dv = math.pairwise_differences(v.values, format=graph.edges)
    viscous_force = dv / nb_rho * distance_gaps
    # --- sum forces, integrate ---
    force = math.sum(pressure_normalization * pressure_force + viscous_normalization * viscous_force, dual)
    a = force / rho + gravity
    v += a * dt  # kick
    return v.shifted(dt * v.values)


@jit_compile
def apply_bounds(v: Field, damping_coef=-0.95) -> Field:
    clipped = math.clip(v.points, domain.lower, domain.upper)
    v = field.where(clipped == v.points, v, v * damping_coef)
    return v.shifted_to(clipped)


def compute_frame(v: Field, steps: int):
    for i in range(steps):
        t0 = time.perf_counter()
        v = sph_step(v, dt)
        v = apply_bounds(v)
        print(i, time.perf_counter() - t0)
    return v


initial_state = Field(particles, vec(x=0, y=0, z=0), 0)
trj = iterate(compute_frame, batch(t=480), initial_state, steps=int(32 / smoothing_length))
plot(trj.elements['x,z'], animate='t')
print("Creating figure...")
vis.savefig('simple_sph.mp4')
print("figure saved")
