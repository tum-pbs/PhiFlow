from tqdm import trange
from phi.torch.flow import *
from phi.physics import sph


def test_sph():
    domain = Box(x=80, y=80, z=80)
    smoothing_length = 0.8
    initial_positions = pack_dims(math.meshgrid(x=25, y=100, z=25), spatial, instance('particles')) * smoothing_length
    initial_positions += 0.001 * smoothing_length * math.random_normal(initial_positions.shape)
    particles = Sphere(initial_positions, volume=smoothing_length**3, radius_variable=False)
    desired_neighbors = sph.expected_neighbors(particles.volume, smoothing_length, 3)

    particle_mass = 0.01 * math.mean(particles.volume)
    dt = 0.01 * smoothing_length
    dynamic_visc = 0.025
    gravity = vec(x=0, y=0, z=-0.1)
    pressure_normalization = -(45.0 * particle_mass) / (PI * smoothing_length**6)
    viscous_normalization = (45.0 * dynamic_visc * particle_mass) / (PI * smoothing_length**6)

    plot([domain, initial_positions], overlay='list', alpha=[.1, .4], size=(3, 3))
    vis.savefig('vis/sph_setup.jpg')

    @jit_compile
    def sph_step(v: Field, dt=dt, isotropic_exp=20., base_density=1.):
        graph = sph.neighbor_graph(v.geometry, 'poly6', desired_neighbors=desired_neighbors, compute='kernel', domain=domain)
        rho = math.sum(graph.edges['kernel'], dual) * particle_mass
        pressure = isotropic_exp * (rho - base_density)
        nb_rho = rename_dims(rho, instance, dual)
        nb_pressure = rename_dims(pressure, instance, dual)
        distance_gaps = smoothing_length - graph.distances
        avg_pressure = (graph.connectivity * pressure + nb_pressure) / (2 * nb_rho)
        pressure_force = -graph.unit_deltas * avg_pressure * distance_gaps ** 2
        dv = math.pairwise_differences(v.values, format=graph.edges)
        viscous_force = dv / nb_rho * distance_gaps
        force = math.sum(pressure_normalization * pressure_force + viscous_normalization * viscous_force, dual)
        a = force / rho + gravity
        v += a * dt
        return v.shifted(dt * v.values)

    @jit_compile
    def apply_bounds(v: Field, damping_coef=-0.95) -> Field:
        clipped = math.clip(v.points, domain.lower, domain.upper)
        v = field.where(clipped == v.points, v, v * damping_coef)
        return v.shifted_to(clipped)

    initial_state = Field(particles, vec(x=0, y=0, z=0), 0)
    trj = iterate(lambda v: apply_bounds(sph_step(v)), batch(t=480), initial_state, substeps=int(32 / smoothing_length), range=trange)

    plot([domain, trj.points.t[::3]], overlay='list', alpha=[.1, .3], animate='t', size=(5, 5), frame_time=50)
    vis.savefig('vis/sph.mp4')

    plot(trj.geometry.t[150]['x,z'], overlay='list', size=(5, 5), frame_time=50)
    vis.savefig('vis/sph_2d_slice.jpg')


if __name__ == '__main__':
    test_sph()
