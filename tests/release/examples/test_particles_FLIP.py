from tqdm import trange
from phi.jax.flow import *
from phi.field._point_cloud import distribute_points


def test_flip():
    domain = Box(x=64, y=64)
    obstacle = Box(x=(1, 25), y=(30, 33)).rotated(-20)
    initial_particles = distribute_points(union(Box(x=(15, 30), y=(50, 60)), Box(x=None, y=(-INF, 5))), x=64, y=64) * (0, 0)
    plot(initial_particles.geometry, obstacle, overlay='args')
    vis.savefig('vis/flip_setup.jpg')

    @jit_compile
    def step(particles: Field, pressure=None, dt=.1, gravity=vec(x=0, y=-9.81)):
        grid_v = prev_grid_v = field.finite_fill(particles.at(StaggeredGrid(0, 0, domain, x=64, y=64), scatter=True, outside_handling='clamp'))
        occupied = resample(field.mask(particles), CenteredGrid(0, grid_v.extrapolation.spatial_gradient(), grid_v.bounds, grid_v.resolution), scatter=True, outside_handling='clamp')
        grid_v, pressure = fluid.make_incompressible(grid_v + gravity * dt, [obstacle], active=occupied)
        particles += resample(grid_v - prev_grid_v, to=particles)
        particles = advect.points(particles, grid_v * resample(~obstacle, to=grid_v), dt, advect.finite_rk4)
        particles = fluid.boundary_push(particles, [obstacle, ~domain], separation=.5)
        return particles, pressure

    part_trj, p_trj = iterate(step, batch(time=100), initial_particles, None, substeps=2, range=trange)

    plot(resample(p_trj, to=part_trj.time[:-1]), obstacle, overlay='args', animate='time')
    vis.savefig('vis/flip.mp4')


if __name__ == '__main__':
    test_flip()
