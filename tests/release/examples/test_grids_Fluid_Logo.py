from tqdm import trange
from phi.jax.flow import *


def test_fluid_logo():
    domain = dict(x=128, y=128, bounds=Box(x=100, y=100))
    geometries = [Box(x=(15 + x * 7, 15 + (x + 1) * 7), y=(41, 83)) for x in range(1, 10, 2)] + [Box['x,y', 43:50, 41:48], Box['x,y', 15:43, 83:90], Box['x,y', 50:85, 83:90]]
    geometry = union(geometries)

    plot(geometry, CenteredGrid(geometry, 0, **domain))
    vis.savefig('vis/fluid_logo_geometry.jpg')

    inflow = CenteredGrid(Box(x=(14, 21), y=(6, 10)), ZERO_GRADIENT, **domain) + \
             CenteredGrid(Box(x=(81, 88), y=(6, 10)), ZERO_GRADIENT, **domain) * 0.9 + \
             CenteredGrid(Box(x=(44, 47), y=(49, 51)), ZERO_GRADIENT, **domain) * 0.4
    plot(inflow)
    vis.savefig('vis/fluid_logo_inflow.jpg')

    @jit_compile
    def step(smoke, v, pressure, inflow, dt=1.):
        smoke = advect.semi_lagrangian(smoke, v, 1) + inflow
        buoyancy_force = resample(smoke * (0, 0.1), to=v)
        v = advect.semi_lagrangian(v, v, 1) + buoyancy_force
        v, pressure = fluid.make_incompressible(v, geometry, Solve('CG-adaptive', 1e-5, x0=pressure))
        return smoke, v, pressure

    v0 = StaggeredGrid(0, boundary=0, **domain)
    smoke0 = CenteredGrid(0, boundary=ZERO_GRADIENT, **domain)
    smoke_trj, v_trj, pressure_trj = iterate(step, batch(time=200), smoke0, v0, None, inflow=inflow, range=trange)

    plot(smoke_trj.time[::2], animate='time')
    vis.savefig('vis/fluid_logo.mp4')


if __name__ == '__main__':
    test_fluid_logo()
