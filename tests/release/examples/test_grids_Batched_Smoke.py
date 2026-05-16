from tqdm import trange
from phi.jax.flow import *


def test_batched_smoke():
    domain = Box(x=100, y=100)

    settings = batch(setting=3)
    inflow_rate = tensor([.1, .2, .3], settings)
    inflow_x = tensor([40, 50, 60], settings)
    obstacle_x = wrap([15, 50, 70], settings)

    obstacle = Cuboid(vec(x=obstacle_x, y=60), half_size=vec(x=15, y=10))
    inflow = Sphere(x=inflow_x, y=9.5, radius=5)
    plot(obstacle, inflow, overlay='args')
    vis.savefig('vis/batched_smoke_setup.jpg')

    @jit_compile
    def step(v, s, p, dt=1.):
        s = advect.mac_cormack(s, v, dt) + inflow_rate * resample(inflow, to=s, soft=True)
        buoyancy = resample(s * (0, 0.1), to=v)
        v = advect.semi_lagrangian(v, v, dt) + buoyancy * dt
        v, p = fluid.make_incompressible(v, obstacle, Solve(x0=p))
        return v, s, p

    v0 = StaggeredGrid(0, 0, domain, x=64, y=64)
    smoke0 = CenteredGrid(0, ZERO_GRADIENT, domain, x=200, y=200)
    v_trj, s_trj, p_trj = iterate(step, batch(time=100), v0, smoke0, None, range=trange)

    plot(obstacle, inflow, s_trj, animate='time', overlay='args')
    vis.savefig('vis/batched_smoke.mp4')


if __name__ == '__main__':
    test_batched_smoke()
