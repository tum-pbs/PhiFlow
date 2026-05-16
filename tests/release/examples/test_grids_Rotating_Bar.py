from tqdm import trange
from phi.jax.flow import *


def test_rotating_bar():
    bar = Obstacle(Cuboid(vec(x=50, y=50), x=6, y=60), angular_velocity=0.05)
    v0 = StaggeredGrid(0, ZERO_GRADIENT, Box(x=100, y=100), x=100, y=100)
    plot(v0, bar.geometry, overlay='args')
    vis.savefig('vis/rotating_bar_setup.jpg')

    @jit_compile
    def step(v, p, bar: Obstacle, dt=1.):
        bar = bar.rotated(bar.angular_velocity * dt)
        v = advect.mac_cormack(v, v, dt)
        v, p = fluid.make_incompressible(v, bar, Solve(x0=p))
        return v, p, bar

    v_trj, p_trj, bar_trj = iterate(step, batch(time=100), v0, None, bar, range=trange)

    plot(dict(**v_trj.vector), bar_trj.geometry, overlay='args', animate='time')
    vis.savefig('vis/rotating_bar.mp4')


if __name__ == '__main__':
    test_rotating_bar()
