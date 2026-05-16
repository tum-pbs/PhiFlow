from tqdm import trange
from phi.jax.flow import *


def test_burgers():
    @jit_compile
    def step(v, dt=.5):
        v = diffuse.implicit(v, 0.1, dt=dt)
        v = advect.semi_lagrangian(v, v, dt=dt)
        return v

    # 1D Simulation
    v0 = CenteredGrid(Noise(smoothness=1.5), PERIODIC, x=64, bounds=Box(x=64))
    v_trj = iterate(step, batch(time=100), v0, range=trange)
    plot(v_trj, animate='time')
    vis.savefig('vis/burgers_1d.mp4')

    # 2D Simulation
    v0 = CenteredGrid(Noise(vector='x,y'), PERIODIC, x=64, y=64, bounds=Box(x=40, y=20))
    v_trj = iterate(step, batch(time=100), v0, range=trange)
    plot(v_trj.as_points(), animate='time')
    vis.savefig('vis/burgers_2d.mp4')


if __name__ == '__main__':
    test_burgers()
