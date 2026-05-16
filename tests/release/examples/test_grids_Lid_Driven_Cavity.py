from tqdm import trange
from phi.jax.flow import *


def test_lid_driven_cavity():
    @jit_compile
    def step(v, p, dt=1., viscosity=.1):
        v = advect.semi_lagrangian(v, v, dt)
        v = diffuse.explicit(v, viscosity, dt)
        v, p = fluid.make_incompressible(v, solve=Solve(x0=p))
        return v, p

    boundary = {'x': 0, 'y-': 0, 'y+': vec(x=1, y=0)}
    v0 = StaggeredGrid(0, boundary, x=50, y=32)
    v_trj, p_trj = iterate(step, batch(time=300), v0, None, range=trange)

    frames = v_trj.time[::8]
    plot(frames, animate='time')
    vis.savefig('vis/lid_driven_cavity.mp4')


if __name__ == '__main__':
    test_lid_driven_cavity()
