from tqdm import trange
from phi.jax.flow import *


def test_multi_grid_fluid():
    large_domain = Box(x=100, y=100)
    small_domain = Box(x=(30, 70), y=(40, 80))
    obstacle = Sphere(x=50, y=60, radius=5)
    inflow = Sphere(x=50, y=9.5, radius=5)
    inflow_rate = 0.2

    plot(large_domain, small_domain, obstacle, inflow, overlay='args', title="Small domain around obstacle")
    vis.savefig('vis/multi_grid_setup.jpg')

    v0_large = StaggeredGrid(0, 0, large_domain, x=32, y=32)
    v0_small = v0_large.at(StaggeredGrid(0, boundary=v0_large, bounds=small_domain, x=64, y=64))
    smoke = CenteredGrid(0, ZERO_GRADIENT, large_domain, x=200, y=200)

    @jit_compile
    def step(v, v_small, s, p, dt=1.):
        s = advect.mac_cormack(s, v_small, dt) + inflow_rate * resample(inflow, s, soft=True)
        buoyancy = s * (0, 0.1)
        v_small = advect.semi_lagrangian(v_small, v_small, dt) + buoyancy.at(v_small) * dt
        v = advect.semi_lagrangian(v, v, dt) + buoyancy.at(v) * dt
        v, p = fluid.make_incompressible(v, [obstacle], Solve(x0=p))
        p_emb_x0 = CenteredGrid(0, p, v_small.bounds, v_small.resolution)
        v_small = StaggeredGrid(v_small, ZERO_GRADIENT, v_small.bounds, v_small.resolution)
        v_small, p_emb = fluid.make_incompressible(v_small, [obstacle], Solve('auto', 1e-5, 1e-5, x0=p_emb_x0))
        v_small = StaggeredGrid(v_small, v, v_small.bounds, v_small.resolution)
        return v, v_small, s, p

    v_large_trj, v_small_trj, s_trj, p_trj = iterate(step, batch(time=200), v0_large, v0_small, smoke, None, range=trange)

    plot([obstacle, inflow, s_trj], [obstacle, v_small_trj], overlay='list', animate='time')
    vis.savefig('vis/multi_grid_fluid.mp4')

    plot([small_domain, v_small_trj.time[-1], v_large_trj.time[-1]], overlay='list', alpha=[.2, 1, 1])
    vis.savefig('vis/multi_grid_final.jpg')


if __name__ == '__main__':
    test_multi_grid_fluid()
