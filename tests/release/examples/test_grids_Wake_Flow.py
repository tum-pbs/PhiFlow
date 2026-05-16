from tqdm import trange
from phi.torch.flow import *


def test_wake_flow():
    cylinder = geom.infinite_cylinder(x=20, y=50, radius=10, inf_dim='z')
    plot({"Top view": cylinder['x,y'], "Side view": cylinder['x,z']})
    vis.savefig('vis/wake_flow_cylinder.jpg')

    @jit_compile
    def step(v, p, dt=1.):
        v = advect.semi_lagrangian(v, v, dt)
        return fluid.make_incompressible(v, cylinder, Solve(x0=p))

    boundary = {'x-': vec(x=2, y=0, z=0), 'x+': ZERO_GRADIENT, 'y': PERIODIC, 'z': PERIODIC}
    v0 = StaggeredGrid((8., 0, 0), boundary, x=128, y=64, z=8, bounds=Box(x=200, y=100, z=5))
    v0, p0 = fluid.make_incompressible(v0, cylinder, Solve('scipy-direct'))
    v_trj, p_trj = iterate(step, batch(time=200), v0, p0, range=trange)

    v_trj_2d = v_trj[{'z': 4, 'vector': 'x,y'}]
    plot(v_trj_2d.time[100:].curl(), animate='time')
    vis.savefig('vis/wake_flow_vorticity.mp4')

    plot(v_trj_2d.time[-1], cylinder['x,y'], overlay='args')
    vis.savefig('vis/wake_flow_final.jpg')


if __name__ == '__main__':
    test_wake_flow()
