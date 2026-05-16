from tqdm import trange
from phi.torch.flow import *


def test_fvm_heat():
    valid = union(
        Box(x=(0, .4), y=(.45, .55)),
        Box(x=(.3, .7), y=(0, .1)),
        Box(x=(.3, .7), y=(.9, 1)),
        Box(x=(.3, .4), y=1),
        Box(x=(.6, .7), y=1),
        Box(x=(.6, 1), y=(.45, .55)),
    )
    mesh = geom.build_mesh(Box(x=1, y=1), x=100, y=100, obstacles=~valid)
    plot(valid, mesh)
    vis.savefig('vis/fvm_heat_mesh.jpg')

    @jit_compile
    def step(t, dt, conductivity=1.):
        return diffuse.implicit(t, conductivity, dt, correct_skew=False)

    boundary = {'x-': 1, 'x+': ZERO_GRADIENT, 'y': ZERO_GRADIENT, 'obstacle': ZERO_GRADIENT}
    t0 = Field(mesh, tensor(0), boundary)
    t_trj = iterate(step, batch(time=100), t0, dt=.01, range=trange)

    plot(t_trj, animate='time', frame_time=50)
    vis.savefig('vis/fvm_heat.mp4')


if __name__ == '__main__':
    test_fvm_heat()
