from tqdm import trange
from phi.jax.flow import *


def test_heat_flow():
    domain = Box(x=10, y=5)
    boundary = {'x-': 1, 'x+': ZERO_GRADIENT, 'y': PERIODIC}
    bars = union(Box(x=(0, 10), y=(2, 3)), Box(x=(4.5, 5.5), y=(1, 4)))
    conductivity = CenteredGrid(bars, ZERO_GRADIENT, domain, x=100, y=50) + .01

    plot({"Geometry": [domain, bars], "Conductivity grid": conductivity}, overlay='list')
    vis.savefig('vis/heat_flow_setup.jpg')

    @jit_compile
    def step(t, dt):
        return diffuse.implicit(t, conductivity, dt)

    t0 = CenteredGrid(0, boundary, domain, x=100, y=50)
    t_trj = iterate(step, batch(time=100), t0, dt=1, range=trange)

    plot(t_trj, animate='time')
    vis.savefig('vis/heat_flow.mp4')


if __name__ == '__main__':
    test_heat_flow()
