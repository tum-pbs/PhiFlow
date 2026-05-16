from tqdm import trange
from phi.jax.flow import *


def test_higher_order_kolmogorov():
    math.set_global_precision(64)

    DOMAIN = dict(extrapolation=extrapolation.PERIODIC, bounds=Box(x=2*PI, y=2*PI), x=100, y=100)
    FORCING = CenteredGrid(lambda x, y: vec(x=math.sin(4 * y), y=0), **DOMAIN) + CenteredGrid(Noise(), **DOMAIN) * 0.01
    plot({'Force along X': FORCING['x'], 'Force along Y': FORCING['y']}, same_scale=False)
    vis.savefig('vis/kolmogorov_forcing.jpg')

    def momentum_equation(v, viscosity=0.001):
        advection = advect.finite_difference(v, v, order=6)
        diffusion = diffuse.finite_difference(v, viscosity, order=6)
        return advection + diffusion + FORCING

    @jit_compile
    def rk4_step(v, p, dt):
        return fluid.incompressible_rk4(momentum_equation, v, p, dt, pressure_order=4, pressure_solve=Solve('CG', 1e-5, 1e-5))

    v0 = CenteredGrid(tensor([0, 0], channel(vector='x, y')), **DOMAIN)
    p0 = CenteredGrid(0, **DOMAIN)
    multi_step = lambda *x, **kwargs: iterate(rk4_step, 25, *x, **kwargs)
    v_trj, p_trj = iterate(multi_step, batch(time=100), v0, p0, dt=0.005, range=trange)

    vis.plot(field.curl(v_trj.with_extrapolation(0)), animate='time')
    vis.savefig('vis/kolmogorov.mp4')


if __name__ == '__main__':
    test_higher_order_kolmogorov()
