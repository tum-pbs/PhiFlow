from tqdm import trange
from phi.jax.flow import *


def test_gravity():
    x0 = stack({
        'Sun': vec(x=0, y=0),
        'Earth': vec(x=-10, y=0),
        'Mars': vec(x=0, y=12)
    }, instance('planets'))
    mass = wrap([1000, 10, 10], instance(x0))
    color = wrap(['#fcd700', '#006dfc', '#fc2e00'], instance(x0))
    plot(Sphere(x0, radius=mass**(1/3)*.15), color=color)
    vis.savefig('vis/gravity_setup.jpg')

    @jit_compile
    def step(x, v, dt=.5):
        dx = math.pairwise_differences(x)
        a = .01 * math.sum(math.safe_div(mass.planets.as_dual() * dx, math.vec_squared(dx) ** 1.5), '~planets')
        return x + v * dt, v + a * dt

    v0 = math.safe_div(math.rotate_vector(x0, PI/2), math.vec_length(x0))
    x_trj, v_trj = iterate(step, batch(time=100), tensor(x0), v0, range=trange)
    plot(Sphere(x_trj, radius=mass**(1/3)*.2), color=color, animate='time')
    vis.savefig('vis/gravity.mp4')


if __name__ == '__main__':
    test_gravity()
