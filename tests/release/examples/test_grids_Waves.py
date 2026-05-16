from tqdm import trange
from phi.jax.flow import *


def test_waves():
    def wave_displace(sphere: Sphere, *fields: Field, mag=.5, t=-math.PI * 0.5):
        return [field.where(sphere, mag * math.sin(t), f) for f in fields]

    @jit_compile
    def step(h_c, h_p, time, dt=1/60./16, k_speed=1.0, k_damp=0.0):
        sphere = Sphere(center=h_c.bounds.center + math.rotate_vector(vec(x=0, y=-12.8/3), time), radius=1.)
        h_c, h_p = wave_displace(sphere, h_c, h_p)
        h_n = 2.0 * h_c - h_p + dt * dt * (k_speed * h_c.laplace() - k_damp * (h_c - h_p))
        return h_n, h_c, time + dt

    h_initial = CenteredGrid(x=128, y=128, bounds=Box(x=12.8, y=12.8), boundary=ZERO_GRADIENT)
    h_trj, *_ = iterate(step, batch(time=300), h_initial, h_initial, 0, substeps=16, range=trange)

    plot(h_trj.time[::3], animate='time', size=(5, 4))
    vis.savefig('vis/waves.mp4')


if __name__ == '__main__':
    test_waves()
