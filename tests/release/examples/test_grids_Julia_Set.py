from phi.jax.flow import *


def test_julia_set():
    def julia_map(z, counter, c):
        next_z = z ** 2 + c
        counter += abs(z) < 2
        return next_z, counter

    def belongs_to_julia_set(z, c, iter_count: int):
        final_z, final_counter = iterate(julia_map, iter_count, z, 0, c=c)
        return final_counter

    c = 0.7885 * math.exp(1j*math.linspace(0, 2*PI, batch(time=100)))
    sampled = CenteredGrid(lambda re, im: belongs_to_julia_set(re + im*1j, c, 50), re=256, im=256, bounds=Box(re=(-2, 2), im=(-2, 2)))

    plot({"c": vec(re=c.real, im=c.imag), "$J_c$": sampled}, animate='time')
    vis.savefig('vis/julia_set.mp4')


if __name__ == '__main__':
    test_julia_set()
