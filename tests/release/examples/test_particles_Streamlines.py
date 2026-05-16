from tqdm import trange
from phi.jax.flow import *


def test_streamlines():
    domain = Box(x=10, y=10)
    v = StaggeredGrid(Noise(vector='x,y'), 0, domain, x=256, y=256)
    v, _ = fluid.make_incompressible(v)
    plot(v.at_centers().downsample(8).as_points(), size=(4, 3))
    vis.savefig('vis/streamlines_field.jpg')

    @jit_compile
    def move_along_field(x, step_size=.1):
        return advect.points(geom.Point(x), v, step_size, integrator=advect.rk4).center

    x0 = vec(x=5, y=5)
    x_trj = iterate(move_along_field, spatial(iter=50), x0, range=trange)
    plot([v, x_trj], overlay='list', alpha=[.1, 1])
    vis.savefig('vis/streamlines_single.jpg')

    x0 = pack_dims(CenteredGrid(0, 0, domain, x=8, y=8).points, spatial, instance('start_point'))
    x_trj = iterate(move_along_field, spatial(iter=50), x0, range=trange)

    v_diff = x_trj.iter[1:] - x_trj.iter[:-1]
    distance = math.sum(math.vec_length(v_diff), 'iter')
    plot(x_trj, color=distance)
    vis.savefig('vis/streamlines.jpg')


if __name__ == '__main__':
    test_streamlines()
