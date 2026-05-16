from tqdm import trange
from phi.torch.flow import *


def test_gradient_descent():
    def potential(pos):
        return math.cos(math.vec_length(pos))

    landscape = CenteredGrid(potential, x=100, y=100, bounds=Box(x=(-5, 5), y=(-5, 5)))
    plot(landscape, size=(4, 3))
    vis.savefig('vis/gradient_descent_landscape.jpg')

    pot_grad = math.gradient(potential, 'pos', get_output=False)
    plot(landscape.with_values(pot_grad) * .2, size=(3, 3))
    vis.savefig('vis/gradient_descent_gradient.jpg')

    def gradient_descent_step(x):
        return x - .1 * pot_grad(x)

    x0 = vec(x=1, y=0)
    opt_trj = iterate(gradient_descent_step, batch(iter=50), x0, range=trange)
    plot(opt_trj.iter.as_spatial(), size=(5, 2))
    vis.savefig('vis/gradient_descent_trajectory.jpg')

    x0 = rename_dims(landscape.points, spatial, batch)
    opt_trj = iterate(gradient_descent_step, batch(iter=50), x0, range=trange)
    plot([landscape, rename_dims(opt_trj, 'x,y', instance)], animate='iter', color='white', alpha=[1, .2], overlay='list', size=(6, 5))
    vis.savefig('vis/gradient_descent_all.mp4')


if __name__ == '__main__':
    test_gradient_descent()
