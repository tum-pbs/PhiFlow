from tqdm import trange
from phi.torch.flow import *


def test_reaction_diffusion():
    @jit_compile
    def reaction_diffusion(u, v, du, dv, f, k, dt):
        uvv = u * v**2
        su = du * field.laplace(u) - uvv + f * (1 - u)
        sv = dv * field.laplace(v) + uvv - (f + k) * v
        return u + dt * su, v + dt * sv

    u0 = [
        CenteredGrid(Noise(scale=20, smoothness=1.3), x=100, y=100) * .2 + .1,
        CenteredGrid(lambda x: math.exp(-0.5 * math.sum((x - 50)**2) / 3**2), x=100, y=100),
        CenteredGrid(lambda x: math.cos(math.vec_length(x-50)/3), x=100, y=100) * .5,
    ]
    u0 = stack(u0, batch('initialization'))
    plot(u0)
    vis.savefig('vis/reaction_diffusion_initial.jpg')

    # Maze
    maze = {'du': 0.19, 'dv': 0.05, 'f': 0.06, 'k': 0.062}
    u_trj, v_trj = iterate(reaction_diffusion, batch(time=100), u0, u0, dt=.5, f_kwargs=maze, substeps=20, range=trange)
    plot(u_trj, animate='time')
    vis.savefig('vis/reaction_diffusion_maze.mp4')

    # Coral
    coral = {'du': 0.16, 'dv': 0.08, 'f': 0.06, 'k': 0.062}
    u_trj, v_trj = iterate(reaction_diffusion, batch(time=200), u0, u0, dt=.5, f_kwargs=coral, substeps=20, range=trange)
    plot(u_trj.time[::2], animate='time')
    vis.savefig('vis/reaction_diffusion_coral.mp4')

    # Dots
    dots = {'du': 0.19, 'dv': 0.03, 'f': 0.04, 'k': 0.061}
    u_trj, v_trj = iterate(reaction_diffusion, batch(time=100), u0, u0, dt=.5, f_kwargs=dots, substeps=20, range=trange)
    plot(u_trj, animate='time')
    vis.savefig('vis/reaction_diffusion_dots.mp4')


if __name__ == '__main__':
    test_reaction_diffusion()
