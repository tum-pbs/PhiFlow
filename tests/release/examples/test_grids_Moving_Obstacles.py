from tqdm import trange
from phi.jax.flow import *


def test_moving_obstacles():
    domain = Box(x=100, y=100)
    obstacles = [
        Obstacle(Cuboid(vec(x=20, y=80), x=20, y=20), velocity=vec(x=5., y=0)),
        Obstacle(Sphere(x=20, y=20, radius=10), velocity=vec(x=1, y=4)),
    ]

    def move_obstacle(obs: Obstacle, dt):
        x = (obs.geometry.center + obs.velocity * dt) % domain.size
        return obs.at(x)

    move_obstacles = lambda *obstacles, dt: tuple([move_obstacle(o, dt) for o in obstacles])

    obs_trj = iterate(move_obstacles, spatial(time=6), *obstacles, dt=6., range=trange)
    plot([o.geometry for o in obs_trj], alpha=.5, overlay='list')
    vis.savefig('vis/moving_obstacles_path.jpg')

    @jit_compile
    def step(v, p, obs1, obs2, dt=.5):
        obs1, obs2 = move_obstacles(obs1, obs2, dt=dt)
        v = advect.mac_cormack(v, v, dt)
        v, p = fluid.make_incompressible(v, (obs1, obs2), Solve(x0=p))
        return v, p, obs1, obs2

    v0 = StaggeredGrid(0, PERIODIC, domain, x=100, y=100)
    v_trj, p_trj, *obs_trjs = iterate(step, batch(time=130), v0, None, *obstacles, range=trange)

    plot(*[o.geometry for o in obs_trjs], v_trj.curl(), animate='time', overlay='args')
    vis.savefig('vis/moving_obstacles.mp4')


if __name__ == '__main__':
    test_moving_obstacles()
