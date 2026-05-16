from tqdm import trange
from phi.torch.flow import *


def test_optimize_throw():
    def simulate_hit(pos, height, vel, angle, gravity=1.):
        vel_x, vel_y = math.cos(angle) * vel, math.sin(angle) * vel
        height = math.maximum(height, .01)
        hit_time = (vel_y + math.sqrt(vel_y**2 + 2 * gravity * height)) / gravity
        return pos + vel_x * hit_time, hit_time, height, vel_x, vel_y

    def sample_trajectory(pos, height, vel, angle, gravity=1., steps=spatial(time=100)):
        _, hit_time, height, vel_x, vel_y = simulate_hit(pos, height, vel, angle, gravity)
        t = math.linspace(0, hit_time, steps)
        return vec(x=pos + vel_x * t, y=height + vel_y * t - gravity / 2 * t ** 2)

    angles = vec('angle', -1, -.5, 0, .5, 1, 1.5)
    trj = sample_trajectory(10, 1, 1, angles)
    plot(trj.time.rename('const'), trj, title="Varying Angle", animate='time', overlay='args', color=math.range(angles.shape), frame_time=40)
    vis.savefig('vis/optimize_throw_angles.mp4')

    @jit_compile
    def loss_function(pos, height, vel, angle, target):
        return math.l2_loss(simulate_hit(pos, height, vel, angle)[0] - target)

    grad_fun = math.gradient(loss_function, wrt='vel', get_output=False)

    def gradient_descent_step(vel, pos, height, angle, target, step_size=.1):
        return vel - step_size * grad_fun(pos, height, vel, angle, target)

    fixed = dict(pos=0, height=1, angle=0)
    vel_trj = iterate(gradient_descent_step, batch(iter=25), 1., target=10, **fixed, range=trange)

    plot(vel_trj.iter.as_spatial(), size=(3, 3), title="Change in Vel")
    vis.savefig('vis/optimize_throw_vel.jpg')

    trj = sample_trajectory(vel=vel_trj, **fixed)
    plot(trj, vec(x=10, y=0), overlay='args', animate='iter', title="Ball trajectory and target", size=(9, 2))
    vis.savefig('vis/optimize_throw.mp4')


if __name__ == '__main__':
    test_optimize_throw()
