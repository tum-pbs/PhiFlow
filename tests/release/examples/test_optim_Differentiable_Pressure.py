from tqdm import trange
from phi.jax.flow import *


def test_differentiable_pressure():
    res = dict(x=80, y=64)
    control_area = union(Sphere(x=20, y=15, radius=10), Sphere(x=20, y=50, radius=10))
    control_mask = StaggeredGrid(control_area, 0, **res)
    target_mask = StaggeredGrid(Box(x=(40, INF), y=None), 0, **res)
    target = target_mask * StaggeredGrid(lambda x: math.exp(-0.5 * math.vec_squared(x - (50, 10), 'vector') / 32**2), 0, **res) * (0, 2)
    plot(control_mask['x'], target, overlay='args', size=(4, 3), title='Control areas (left) and target flow (right)')
    vis.savefig('vis/differentiable_pressure_setup.jpg')

    @jit_compile
    def loss(v0):
        v1, p = fluid.make_incompressible(v0 * control_mask)
        return field.l2_loss((v1 - target) * target_mask), v1, p

    grad_fun = field.functional_gradient(loss, wrt='v0', get_output=True)

    def gradient_descent_step(v0, _l, _v, step_size=1.):
        (loss, v, p), dv0 = grad_fun(v0)
        return v0 - step_size * dv0, loss, v

    velocity_fit = StaggeredGrid(Noise(), 0, **res) * 0.1 * control_mask
    ctrl_trj, loss_trj, v_trj = iterate(gradient_descent_step, batch(iter=20), velocity_fit, None, None, range=trange)

    plot(loss_trj.iter.as_spatial(), size=(3, 3), title="Loss")
    vis.savefig('vis/differentiable_pressure_loss.jpg')

    plot(control_area, v_trj, overlay='args', animate='iter', frame_time=200, size=(5, 4))
    vis.savefig('vis/differentiable_pressure.mp4')


if __name__ == '__main__':
    test_differentiable_pressure()
