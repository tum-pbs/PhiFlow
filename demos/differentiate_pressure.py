""" Differentiate through Pressure Solve

This application demonstrates the backpropagation through the pressure solve operation used in simulating incompressible fluids.

The demo Optimizes the velocity of an incompressible fluid in the left half of a closed space to match the TARGET in the right half.
"""
from phi.torch.flow import *
# from phi.jax.flow import *
# from phi.tf.flow import *


DOMAIN = dict(x=80, y=64)
LEFT = StaggeredGrid(Box(x=(-INF, 40), y=None), 0, **DOMAIN)
RIGHT = StaggeredGrid(Box(x=(40, INF), y=None), 0, **DOMAIN)
TARGET = RIGHT * StaggeredGrid(lambda x: math.exp(-0.5 * math.vec_squared(x - (50, 10), 'vector') / 32**2), 0, **DOMAIN) * (0, 2)


def loss(v0, p0):
    v1, p = fluid.make_incompressible(v0 * LEFT, solve=Solve('CG-adaptive', 1e-5, x0=p0))
    return field.l2_loss((v1 - TARGET) * RIGHT), v1, p


eval_grad_v0 = field.functional_gradient(loss, 'v0', get_output=True)
p0 = None
velocity_fit = StaggeredGrid(Noise(), 0, **DOMAIN) * 0.1 * LEFT
viewer = view('incompressible_velocity', TARGET, 'gradient', velocity_fit, 'remaining_divergence', play=False, namespace=globals())

for iteration in viewer.range(warmup=1):
    (loss, incompressible_velocity, pressure_guess), gradient = eval_grad_v0(velocity_fit, p0)
    remaining_divergence = field.divergence(incompressible_velocity)
    viewer.log_scalars(loss=loss)
    velocity_fit -= gradient
