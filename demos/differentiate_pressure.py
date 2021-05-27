""" Differentiate through Pressure Solve

This application demonstrates the backpropagation through the pressure solve operation used in simulating incompressible fluids.

The demo Optimizes the velocity of an incompressible fluid in the left half of a closed space to match the TARGET in the right half.
"""
# from phi.torch.flow import *
# from phi.jax.flow import *
from phi.tf.flow import *


DOMAIN = Domain(x=80, y=64, boundaries=CLOSED)
LEFT = DOMAIN.staggered_grid(HardGeometryMask(Box[:40, :]))
RIGHT = DOMAIN.staggered_grid(HardGeometryMask(Box[40:, :]))
CELL_CENTERS = DOMAIN.cells.center
TARGET = 2 * math.exp(-0.5 * ((CELL_CENTERS.vector[0] - 50) ** 2 + (CELL_CENTERS.vector[1] - 10) ** 2) / 32 ** 2) * (0, 1)
TARGET = DOMAIN.staggered_grid(DOMAIN.vector_grid(TARGET)) * RIGHT


def loss(v0, p0):
    v1, p = fluid.make_incompressible(v0 * LEFT, DOMAIN, solve=Solve('CG-adaptive', 1e-5, 0, x0=p0))
    return field.l2_loss((v1 - TARGET) * RIGHT), v1, p


eval_grad = field.functional_gradient(loss, [0], get_output=True)
p0 = None
velocity_fit = DOMAIN.staggered_grid(Noise()) * 0.1 * LEFT
viewer = view('incompressible_velocity', TARGET, 'gradient', velocity_fit, 'remaining_divergence', play=False)

for iteration in viewer.range(warmup=1):
    (loss, incompressible_velocity, pressure_guess), (gradient,) = eval_grad(velocity_fit, p0)
    remaining_divergence = field.divergence(incompressible_velocity)
    viewer.log_scalars(loss=loss)
    velocity_fit -= gradient
