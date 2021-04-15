""" Differentiate through Pressure Solve

This application demonstrates the backpropagation through the pressure solve operation used in simulating incompressible fluids.

The demo Optimizes the velocity of an incompressible fluid in the left half of a closed space to match the TARGET in the right half.
"""
from phi.torch.flow import *
# from phi.jax.flow import *
# from phi.tf.flow import *


DOMAIN = Domain(x=62, y=62, boundaries=CLOSED)
LEFT = DOMAIN.staggered_grid(HardGeometryMask(Box[:31, :]))
RIGHT = DOMAIN.staggered_grid(HardGeometryMask(Box[31:, :]))
CELL_CENTERS = DOMAIN.cells.center
TARGET = 2 * math.exp(-0.5 * ((CELL_CENTERS.vector[0] - 40) ** 2 + (CELL_CENTERS.vector[1] - 10) ** 2) / 32 ** 2) * (0, 1)
TARGET = DOMAIN.staggered_grid(DOMAIN.vector_grid(TARGET)) * RIGHT


def loss(v0, p0):
    v1, p0, _, _ = fluid.make_incompressible(v0 * LEFT, DOMAIN, pressure_guess=p0)
    return field.l2_loss((v1 - TARGET) * RIGHT), v1, p0


gradient_function = field.functional_gradient(loss, [0], get_output=True)

velocity_fit = DOMAIN.staggered_grid(lambda x: math.random_normal(x.shape.without('vector'))) * 0.2 * LEFT
_pressure_guess = DOMAIN.scalar_grid(0)
incompressible_velocity = DOMAIN.staggered_grid(0)
gradient = DOMAIN.staggered_grid(0)
app = view(incompressible_velocity, gradient, velocity_fit, TARGET, play=False)

for iteration in app.range(warmup=1):
    loss, incompressible_velocity, pressure_guess, gradient = gradient_function(velocity_fit, _pressure_guess)
    app.info(f"Loss = {loss:.2f}")
    app.log_scalars(loss=loss)
    velocity_fit -= gradient
