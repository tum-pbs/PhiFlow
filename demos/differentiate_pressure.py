""" Differentiate through Pressure Solve

This application demonstrates the backpropagation through the pressure solve operation used in simulating incompressible fluids.

The demo Optimizes the velocity of an incompressible fluid in the left half of a closed space to match the target in the right half.
"""
from phi.tf.flow import *
# from phi.torch.flow import *


DOMAIN = Domain(x=62, y=62, boundaries=CLOSED)
LEFT = DOMAIN.staggered_grid(HardGeometryMask(Box[:31, :]))
RIGHT = DOMAIN.staggered_grid(HardGeometryMask(Box[31:, :]))
CELL_CENTERS = DOMAIN.cells.center

velocity = incompressible_velocity = DOMAIN.staggered_grid(lambda x: math.random_normal(x.shape.without('vector'))) * 0.2 * LEFT
target = 2 * math.exp(-0.5 * ((CELL_CENTERS.vector[0] - 40) ** 2 + (CELL_CENTERS.vector[1] - 10) ** 2) / 32 ** 2) * (0, 1)
target = DOMAIN.staggered_grid(DOMAIN.vector_grid(target)) * RIGHT
pressure_guess = None
for _iter in ModuleViewer(['velocity', 'target', 'incompressible_velocity']).range():
    with math.record_gradients(velocity.values):
        incompressible_velocity, pressure_guess, _, _ = fluid.make_incompressible(velocity * LEFT, DOMAIN, pressure_guess=pressure_guess)
        loss = field.l2_loss((incompressible_velocity - target) * RIGHT)
        grad = math.gradients(loss)
    velocity -= DOMAIN.staggered_grid(grad)
