""" Differentiate through Pressure Solve

This application demonstrates the backpropagation through the pressure solve operation used in simulating incompressible fluids.

The demo Optimizes the velocity of an incompressible fluid in the left half of a closed space to match the target in the right half.
"""
from phi.tf.flow import *


domain = Domain(x=62, y=62, boundaries=CLOSED)
left = domain.staggered_grid(Box[:31, :])
right = domain.staggered_grid(Box[31:, :])

velocity = incompressible_velocity = domain.staggered_grid(lambda x: math.random_normal(x.shape.without('vector'))) * 0.2 * left
centers = domain.cells.center
target = 2 * math.exp(-0.5 * ((centers.vector[0] - 40) ** 2 + (centers.vector[1] - 10) ** 2) / 32 ** 2) * (0, 1)
target = domain.staggered_grid(domain.vgrid(target)) * right
pressure_guess = None
for _iter in ModuleViewer(['velocity', 'target', 'incompressible_velocity']).range():
    with GradientTape(watch=[velocity.values]) as tape:
        incompressible_velocity, p, _, _ = fluid.make_incompressible(velocity * left, domain, pressure_guess=pressure_guess)
        loss = field.l2_loss((incompressible_velocity - target) * right)
    grad = gradients(loss, velocity.values, tape)
    velocity -= domain.staggered_grid(grad)
