""" Differentiate through Advection

This application demonstrates the backpropagation through an advection operation.

The demo Optimizes a velocity field so that a passively advected marker quantity matches the specified target density after advection.
"""
from phi.torch.flow import *
# from phi.jax.flow import *
# from phi.tf.flow import *


DOMAIN = Domain(x=50, y=50, boundaries=CLOSED, bounds=Box[0:100, 0:100])
MARKER_0 = DOMAIN.scalar_grid(Sphere((40, 50), radius=20))
MARKER_TARGET = DOMAIN.scalar_grid(Sphere((60, 50), radius=20))


def loss(velocity):
    advected = advect.mac_cormack(MARKER_0, velocity, dt=1.0)
    smooth_diff = diffuse.explicit(advected - MARKER_TARGET, 10.0, 1, 50)
    return field.l2_loss(smooth_diff), advected, smooth_diff


gradient_function = field.functional_gradient(loss, get_output=True)

velocity_fit = DOMAIN.staggered_grid(0)
marker_fit = DOMAIN.scalar_grid(0)
smooth_difference = DOMAIN.scalar_grid(0)
gradient = DOMAIN.staggered_grid(0)
app = view(display=['marker_fit', 'gradient'], play=False)

for iteration in app.range(warmup=1):
    (loss, marker_fit, smooth_difference), (gradient,) = gradient_function(velocity_fit)
    app.info(f"Loss = {loss:.2f}")
    app.log_scalars(loss=loss)
    velocity_fit -= gradient
