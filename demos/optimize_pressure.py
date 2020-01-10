# pylint: disable-msg = not-an-iterable
from phi.tf.flow import *
from phi.math.math_util import randn
from phi.viz.plot import PlotlyFigureBuilder


DESCRIPTION = """
This application demonstrates the backpropagation through the pressure solve operation used in simulating incompressible fluids.

The demo Optimizes the velocity of an incompressible fluid in the left half of a closed space to match the target in the right half.

Select the Y component in the UI to see how the target is approached.
"""


class PressureOptimization(LearningApp):

    def __init__(self):
        LearningApp.__init__(self, 'Pressure Optimization', DESCRIPTION, learning_rate=0.1, epoch_size=5)
        # --- Physics ---
        domain = Domain([62, 62], boundaries=CLOSED)
        with self.model_scope():
            initial_velocity = randn(domain.staggered_grid(0).shape) * 0.2
            optimizable_velocity = variable(initial_velocity)
        velocity = optimizable_velocity * mask(box[0:62, 0:31])
        velocity = divergence_free(velocity, domain)
        # --- Target ---
        y, x = np.meshgrid(*[np.arange(-0.5, dim + 0.5) for dim in domain.resolution])
        target_velocity_y = 2 * np.exp(-0.5 * ((x - 40) ** 2 + (y - 10) ** 2) / 32 ** 2)
        target_velocity_y[:, 0:32] = 0
        target_velocity = math.expand_dims(np.stack([target_velocity_y, np.zeros_like(target_velocity_y)], axis=-1), 0)
        target_velocity *= self.editable_float('Target_Direction', 1, [-1, 1], log_scale=False)
        # --- Optimization ---
        loss = math.l2_loss(math.sub(velocity.staggered_tensor()[:, :, 31:, :], target_velocity[:, :, 31:, :]))
        self.add_objective(loss, 'Loss')
        # --- Display ---
        gradient = StaggeredGrid(tf.gradients(loss, [component.data for component in optimizable_velocity.unstack()]))
        self.add_field('Initial Velocity (Optimizable)', optimizable_velocity)
        self.add_field('Gradient', gradient)
        self.add_field('Final Velocity', velocity)
        self.add_field('Target Velocity', target_velocity)


show(display=('Final Velocity', 'Target Velocity'))
