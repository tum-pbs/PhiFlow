# pylint: disable-msg = not-an-iterable
from phi.tf.flow import *
from phi.math.initializers import randn
from phi.viz.plot import PlotlyFigureBuilder


class PressureOptim(TFApp):

    def __init__(self):
        TFApp.__init__(self, "Pressure Optimization",
                       "Optimize velocity in left half of closed room to match target in right half",
                       stride=100, force_custom_stride=True, learning_rate=0.1)
        # Physics
        domain = Domain([62, 62], SLIPPERY)
        with self.model_scope():
            optimizable_velocity = variable(randn(domain.staggered_shape())*0.2)
        velocity = optimizable_velocity * mask(box[0:62, 0:31])
        velocity = divergence_free(velocity, domain)

        # Target
        y, x = np.meshgrid(*[np.arange(-0.5, dim + 0.5) for dim in domain.resolution])
        target_velocity_y = 2 * np.exp(-0.5 * ((x - 40) ** 2 + (y - 10) ** 2) / 32 ** 2)
        target_velocity_y[:, 0:32] = 0
        target_velocity = math.expand_dims(np.stack([target_velocity_y, np.zeros_like(target_velocity_y)], axis=-1), 0)
        target_velocity *= self.editable_int("Target_Direction", 1, [-1, 1])

        # Optimization
        loss = math.l2_loss(velocity.staggered_tensor()[:, :, 31:, :] - target_velocity[:, :, 31:, :])
        self.add_objective(loss, "Loss")

        self.add_field("Var", optimizable_velocity)
        self.add_field("Final Velocity", velocity)
        self.add_field("Target Velocity", target_velocity)

        self.custom_stride = 100


show(display=("Final Velocity", "Target Velocity"),
     figure_builder=PlotlyFigureBuilder(batches=[0], depths=[0], component='vec2'))
