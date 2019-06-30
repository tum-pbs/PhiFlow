from phi.tf.flow import *


class PressureOptim(TFModel):
    def __init__(self, size=(64, 64)):
        TFModel.__init__(self, "Pressure Optimization",
                         "Optimize velocity in left half of closed room to match target in right half",
                         stride=100, learning_rate=0.1)
        # Physics
        with self.model_scope():
            optimizable_velocity = tf.Variable(tf.random_normal([1, 63, 32, 2]) * 0.2, dtype=tf.float32)
        self.reset_velocity = optimizable_velocity.assign(tf.random_normal([1, 63, 32, 2]))
        velocity = StaggeredGrid(tf.concat([optimizable_velocity, np.zeros([1, 63, 31, 2], np.float32)], axis=-2))
        velocity = velocity.pad(1, 1, "constant")
        final_velocity = divergence_free(velocity, DomainCache(Domain(size, SLIPPERY)))

        # Target
        y, x = np.meshgrid(*[np.arange(-0.5, dim + 0.5) for dim in size])
        target_velocity_y = 2 * np.exp(-0.5 * ((x - 40) ** 2 + (y - 10) ** 2) / 32 ** 2)
        target_velocity_y[:, 0:32] = 0
        target_velocity = expand_dims(np.stack([target_velocity_y, np.zeros_like(target_velocity_y)], axis=-1), 0)
        target_velocity = StaggeredGrid(tf.constant(target_velocity, tf.float32) * self.editable_int("Target_Direction", 1, (-1,1)))

        # Optimization
        loss = l2_loss(final_velocity[:, :, 33:, :] - target_velocity[:, :, 33:, :])
        self.add_objective("Loss", loss)

        self.add_field("Velocity n", velocity)
        self.add_field("Final Velocity", final_velocity)
        self.add_field("Target Velocity", target_velocity)

    def action_reset(self):
        self.session.run(self.reset_velocity)
        self.time = 0


app = PressureOptim().show(display=("Final Velocity", "Target Velocity"), production=__name__!="__main__")
