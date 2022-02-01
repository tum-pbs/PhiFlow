""" TensorFlow Network Training Demo
Trains a simple CNN to make velocity fields incompressible.
This script runs for a certain number of steps before saving the trained network and halting.
"""
from phi.tf.flow import *


math.seed(0)  # Make the results reproducible
net = u_net(2, 2)  # for a fully connected network, use   net = dense_net(2, 2, [64, 64, 64])
optimizer = keras.optimizers.Adam(1e-3)


@vis.action  # make this function callable from the user interface
def save_model(step):
    path = viewer.scene.subpath(f"net_{step}.h5")
    net.save_weights(path)
    viewer.info(f"Model saved to {path}.")


@vis.action
def reset():
    math.seed(0)
    net.load_weights(viewer.scene.subpath(f"net_0.h5"))
    global optimizer
    optimizer = keras.optimizers.Adam(1e-3)


prediction = CenteredGrid((0, 0), extrapolation.BOUNDARY, x=64, y=64)
prediction_div = CenteredGrid(0, 0, x=64, y=64)
viewer = view(scene=True, namespace=globals(), select='batch')
save_model(0)
reset()  # Ensure that the first run will be identical to every time reset() is called

for step in viewer.range():
    # Load or generate training data
    data = CenteredGrid(Noise(batch(batch=8), channel(vector=2)), extrapolation.BOUNDARY, x=64, y=64)
    with tf.GradientTape() as tape:
        # Prediction
        prediction = field.native_call(net, data)  # calls net with shape (BATCH_SIZE, channels, spatial...)
        # Simulation
        prediction_div = field.divergence(prediction)
        # Define loss and compute gradients
        loss = field.l2_loss(prediction_div) + field.l2_loss(prediction - data)
        gradients = tape.gradient(loss.mean, net.trainable_variables)
    # Show curves in user interface
    viewer.log_scalars(loss=loss, div=field.mean(abs(prediction_div)), distance=math.vec_abs(field.mean(abs(prediction - data))))
    # Compute gradients and update weights
    optimizer.apply_gradients(zip(gradients, net.trainable_variables))
    if (step + 1) % 100 == 0:
        save_model(step + 1)
