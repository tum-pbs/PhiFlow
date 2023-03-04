""" Neural Network Training Demo
Trains a U-Net to make velocity fields incompressible.
This script can be run with PyTorch, TensorFlow and Jax by selecting the corresponding import statement.
"""
from phi.torch.flow import *
# from phi.tf.flow import *
# from phi.jax.stax.flow import *


math.seed(0)  # Make the results reproducible
net = u_net(2, 2)  # for a fully connected network, use   net = dense_net(2, 2, [64, 64, 64])
optimizer = adam(net, 1e-3)


def loss_function(x):
    y = field.native_call(net, x)
    div = field.divergence(y)
    divergence_loss = field.l2_loss(div)
    similarity_loss = field.l2_loss(y - x)
    return divergence_loss + similarity_loss, divergence_loss, similarity_loss, y, div


@vis.action  # make this function callable from the user interface
def save_model(i=None):
    i = i if i is not None else step + 1
    save_state(net, viewer.scene.subpath(f"net_{i}"))
    save_state(optimizer, viewer.scene.subpath(f"opt_{type(optimizer).__name__}_{i}"))
    viewer.info(f"Model at {i} saved to {viewer.scene.path}.")


@vis.action
def reset():
    math.seed(0)
    load_state(net, viewer.scene.subpath('net_0'))
    load_state(optimizer, viewer.scene.subpath(f"opt_{type(optimizer).__name__}_0"))
    viewer.info(f"Model loaded.")


prediction = CenteredGrid((0, 0), ZERO_GRADIENT, x=64, y=64)
viewer = view('divergence', scene=True, namespace=globals(), select='batch')
save_model(0)
reset()  # Ensure that the first run will be identical to every time reset() is called


for step in viewer.range():
    data = CenteredGrid(Noise(batch(batch=8), channel(vector=2)), ZERO_GRADIENT, x=64, y=64)
    loss, div_loss, sim_loss, prediction, divergence = update_weights(net, optimizer, loss_function, data)
    viewer.log_scalars(loss=loss, divergence_loss=div_loss, similarity_loss=sim_loss)
    if (step + 1) % 100 == 0:
        save_model()
