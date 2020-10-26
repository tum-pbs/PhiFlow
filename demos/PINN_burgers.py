"""
Physics-informed neural networks (PINN) in Φ-Flow

This is an implementation of the Burgers example from https://arxiv.org/pdf/1711.10561.pdf
"""
from phi.tf.flow import *


app = LearningApp('Physics-informed Burgers')
rnd = TF_BACKEND  # sample different points each iteration
# rnd = math.choose_backend(1)  # use same random points for all iterations


def network(x, t):
    """ Dense neural network with 3021 parameters """
    y = math.stack([x, t], axis=-1)
    for i in range(8):
        y = tf.layers.dense(y, 20, activation=tf.math.tanh, name='layer%d' % i, reuse=tf.AUTO_REUSE)
    return tf.layers.dense(y, 1, activation=None, name='layer_out', reuse=tf.AUTO_REUSE)


def f(u, x, t):
    """ Physics loss function based on Burgers equation """
    u_t = gradients(u, t)
    u_x = gradients(u, x)
    u_xx = gradients(u_x, x)
    return u_t + u*u_x - (0.01 / np.pi) * u_xx


def boundary_t0(N):
    x = rnd.random_uniform([N], -1, 1)
    t = rnd.zeros_like(x)
    u = - math.sin(np.pi * x)
    return x, t, u


def open_boundary(N):
    t = rnd.random_uniform([N], 0, 1)
    x = math.concat([math.zeros([N//2]) + 1, math.zeros([N//2]) - 1], axis=0)
    u = math.zeros([N])
    return x, t, u


# Boundary loss
x_bc, t_bc, u_bc = [math.concat([v_t0, v_x], axis=0) for v_t0, v_x in zip(boundary_t0(100), open_boundary(100))]
with app.model_scope():
    loss_u = math.l2_loss(network(x_bc, t_bc)[:, 0] - u_bc)  # normalizes by first dimension, N_bc

# Physics loss
x_ph, t_ph = tf.convert_to_tensor(rnd.random_uniform([1000], -1, 1)), tf.convert_to_tensor(rnd.random_uniform([1000], 0, 1))
with app.model_scope():
    loss_ph = math.l2_loss(f(network(x_ph, t_ph)[:, 0], x_ph, t_ph))  # normalizes by first dimension, N_ph

app.add_objective(loss_u, reg=loss_ph)  # allows us to control the influence of loss_ph as a slider in the interface

# Display u on a grid
grid_x, grid_t = [tf.convert_to_tensor(t, tf.float32) for t in np.meshgrid(np.linspace(-1, 1, 128), np.linspace(0, 1, 33), indexing='ij')]
with app.model_scope():
    grid_u = math.expand_dims(network(grid_x, grid_t))
app.add_field('u', grid_u)
app.add_field('u0', grid_u[:, :, 0, :])

app.action_reset = lambda: app.load_model(init_weights)  # Adds "Reset" button to the interface
app.prepare()  # Initializes variables, adds controls
init_weights = app.save_model()
show(app)  # launches web interface
