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
    y = tf.layers.dense(y, 1, activation=None, name='layer_out', reuse=tf.AUTO_REUSE)
    return y


def f(u, x, t):
    """ Physics loss function based on Burgers equation """
    u_t = gradients(u, t)
    u_x = gradients(u, x)
    u_xx = gradients(u_x, x)
    return u_t + u*u_x - (0.05 / np.pi) * u_xx


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
N_bc = 50
x, t, u_bc = [math.concat([v_t0, v_x], axis=0) for v_t0, v_x in zip(boundary_t0(N_bc), open_boundary(N_bc))]
with app.model_scope():
    u = network(x, t)
loss_u = math.l2_loss(u[:, 0] - u_bc)

# Physics loss
N_f = 31 * 128
x = tf.convert_to_tensor(rnd.random_uniform([N_f], -1, 1))
t = tf.convert_to_tensor(rnd.random_uniform([N_f], 0, 1))
with app.model_scope():
    u = network(x, t)
loss_f = math.l2_loss(f(u[:, 0], x, t))

app.add_objective(loss_u, reg=loss_f * 0.32)

# Display u on a grid
grid_x, grid_t = np.meshgrid(np.linspace(-1, 1, 128), np.linspace(0, 1, 33), indexing='ij')
grid_x = tf.convert_to_tensor(grid_x, tf.float32)
grid_t = tf.convert_to_tensor(grid_t, tf.float32)
with app.model_scope():
    grid_u = math.expand_dims(network(grid_x, grid_t))
app.add_field('u', grid_u)
app.add_field('u0', grid_u[:, :, 0, :])

app.action_reset = lambda: app.load_model(init_weights)
app.prepare()
init_weights = app.save_model()
show(app)
