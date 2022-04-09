""" Kuramoto–Sivashinsky Equation
Simulates the KS equation in one dimension.
Supports PyTorch, TensorFlow and Jax; select backend via import statement.
"""
from phi.torch.flow import *
# from phi.tf.flow import *
# from phi.jax.flow import *


dt = .25


def kuramoto_sivashinsky(u: Grid):
    # --- Operators in Fourier space ---
    frequencies = math.fftfreq(u.resolution) / u.dx
    lin_op = frequencies ** 2 - (1j * frequencies) ** 4  # Fourier operator for linear terms. You'd think that 1j**4 == 1 but apparently the rounding errors have a major effect here even with FP64...
    inv_lin_op = math.divide_no_nan(1, lin_op)  # Removes f=0 component but there is no noticeable difference
    exp_lin_op = math.exp(lin_op * dt)  # time evolution operator for linear terms in Fourier space
    # --- RK2 for non-linear terms, exponential time-stepping for linear terms ---
    non_lin_current = -0.5j * frequencies * math.fft(u.values ** 2)
    u_intermediate = exp_lin_op * math.fft(u.values) + non_lin_current * (exp_lin_op - 1) * inv_lin_op  # intermediate for RK2
    non_lin_intermediate = -0.5j * frequencies * math.fft(math.ifft(u_intermediate).real ** 2)
    u_new = u_intermediate + (non_lin_intermediate - non_lin_current) * (exp_lin_op - 1 - lin_op * dt) * (1 / dt * inv_lin_op ** 2)
    return u.with_values(math.ifft(u_new).real.vector['x'])


def initial(x: math.Tensor):
    return math.cos(x) - 0.1 * math.cos(x / 16) * (1 - 2 * math.sin(x / 16))


trajectory = [CenteredGrid(initial, x=128, bounds=Box(x=22))]
for i in range(1000):
    print(f"Step {i}: max value {trajectory[-1].values.max}")
    trajectory.append(kuramoto_sivashinsky(trajectory[-1]).vector['x'])
trajectory = field.stack(trajectory, spatial('time'), Box(time=len(trajectory) * dt))
vis.show(trajectory.vector[0], aspect='auto', size=(8, 6))

