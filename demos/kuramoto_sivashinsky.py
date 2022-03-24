""" Kuramoto–Sivashinsky Equation
Simulates the KS equation in one dimension.
Supports PyTorch, TensorFlow and Jax; select backend via import statement.
"""
from phi.torch.flow import *
# from phi.tf.flow import *
# from phi.jax.flow import *


RESOLUTION = spatial(x=128)
SIZE = 22
dt = .25
dx = SIZE / RESOLUTION.size

# Matrices for Linear Multistep and semi-implicit schemes
wave_numbers = math.fftfreq(RESOLUTION) * 1j / dx
L_mat = - wave_numbers ** 2 - wave_numbers ** 4
A_mat = math.ones(RESOLUTION) + dt * 0.5 * L_mat
B_mat = 1 / (math.ones(RESOLUTION) - dt * 0.5 * L_mat)
# Matrices for exp. time stepping
exp_lin = math.exp(L_mat * dt)
L_mat_inv = math.divide_no_nan(1, math.concat([math.ones(spatial(x=1)), L_mat.x[1:]], spatial('x')))


def etrk2(u):  # RK time stepping
    nonlin_current = calc_nonlinear(u)
    u_interm = exp_lin * math.fft(u) + nonlin_current * (exp_lin - 1) * L_mat_inv
    u_new = u_interm + (calc_nonlinear(math.ifft(u_interm).real) - nonlin_current) * (exp_lin - 1 - L_mat * dt) * (1 / dt * L_mat_inv ** 2)
    return math.ifft(u_new).real


def calc_nonlinear(u):
    return -0.5 * wave_numbers * math.fft(u ** 2)


x = math.range(RESOLUTION) * SIZE / RESOLUTION.size
trajectory = [math.cos(x) - 0.1 * math.cos(x / 16) * (1 - 2 * math.sin(x / 16))]
for i in range(1000):
    print(f"Step {i}: max value {trajectory[-1].max}")
    trajectory.append(etrk2(trajectory[-1]).vector['x'])
u_hist_etrk = math.stack(trajectory, spatial('time'))
vis.show(CenteredGrid(u_hist_etrk, bounds=Box(x=SIZE, time=len(trajectory) * dt)), aspect='auto', size=(8, 6))
