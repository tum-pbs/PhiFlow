"""
Profiles the common fluid operations advection and pressure solve.
The profile is stored in the working directory and can be viewed with e.g. with Google chrome.
"""
from phi.flow import *
# from phi.torch.flow import *
# from phi.tf.flow import *
# from phi.jax.flow import *


velocity = StaggeredGrid(0, x=64, y=64, bounds=Box(x=100, y=100))  # or CenteredGrid(...)
smoke = CenteredGrid(0, ZERO_GRADIENT, x=200, y=200, bounds=Box(x=100, y=100))
INFLOW = 0.2 * resample(Sphere(x=50, y=9.5, radius=5), to=smoke, soft=True)
velocity, pressure = fluid.make_incompressible(velocity)


@jit_compile  # Only for PyTorch, TensorFlow and Jax
def step(v, s, p, dt=1.):
    print("step")
    s = advect.mac_cormack(s, v, dt) + INFLOW
    buoyancy = resample(s * (0, 0.1), to=v)
    v = advect.semi_lagrangian(v, v, dt) + buoyancy * dt
    v, p = fluid.make_incompressible(v, (), Solve(x0=p, rel_tol=1e-3, preconditioner='auto'))
    return v, s, p


for _ in range(2):
    velocity, smoke, pressure = step(velocity, smoke, pressure)
print("Profiling")

profile = backend.profile_function(step, [velocity, smoke, pressure], warmup=0, call_count=1)
profile.save('profile-smoke-3.0.json')
print("Profile saved")
