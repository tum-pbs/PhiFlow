from phi.flow import *

velocity = StaggeredGrid((0, 0), 0, x=32, y=32, bounds=Box(x=100, y=100))  # or CenteredGrid(...)
velocity_emb = velocity @ StaggeredGrid(0, velocity, x=64, y=64, bounds=Box(x=(30, 70), y=(40, 80)))
smoke = CenteredGrid(0, extrapolation.BOUNDARY, x=200, y=200, bounds=Box(x=100, y=100))

OBSTACLE = Obstacle(Sphere(x=50, y=60, radius=5))
INFLOW = 0.2 * CenteredGrid(SoftGeometryMask(Sphere(x=50, y=9.5, radius=5)), 0, smoke.bounds, smoke.resolution)
pressure = None


# @jit_compile  # Only for PyTorch, TensorFlow and Jax
def step(v, v_emb, s, p, dt=1.):
    s = advect.mac_cormack(s, v_emb, dt) + INFLOW
    buoyancy = s * (0, 0.1)
    v_emb = advect.semi_lagrangian(v_emb, v_emb, dt) + (buoyancy @ v_emb) * dt
    v = advect.semi_lagrangian(v, v, dt) + (buoyancy @ v) * dt
    v, p = fluid.make_incompressible(v, [OBSTACLE], Solve('auto', 1e-5, 0, x0=p))
    # Perform the embedded pressure solve
    p_emb_x0 = CenteredGrid(0, p, v_emb.bounds, v_emb.resolution)
    v_emb = StaggeredGrid(v_emb, extrapolation.BOUNDARY, v_emb.bounds, v_emb.resolution)
    v_emb, p_emb = fluid.make_incompressible(v_emb, [OBSTACLE], Solve('auto', 1e-5, 1e-5, x0=p_emb_x0))
    v_emb = StaggeredGrid(v_emb, v, v_emb.bounds, v_emb.resolution)
    return v, v_emb, s, p


def full_velocity():
    return StaggeredGrid(velocity_emb, 0, velocity.bounds, velocity.resolution)


for _ in view('smoke, full_velocity, velocity_emb, velocity, pressure', play=True, namespace=globals(), gui='dash').range(warmup=1):
    velocity, velocity_emb, smoke, pressure = step(velocity, velocity_emb, smoke, pressure)
    velocity_all = vis.overlay(velocity, velocity_emb)
