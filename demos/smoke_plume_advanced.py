""" Interactive Smoke Plume
Hot smoke is emitted from a circular region at the bottom.
The simulation computes the resulting air flow in a closed box.

The grid resolution of the smoke density and velocity field can be changed during the simulation.
The actual resolution values are the squares of the slider values.

Performance curves are available in Φ-Board.
"""

from phi.flow import *

smoke_res = vis.control(8, (3, 20))
v_res = vis.control(8, (3, 20))
pressure_solver = vis.control('auto', ('auto', 'CG', 'CG-adaptive', 'CG-native', 'direct', 'GMres', 'lGMres', 'biCG', 'CGS', 'QMR', 'GCrotMK'))

BOUNDS = Box(x=100, y=100)
INFLOW = Sphere(x=50, y=10, radius=5)
velocity = StaggeredGrid((0, 0), 0, x=v_res ** 2, y=v_res ** 2, bounds=BOUNDS)
smoke = CenteredGrid(0, ZERO_GRADIENT, x=smoke_res ** 2, y=smoke_res ** 2, bounds=BOUNDS)

viewer = view(smoke, velocity, namespace=globals(), play=False)
for _ in viewer.range(warmup=1):
    # Resize grids if needed
    inflow = resample(INFLOW, CenteredGrid(0, smoke.extrapolation, x=smoke_res ** 2, y=smoke_res ** 2, bounds=BOUNDS), soft=True)
    smoke = resample(smoke, inflow)
    velocity = velocity.at(StaggeredGrid(0, velocity.extrapolation, x=v_res ** 2, y=v_res ** 2, bounds=BOUNDS))
    # Physics step
    smoke = advect.mac_cormack(smoke, velocity, 1) + inflow
    buoyancy_force = (smoke * (0, 0.1)).at(velocity)
    velocity = advect.semi_lagrangian(velocity, velocity, 1) + buoyancy_force
    try:
        with math.SolveTape() as solves:
            velocity, pressure = fluid.make_incompressible(velocity, (), Solve(pressure_solver, 1e-5))
        viewer.log_scalars(solve_time=solves[0].solve_time)
        viewer.info(f"Presure solve {v_res**2}x{v_res**2} with {solves[0].method}: {solves[0].solve_time * 1000:.0f} ms ({solves[0].iterations} iterations)")
    except ConvergenceException as err:
        viewer.info(f"Presure solve {v_res**2}x{v_res**2} with {err.result.method}: {err}\nMax residual: {math.max(abs(err.result.residual.values))}")
        velocity -= field.spatial_gradient(err.result.x, velocity.extrapolation, type=type(velocity))
