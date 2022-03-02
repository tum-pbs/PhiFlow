""" Smoke Plume
Hot smoke is emitted from a circular region at the bottom.
The simulation computes the resulting air flow in a closed box.
"""
import time
import uuid
from phi.torch.flow import *
TORCH.set_default_device('GPU')

# Parameters
math_precision = 64
err_thresh = 1e-5
max_steps = 100
JIT = 1
#####

math.set_global_precision(math_precision)
measurement_ID = uuid.uuid4()

DOMAIN = dict(x=64, y=64, bounds=Box[0:100, 0:100])
velocity = StaggeredGrid((0, 0), extrapolation.ZERO, **DOMAIN)  # or use CenteredGrid
smoke = CenteredGrid(0, extrapolation.BOUNDARY, x=200, y=200, bounds=DOMAIN['bounds'])
INFLOW = 0.2 * CenteredGrid(SoftGeometryMask(Sphere(center=(50, 10), radius=5)), extrapolation.ZERO, resolution=smoke.resolution, bounds=smoke.bounds)
pressure = CenteredGrid(0, extrapolation.BOUNDARY, **DOMAIN)

@math.jit_compile
def step(smoke, velocity, pressure):
    smoke = advect.mac_cormack(smoke, velocity, 1) + INFLOW
    buoyancy_force = smoke * (0, 0.1) @ velocity  # resamples smoke to velocity sample points
    velocity = advect.semi_lagrangian(velocity, velocity, 1) + buoyancy_force
    with math.SolveTape() as tape:
        velocity, pressure = fluid.make_incompressible(velocity, (), Solve('CG', err_thresh, err_thresh, x0=pressure))
    return smoke, velocity, pressure, tape.solves[0].iterations


print('ID,STEP,TIME,ITERATIONS,CG_TIME,ERR_THRESH,TOTAL_STEPS,IMPLEMENTATION,JIT,PRECISION')
viewer = view(smoke, velocity, 'pressure', play=True, scene=False, keep_alive=False, namespace=globals())
for i in viewer.range(max_steps):
    time_0 = time.time()
    smoke, velocity, pressure, iterations = step(smoke, velocity, pressure)
    time_1 = time.time()
    print(f'{measurement_ID},{i},{round(time_1 - time_0,3)},{iterations},{err_thresh},{max_steps},C++_opt,{JIT},{math_precision}')