""" Fluid Logo

Incompressible fluid simulation with obstacles and buoyancy.

Specify the backend to be used in the commandline (after the resolution argument):

* Pass `tf` for TensorFlow mode
* Pass `torch` for PyTorch mode
* Else, NumPy mode

Example: `python fluid_logo.py 100 tf`
"""

import sys
if 'tf' in sys.argv:
    from phi.tf.flow import *
    MODE = 'TensorFlow'
elif 'torch' in sys.argv:
    from phi.torch.flow import *
    MODE = 'PyTorch'
else:
    from phi.flow import *  # Use NumPy
    MODE = 'NumPy'


RESOLUTION = int(sys.argv[1]) if len(sys.argv) > 1 and __name__ == '__main__' else 100
DOMAIN = Domain(x=RESOLUTION, y=RESOLUTION, boundaries=CLOSED, bounds=Box[0:100, 0:100])

OBSTACLE_GEOMETRIES = [Box[15 + x * 7:15 + (x + 1) * 7, 41:83] for x in range(1, 10, 2)] + [Box[43:50, 41:48], Box[15:43, 83:90], Box[50:85, 83:90]]
OBSTACLE = Obstacle(union(OBSTACLE_GEOMETRIES))
OBSTACLE_MASK = HardGeometryMask(OBSTACLE.geometry) >> DOMAIN.scalar_grid()

INFLOW = DOMAIN.scalar_grid(Box[14:21, 6:10]) + \
         DOMAIN.scalar_grid(Box[79:86, 6:10]) * 0.8 + \
         DOMAIN.scalar_grid(Box[44:47, 49:50]) * 0.1
velocity = DOMAIN.staggered_grid(0)
smoke = pressure = divergence = remaining_divergence = DOMAIN.scalar_grid(0)

for _ in ModuleViewer(display=('smoke', 'velocity', 'pressure', 'OBSTACLE_MASK')).range():
    smoke = advect.semi_lagrangian(smoke, velocity, 1) + INFLOW
    buoyancy_force = smoke * (0, 0.1) >> velocity  # resamples density to velocity sample points
    velocity = advect.semi_lagrangian(velocity, velocity, 1) + buoyancy_force
    velocity, pressure, iterations, divergence = fluid.make_incompressible(velocity, DOMAIN, obstacles=(OBSTACLE,), pressure_guess=pressure)
    remaining_divergence = field.divergence(velocity)
