from phi.flow import *


DOMAIN = dict(x=30, y=30)
DT = 0.1


def move_obstacle(obstacle):
    if (obstacle.geometry.center[0]) > 35:
        new_geometry = Box[-6:0, 10:16]
    else:
        new_geometry = obstacle.geometry.shifted([1. * DT, 0])
    return obstacle.copied_with(geometry=new_geometry)


obstacle = Obstacle((Box[5:11, 10:16]), velocity=[1., 0], angular_velocity=tensor(0,))
velocity = StaggeredGrid(0, extrapolation.ZERO, **DOMAIN)
obstacle_mask = CenteredGrid(HardGeometryMask(obstacle.geometry), extrapolation.BOUNDARY, **DOMAIN)
pressure = None

for _ in view(velocity, obstacle_mask, play=True, namespace=globals()).range():
    obstacle = move_obstacle(obstacle)
    velocity = advect.mac_cormack(velocity, velocity, DT)
    velocity, pressure = fluid.make_incompressible(velocity, (obstacle,))
    fluid.masked_laplace.tracers.clear()  # we will need to retrace because the matrix changes each step. This is not needed when JIT-compiling the physics.
    obstacle_mask = HardGeometryMask(obstacle.geometry) @ pressure
