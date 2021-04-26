from phi.flow import *

DOMAIN = Domain(x=30, y=30, bounds=Box[0:30, 0:30])
DT = 0.1

def move_obstacle(obstacle):
    if (obstacle.geometry.center[0]) > 35:
        new_geometry = Box[-6:0, 10:16]
    else:
        new_geometry = obstacle.geometry.shifted([1. * DT, 0])
    return obstacle.copied_with(geometry=new_geometry)

obstacle = Obstacle((Box[5:11, 10:16]), velocity=[1.,0])
velocity = DOMAIN.staggered_grid((0, 0))
pressure = DOMAIN.scalar_grid()
obstacle_mask = HardGeometryMask(obstacle.geometry) >> pressure
for _ in ModuleViewer(display=('velocity', 'obstacle_mask'), auto_play=True).range():
    obstacle = move_obstacle(obstacle)
    velocity = advect.mac_cormack(velocity, velocity, DT)
    velocity, pressure, _iter, _ = fluid.make_incompressible(velocity, DOMAIN, (obstacle,), solve_params=math.LinearSolve(absolute_tolerance=1e-2 , max_iterations=1e5),  pressure_guess=pressure)
    obstacle_mask = HardGeometryMask(obstacle.geometry) >> pressure
