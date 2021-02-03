from phi.flow import *


DOMAIN = Domain(x=64, y=64, boundaries=CLOSED, bounds=Box(0, (100, 100)))
points = DOMAIN.points([(10, 10), (20, 20)], color=['#ba0a04', '#344feb'])

# Advection
velocity = DOMAIN.grid([-1, 1])
points = advect.advect(points, velocity, 10)  # RK4
points = advect.advect(points, points * (-1, 1), -5)  # Euler

# Grid sampling
scattered_data = points.sample_in(DOMAIN.cells)
scattered_grid = points >> DOMAIN.grid()
scattered_sgrid = points >> DOMAIN.sgrid()

ModuleViewer()
