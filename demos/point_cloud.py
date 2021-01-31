from phi.flow import *


DOMAIN = Domain(x=64, y=64, boundaries=CLOSED, bounds=Box(0, (100, 100)))
points = DOMAIN.points([(10, 10), (20, 20)], color=['#ba0a04', '#344feb'])

<<<<<<< HEAD
points1 = DOMAIN.points((10, 10), color='#ba0a04')
points2 = DOMAIN.points((20, 20), color='#344feb')
points = points1 & points2
points = field.concat(points1, points2, dim='points')

# Advection
velocity = DOMAIN.grid([-1, 1])
points = advect.advect(points, velocity, 10)  # RK4
points = advect.advect(points, points * (-1, 1), -5)  # Euler

# Grid sampling
scattered_data = points.sample_in(DOMAIN.cells)
scattered_grid = points >> DOMAIN.grid()
scattered_sgrid = points >> DOMAIN.staggered_grid()

ModuleViewer()
=======
positions = math.tensor([(10, 10)], names='points,vector')
assert 'points' in positions.shape
cloud = PointCloud(Sphere(positions, 1)) * [-1, 1]

# Advection
velocity = DOMAIN.grid([-1, 1])
cloud = advect.advect(cloud, velocity, 10)  # RK4
cloud = advect.advect(cloud, cloud, -5)  # Euler

# Grid sampling
scattered_data = cloud.sample_in(DOMAIN.cells)
scattered_grid = cloud >> DOMAIN.grid()
scattered_sgrid = cloud >> DOMAIN.sgrid()

app = App()
app.add_field('Scattered', scattered_grid)
app.add_field('Scattered (Staggered)', scattered_sgrid)
show(app)
>>>>>>> Reset point_cloud demo.
