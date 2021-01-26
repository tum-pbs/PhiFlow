from phi.flow import *

DOMAIN = Domain(x=64, y=64, boundaries=CLOSED, bounds=Box(0, (100, 100)))

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
