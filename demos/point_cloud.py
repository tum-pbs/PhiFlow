from phi.flow import *

domain = Domain([64, 64], CLOSED, bounds=Box(0, (100, 100)))

positions = math.tensor([(10, 10), (50, 50), (200, -10)], names='points,:')
cloud = PointCloud(Sphere(positions, 1)) * [-1, 1]

# Advection
velocity = domain.grid([-1, 1])
cloud = advect.advect(cloud, velocity, 10)  # RK4
cloud = advect.advect(cloud, cloud, -5)  # Euler

# Grid sampling
scattered_grid = cloud.at(domain.grid())
scattered_data = cloud.sample_in(domain.cells)
scattered_sgrid = cloud.at(domain.sgrid())

app = App()
app.add_field('Scattered', scattered_grid)
app.add_field('Scattered (Staggered)', scattered_sgrid)
show(app)
