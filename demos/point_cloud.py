from phi.flow import *

DOMAIN = Domain(x=64, y=64, boundaries=CLOSED, bounds=Box(0, (100, 100)))

positions1 = math.tensor([(10, 10)], names='points,vector')
positions2 = math.tensor([(20, 20), (30, 30)], names='points,vector')
assert 'points' in positions1.shape
assert 'points' in positions2.shape
cloud1 = PointCloud(Sphere(positions1, 1), bounds=DOMAIN.bounds, color='#344feb') * [-1, 1]
cloud2 = PointCloud(Sphere(positions2, 1), bounds=DOMAIN.bounds, color='#bd1a08') * [-1, 1]

# Advection
velocity = DOMAIN.grid([-1, 1])
cloud1 = advect.advect(cloud1, velocity, 10)  # RK4
cloud1 = advect.advect(cloud1, cloud1, -5)  # Euler

# Grid sampling
scattered_data = cloud1.sample_in(DOMAIN.cells)
scattered_grid = cloud1 >> DOMAIN.grid()
scattered_sgrid = cloud1 >> DOMAIN.sgrid()

app = App()
app.add_field('Scattered', scattered_grid)
app.add_field('Scattered (Staggered)', scattered_sgrid)
app.add_field('Point clouds', [cloud1, cloud2])
show(app)
