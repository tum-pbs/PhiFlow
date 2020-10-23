from phi.flow import *

domain = Domain([64, 64], CLOSED, box=Box(0, (100, 100)))

positions = math.tensor([(10, 10), (50, 50)], names=('points', 'vector'))  # , (10, 50), (200, -10)
cloud = PointCloud(Sphere(positions, 0)) * [-1, 1]


velocity = domain.grid([-1, 1])
cloud = advect.runge_kutta_4(cloud, velocity, 10)

scattered_grid = cloud.at(domain.grid())
scattered_data = cloud.sample_at(domain.cells)

app = App()
app.add_field('Scattered', scattered_grid)
show(app)
