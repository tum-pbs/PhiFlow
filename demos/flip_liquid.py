from phi.flow import *

# --- Setup environment ---
inflow = 0
domain = Domain(x=64, y=64, boundaries=CLOSED, bounds=Box[0:64, 0:64])
obstacles = [Obstacle(Box[30:35, 30:35].rotated(math.tensor(-20)))]
bcs = flip.get_bcs(domain, obstacles)

# --- Initialize particles ---
point_mask = domain.grid().values
point_mask.native()[20:40, 50:60] = 1
point_mask.native()[:, :10] = 1
initial_points = distribute_points(point_mask, 8)
initial_velocity = math.tensor(np.zeros(initial_points.shape), names=['points', 'vector'])
initial_particles = PointCloud(Sphere(initial_points, 0), values=initial_velocity)

state = dict(particles=initial_particles, v_field=initial_particles.at(domain.sgrid()), pressure=domain.grid(0),
             t=0, cmask=PointCloud(initial_particles.elements).at(domain.grid()))


def step(particles, v_field, pressure, dt, t, **kwargs):
    points = PointCloud(particles.elements, values=1)
    cmask = points.at(domain.grid())
    smask = points.at(domain.sgrid())
    v_force_field = flip.apply_gravity(dt, v_field)
    v_div_free_field, pressure = flip.make_incompressible(v_force_field, bcs, cmask, smask, pressure)
    particles = flip.map2particle(particles, v_div_free_field, smask, initial_v_field=v_field)
    particles = advect.advect(particles, v_div_free_field, dt, mask=smask, bcs=bcs, mode='rk4_extp')
    if t < inflow:
        particles = flip.add_inflow(particles, initial_points, initial_velocity)
    particles = flip.respect_boundaries(domain, obstacles, particles)
    return dict(particles=particles, v_field=particles.at(domain.sgrid()), pressure=pressure, t=t + 1, cmask=cmask)


app = App()
app.set_state(state, step_function=step, dt=0.1, show=['v_field', 'pressure', 'cmask'])
show(app, display='cmask')
