from phi.flow import *

# --- Setup environment ---
inflow = 0
domain = Domain(x=64, y=64, boundaries=CLOSED, bounds=Box[0:64, 0:64])
obstacles = [Obstacle(Box[30:35, 30:35].rotated(math.tensor(-20)))]
bcs = flip.get_accessible_mask(domain, obstacles)

# --- Initialize particles ---
point_mask = domain.grid(HardGeometryMask(union([Box[20:40, 50:60], Box[:, :10]])))
initial_points = distribute_points(point_mask.values, 8)
initial_velocity = math.tensor(np.zeros(initial_points.shape), names=['points', 'vector'])
initial_particles = PointCloud(Sphere(initial_points, 0), values=initial_velocity)

state = dict(particles=initial_particles, v_field=initial_particles.at(domain.sgrid()), pressure=domain.grid(0),
             t=0, cmask=PointCloud(initial_particles.elements).at(domain.grid()))


def step(particles, v_field, pressure, dt, t, **kwargs):
    points = PointCloud(particles.elements, values=1)
    cmask = points >> domain.grid()
    smask = points >> domain.sgrid()
    v_force_field = v_field + dt * gravity_tensor(Gravity(), v_field.shape.spatial.rank)
    v_div_free_field, pressure = flip.make_incompressible(v_force_field, bcs, cmask, smask, pressure)
    particles = flip.map_velocity_to_particles(particles, v_div_free_field, smask, previous_velocity_grid=v_field)
    particles = advect.advect(particles, v_div_free_field, dt, occupied=smask, valid=bcs, mode='rk4')
    if t < inflow:
        particles = particles & initial_particles
    particles = flip.respect_boundaries(particles, domain, obstacles)
    return dict(particles=particles, v_field=particles.at(domain.sgrid()), pressure=pressure, t=t + 1, cmask=cmask)


app = App()
app.set_state(state, step_function=step, dt=0.1, show=['v_field', 'pressure', 'cmask'])
show(app, display='cmask')
