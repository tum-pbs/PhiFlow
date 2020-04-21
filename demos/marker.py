from phi.flow import *


DESCRIPTION = """
Fluid simulation with additional marker fields that are passively transported with the fluid.

The dense marker is sampled on a regular grid while the sparse marker is a collection of particles.
"""


def checkerboard(resolution, size=8, offset=2):
    data = math.zeros([1]+list(resolution)+[1])
    for y in range(size):
        for x in range(size):
            data[:, y+offset::size*2, x+offset::size*2, :] = 1
    return data


def regular_locations(box, count=16):
    return np.reshape(CenteredGrid.getpoints(box, [count] * box.rank).data, (1, -1, box.rank))


domain = Domain([160, 126], CLOSED)
smoke = world.add(Fluid(domain, buoyancy_factor=0.1), physics=IncompressibleFlow())
world.add(Inflow(Sphere((18, 64), 10), rate=0.2))
# --- Markers ---
dense_marker = world.add(CenteredGrid(checkerboard(domain.resolution), box=domain.box, extrapolation='constant'), physics=Drift())
sparse_marker = world.add(SampledField(regular_locations(domain.box)), physics=Drift())

app = App('Passive Markers', DESCRIPTION, framerate=10, dt=0.2)
app.add_field('Density', lambda: smoke.density)
app.add_field('Velocity', lambda: smoke.velocity)
app.add_field('Dense Marker', dense_marker)
app.add_field('Sparse Marker', lambda: sparse_marker.at(smoke.density))
show(app)
