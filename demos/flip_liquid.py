""" FLIP simulation for liquids

A liquid block is coliding with a rotated obstacle and falling into a liquid pool.
"""

from phi.flow import *

DOMAIN = Domain(x=64, y=64, boundaries=CLOSED, bounds=Box[0:64, 0:64])
gravity = math.tensor([0, -9.81])
dt = 0.1
not_accessible = [Box[20:35, 30:35].rotated(math.tensor(-20))]
obstacles = flip.get_points(DOMAIN, not_accessible, color='#000000')
accessible = flip.get_accessible_mask(DOMAIN, not_accessible)
particles = flip.get_points(DOMAIN, [Box[20:40, 50:60], Box[:, :5]])
velocity = particles >> DOMAIN.staggered_grid()
pressure = DOMAIN.grid(0)
scene = particles & obstacles

for _ in ModuleViewer(display='scene').range():
    force_velocity = velocity + dt * gravity
    div_free_velocity, pressure, occupied = flip.make_incompressible(force_velocity, accessible, particles, DOMAIN)
    particles = flip.map_velocity_to_particles(particles, div_free_velocity, occupied, previous_velocity_grid=velocity)
    particles = advect.advect(particles, div_free_velocity, dt, accessible=accessible, occupied=occupied)
    particles = flip.respect_boundaries(particles, DOMAIN, not_accessible)
    velocity = particles >> DOMAIN.staggered_grid()
    scene = particles & obstacles
