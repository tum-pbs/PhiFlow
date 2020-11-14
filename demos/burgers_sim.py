from phi.flow import *


domain = Domain(x=64, y=64, boundaries=PERIODIC, bounds=Box[0:100, 0:100])
velocity = domain.grid(Noise(vector=2)) * 2

for _ in ModuleViewer(framerate=10).range():
    velocity = field.diffuse(velocity, 0.1, 1)
    velocity = advect.semi_lagrangian(velocity, velocity, 1)
