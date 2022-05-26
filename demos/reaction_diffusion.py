""" Reaction-Diffusion
Simulates a 2D system governed by a simple reaction-diffusion equation.
You can select one of the predefined configurations.
"""
from phi.flow import *


SAMPLE_PATTERNS = {
    'diagonal': {'du': 0.17, 'dv': 0.03, 'f': 0.06, 'k': 0.056},
    'maze': {'du': 0.19, 'dv': 0.05, 'f': 0.06, 'k': 0.062},
    'coral': {'du': 0.16, 'dv': 0.08, 'f': 0.06, 'k': 0.062},
    'flood': {'du': 0.19, 'dv': 0.05, 'f': 0.06, 'k': 0.02},
    'dots': {'du': 0.19, 'dv': 0.05, 'f': 0.04, 'k': 0.065},
    'dots_and_stripes': {'du': 0.19, 'dv': 0.03, 'f': 0.04, 'k': 0.061},
}

# Initial condition
# u = v = CenteredGrid(Sphere(x=50, y=50, radius=2), x=100, y=100)
# u = v = CenteredGrid(lambda x: math.exp(-0.5 * math.sum((x - 50)**2) / 3**2), x=100, y=100)
u = v = CenteredGrid(Noise(scale=20, smoothness=1.3), x=100, y=100) * .3 + .1


def reaction_diffusion(u, v, du, dv, f, k):
    uvv = u * v**2
    su = du * field.laplace(u) - uvv + f * (1 - u)
    sv = dv * field.laplace(v) + uvv - (f + k) * v
    return u + dt * su, v + dt * sv


dt = vis.control(1.)
pattern = vis.control('maze', tuple(SAMPLE_PATTERNS))
viewer = view('u,v', namespace=globals())
for _ in viewer.range():
    u, v = reaction_diffusion(u, v, **SAMPLE_PATTERNS[pattern])
