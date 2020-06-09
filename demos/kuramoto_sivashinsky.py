from phi.flow import *
from phi.physics.flame import KuramotoSivashinsky


flame = CenteredGrid.sample(Noise(scale=5) * 2, Domain([32], PERIODIC, box=box(32)), name='flame')
world.add(flame, KuramotoSivashinsky())

show(App('Kuramoto-Sivashinsky', 'Chaotic flame simulation', dt=0.1, framerate=10))
