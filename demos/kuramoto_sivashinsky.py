from phi.flow import *
from phi.physics.flame import KuramotoSivashinsky


flame1 = CenteredGrid.sample(Noise(scale=5) * 2, Domain([32], PERIODIC, box=box(32)), name='flame1')
flame2 = (flame1 + Noise(scale=1) * 0.3).at(flame1).copied_with(name='flame2')
world.add(flame1, KuramotoSivashinsky())
world.add(flame2, KuramotoSivashinsky())

app = App('Kuramoto-Sivashinsky', 'Chaotic flame simulation', dt=0.1, framerate=10)
show(app)
