from phi import math
from . import Physics
from .field import CenteredGrid


class KuramotoSivashinsky(Physics):

    def __init__(self):
        Physics.__init__(self)

    def step(self, u, dt=1.0, **dependent_states):
        assert isinstance(u, CenteredGrid)
        grad = u.gradient()
        laplace = u.laplace()
        laplace2 = laplace.laplace()
        du_dt = -laplace - laplace2 - 0.5 * grad ** 2
        result = u + dt * du_dt
        result -= math.mean(result.data, axis=tuple(range(1, len(math.staticshape(result.data)))), keepdims=True)
        return result.copied_with(age=u.age + dt, name=u.name)
