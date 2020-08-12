import numpy as np

from phi import math, struct

from . import Physics
from .domain import DomainState
from .field import AnalyticField


@struct.definition()
class Pattern(DomainState):

    def __init__(self, domain, u=0, v=0, tags=('pattern',), **kwargs):
        DomainState.__init__(self, **struct.kwargs(locals()))

    @struct.variable(default=0, dependencies=DomainState.domain)
    def u(self, U):
        return self.centered_grid('U', U, dtype=np.float32)

    @struct.variable(default=0, dependencies=DomainState.domain)
    def v(self, V):
        return self.centered_grid('V', V, dtype=np.float32)

    @struct.constant(default=0.19)
    def du(self, du):
        return du

    @struct.constant(default=0.05)
    def dv(self, dv):
        return dv

    @struct.constant(default=0.04)
    def f(self, f):
        return f

    @struct.constant(default=0.05)
    def k(self, k):
        return k


class ReactionDiffusion(Physics):

    def __init__(self):
        Physics.__init__(self)

    def step(self, pattern, dt=1.0, **kwargs):
        # if struct.all(Material.periodic(pattern.domain.boundaries)):
        #     dx = math.mean(pattern.domain.dx)
        #     lu = math.fourier_laplace(pattern.u) / dx ** 2
        #     lv = math.fourier_laplace(pattern.v) / dx ** 2
        # else:
        lu = pattern.u.laplace()
        lv = pattern.v.laplace()
        uvv = pattern.u * pattern.v ** 2
        su = pattern.du * lu - uvv + pattern.f * (1 - pattern.u)
        sv = pattern.dv * lv + uvv - (pattern.f + pattern.k) * pattern.v
        return pattern.copied_with(u=pattern.u + dt * su, v=pattern.v + dt * sv)


@struct.definition()
class Seed(AnalyticField):

    def __init__(self, center=(0, 0), size=0, mode='RANDOM', factor=1.0, name='Seed', data=1.0, **kwargs):
        rank = math.staticshape(center)[-1]
        AnalyticField.__init__(self, **struct.kwargs(locals()))

    @struct.constant()
    def center(self, center):
        return center

    @struct.constant()
    def size(self, size):
        return size

    @struct.constant()
    def factor(self, factor):
        return factor

    @struct.constant()
    def mode(self, mode):
        return mode

    def sample_at(self, points):
        if self.mode == 'EXP':
            envelope = math.exp(-0.5 * math.sum((points - self.center) ** 2, axis=-1, keepdims=True) / self.size ** 2)
            envelope = math.to_float(envelope)
            return envelope * self.factor

        elif self.mode == 'RECT':
            conf = np.zeros(points.shape)
            conf[:, self.center[0] - self.size:self.center[0] + self.size, self.center[1] - self.size:self.center[1] + self.size, :] = np.ones(conf[:, self.center[0] - self.size:self.center[0] + self.size, self.center[1] - self.size:self.center[1] + self.size, :].shape)
            return conf[:, :, :, :-1] * self.factor

        elif self.mode == 'RANDOM':

            conf = np.random.random_sample(points.shape)
            conf[:, 0, :, :] *= 0
            conf[:, -1, :, :] *= 0
            conf[:, :, 0, :] *= 0
            conf[:, :, -1, :] *= 0

            return conf[:, :, :, :-1] * self.factor
