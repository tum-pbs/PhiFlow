import numpy as np
from .domain import DomainState
from .field.effect import effect_applied, FieldEffect, ADD
from .field import Field, union_mask, GeometryMask
from . import StateDependency, Physics
from phi import math, struct


class QuantumWave(DomainState):

    def __init__(self, domain, amplitude=1, mass=0.1, tags=('qwave',), **kwargs):
        DomainState.__init__(**struct.kwargs(locals()))

    @struct.attr(default=1)
    def amplitude(self, amplitude):
        return self.centered_grid('amplitude', amplitude, dtype=np.complex64)

    @struct.prop(default=0.1)
    def mass(self, mass): return mass

    def default_physics(self): return SCHROEDINGER


def normalize_probability(probability_amplitude):
    p = math.to_float(abs(probability_amplitude) ** 2)
    P = math.sum(p, math.spatial_dimensions(p), keepdims=True)
    return probability_amplitude / math.to_complex(math.sqrt(P))


def psquare(complex):
    return math.imag(complex) ** 2 + math.real(complex) ** 2


class Schroedinger(Physics):

    def __init__(self, margin=1):
        Physics.__init__(self, [StateDependency('obstacles', 'obstacle'),
                                StateDependency('potentials', 'potential_effect', blocking=True)])
        self.margin = margin

    def step(self, state, dt=1.0, potentials=(), obstacles=()):
        if len(potentials) == 0:
            potential = 0
        else:
            potential = math.zeros_like(math.real(state.amplitude))  # for the moment, allow only real potentials
            for pot in potentials:
                potential = effect_applied(pot, potential, dt)
            potential = potential.data

        amplitude = state.amplitude.data

        # Rotate by potential
        rotation = math.exp(1j * math.to_complex(potential * dt))
        amplitude = amplitude * rotation

        # Move by rotating in Fourier space
        amplitude_fft = math.fft(amplitude)
        laplace = math.fftfreq(state.resolution, mode='square')
        amplitude_fft *= math.exp(-1j * (2 * np.pi)**2 * math.to_complex(dt) * laplace / (2*state.mass))
        amplitude = math.ifft(amplitude_fft)

        obstacle_mask = union_mask([obstacle.geometry for obstacle in obstacles]).at(state.amplitude).data
        amplitude *= 1 - obstacle_mask

        normalized = False
        symmetric = False
        if not symmetric:
            boundary_mask = math.zeros(state.domain.centered_shape(1, batch_size=1)).data
            boundary_mask[[slice(None)] + [slice(self.margin,-self.margin) for i in math.spatial_dimensions(boundary_mask)] + [slice(None)]] = 1
            amplitude *= boundary_mask

        if len(obstacles) > 0 or not symmetric:
            amplitude = normalize_probability(amplitude)
            normalized = True

        return state.copied_with(amplitude=amplitude)


SCHROEDINGER = Schroedinger()


StepPotential = lambda geometry, height: FieldEffect(GeometryMask('potential', [geometry], height), ['potential'], mode=ADD)


class WavePacket(Field):

    def __init__(self, center, size, wave_vector, name='wave_packet', **kwargs):
        bounds = None
        data = None
        Field.__init__(**struct.kwargs(locals()))

    @struct.prop()
    def center(self, center): return center

    @struct.prop()
    def size(self, size): return size

    @struct.attr()
    def data(self, data):
        assert data is None
        return None

    @struct.prop()
    def wave_vector(self, wave_vector):
        if len(math.shape(wave_vector)) == 0:
            wave_vector = math.expand_dims(wave_vector, 0)
        return wave_vector

    def sample_at(self, points, collapse_dimensions=True):
        envelope = math.exp(-0.5 * math.sum((points - self.center) ** 2, axis=-1, keepdims=True) / self.size ** 2)
        wave = math.exp(1j * math.expand_dims(np.dot(points, self.wave_vector), -1)) * envelope
        return wave

    @property
    def rank(self):
        return len(self.center)

    @property
    def component_count(self):
        return 1

    def unstack(self):
        return [self]

    @property
    def points(self):
        return None

    def compatible(self, other_field):
        return True

    def __repr__(self):
        return 'WavePacket(%s)' % self.center


def harmonic_potential(grid, center, unit_distance, maximum_value=1.0, dtype=np.float32):
    x = (grid.center_points() - center) / unit_distance
    pot = math.sum(x ** 2, -1, keepdims=True)
    if maximum_value is not None:
        pot = math.minimum(pot, maximum_value)
    return math.cast(pot, dtype)


def sin_potential(grid, k, phase_offset=0, dtype=np.float32):
    x = grid.center_points()
    phase_offset = math.expand_dims(phase_offset, -1, grid.rank+1)
    x_k = math.expand_dims(np.dot(x, k), -1)
    wave = math.sin(x_k + phase_offset)
    return math.cast(wave, dtype)
