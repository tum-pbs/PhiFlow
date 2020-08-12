import numpy as np
from phi import math, struct

from . import Physics, StateDependency
from .domain import DomainState
from .field import AnalyticField, GeometryMask, union_mask
from .field.effect import ADD, FieldEffect, effect_applied


@struct.definition()
class QuantumWave(DomainState):

    def __init__(self, domain, amplitude=1, mass=0.1, tags=('qwave',), **kwargs):
        DomainState.__init__(self, **struct.kwargs(locals()))

    @struct.variable(default=1, dependencies=DomainState.domain)
    def amplitude(self, amplitude):
        return self.centered_grid('amplitude', amplitude, dtype=np.complex64)

    @struct.constant(default=0.1)
    def mass(self, mass):
        return mass

    def default_physics(self):
        return SCHROEDINGER


def normalize_probability(probability_amplitude):
    p = math.to_float(abs(probability_amplitude) ** 2)
    P = math.sum(p, math.spatial_dimensions(p), keepdims=True)
    return probability_amplitude / math.to_complex(math.sqrt(P))


psquare = math.abs_square


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
        amplitude_fft *= math.exp(-1j * (2 * np.pi)**2 * math.to_complex(dt) * laplace / (2 * state.mass))
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


def StepPotential(geometry, height):
    return FieldEffect(GeometryMask(geometry, name='potential') * height, ['potential'], mode=ADD)


@struct.definition()
class WavePacket(AnalyticField):

    def __init__(self, center, size, wave_vector, name='wave_packet', data=1.0, **kwargs):
        rank = math.staticshape(center)[-1]
        AnalyticField.__init__(self, **struct.kwargs(locals()))

    @struct.constant()
    def center(self, center): return center

    @struct.constant()
    def size(self, size): return size

    @struct.constant()
    def wave_vector(self, wave_vector):
        if len(math.shape(wave_vector)) == 0:
            wave_vector = math.expand_dims(wave_vector, 0)
        return wave_vector

    def sample_at(self, points):
        envelope = math.exp(-0.5 * math.sum((points - self.center) ** 2, axis=-1, keepdims=True) / self.size ** 2)
        envelope = math.to_float(envelope)
        wave = math.exp(1j * math.to_float(math.expand_dims(np.dot(points, self.wave_vector), -1))) * envelope
        return wave * self.data


@struct.definition()
class HarmonicPotential(AnalyticField):

    def __init__(self, center, unit_distance, maximum_value=1.0, data=1.0, name='harmonic', **kwargs):
        rank = math.shape(center)[-1]
        AnalyticField.__init__(self, **struct.kwargs(locals()))

    @struct.constant()
    def center(self, center): return center

    @struct.constant()
    def unit_distance(self, distance): return distance

    @struct.constant()
    def maximum_value(self, maximum_value): return maximum_value

    def sample_at(self, points):
        x = (points - self.center) / self.unit_distance
        pot = math.sum(x ** 2, -1, keepdims=True) * self.data
        if self.maximum_value is not None:
            pot = math.minimum(pot, self.maximum_value)
        return math.to_float(pot)


@struct.definition()
class SinPotential(AnalyticField):

    def __init__(self, k, phase_offset=0, data=1.0, name='harmonic', **kwargs):
        rank = math.size(k)
        AnalyticField.__init__(self, **struct.kwargs(locals()))

    @struct.variable()
    def k(self, k):
        """ Wave vector. Determines wave length and direction. """
        return k

    @struct.variable()
    def phase_offset(self, phase_offset): return phase_offset

    def sample_at(self, x):
        phase_offset = math.batch_align_scalar(self.phase_offset, 0, x)
        k = math.batch_align(self.k, 1, x)
        data = math.batch_align(self.data, 1, x)
        spatial_phase = math.sum(k * x, -1, keepdims=True)
        result = math.sin(math.to_float(spatial_phase + phase_offset)) * math.to_float(data)
        return result

    def __repr__(self):
        return 'Sin(x*%s)' % self.k

    @property
    def component_count(self):
        return math.staticshape(self.data)[-1]
