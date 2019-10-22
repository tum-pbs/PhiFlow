from .domain import *
from .effect import *



class QuantumWave(DomainState):
    __struct__ = DomainState.__struct__.extend(['_amplitude'], ['_mass'])

    def __init__(self, domain, amplitude=1, mass=0.1, batch_size=None):
        DomainState.__init__(self, domain, tags=('qwave',), batch_size=batch_size)
        self._amplitude = amplitude
        self._mass = mass
        self.__validate__()

    @property
    def amplitude(self):
        return self._amplitude

    def __validate_amplitude__(self):
        self._amplitude = self.centered_grid('amplitude', self._amplitude, dtype=np.complex64)

    @property
    def mass(self):
        return self._mass

    def default_physics(self):
        return SCHROEDINGER


def normalize_probability(probability_amplitude):
    p = math.to_float(abs(probability_amplitude) ** 2)
    P = math.sum(p, math.spatial_dimensions(p), keepdims=True)
    return probability_amplitude / math.to_complex(math.sqrt(P))


def psquare(complex):
    return math.imag(complex) ** 2 + math.real(complex) ** 2


class Schroedinger(Physics):

    def __init__(self, margin=1):
        Physics.__init__(self, dependencies={'obstacles': 'obstacle'},
                         blocking_dependencies={'potentials': 'potential_effect'})
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

        obstacle_mask = union([obstacle.geometry for obstacle in obstacles]).at(state.amplitude).data
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
    __struct__ = State.__struct__.extend([], ['_center', '_size', '_wave_vector', '_bounds', '_name', '_flags'])

    def __init__(self, center, size, wave_vector):
        Field.__init__(self, 'wave-packet', None, None)
        self._center = center
        self._size = size
        self._wave_vector = wave_vector
        self.__validate__()

    def __validate_wave_vector__(self):
        if len(math.shape(self._wave_vector)) == 0:
            self._wave_vector = math.expand_dims(self._wave_vector, 0)

    @property
    def center(self):
        return self._center

    @property
    def size(self):
        return self._size

    @property
    def wave_vector(self):
        return self._wave_vector

    def sample_at(self, points, collapse_dimensions=True):
        envelope = math.exp(-0.5 * math.sum((points - self.center) ** 2, axis=-1, keepdims=True) / self.size ** 2)
        wave = math.exp(1j * math.expand_dims(np.dot(points, self.wave_vector), -1)) * envelope
        return wave

    @property
    def rank(self):
        return len(self._center)

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
        return 'WavePacket(%s)' % self._center


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