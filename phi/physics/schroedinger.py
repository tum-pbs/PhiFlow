from .domain import *
from .smoke import initialize_field



class QuantumWave(State):

    __struct__ = State.__struct__.extend(['_amplitude'], ['_domain', '_is_normalized', '_mass'])

    def __init__(self, domain, amplitude=1, is_normalized=False, mass=0.1, batch_size=None):
        State.__init__(self, tags=('qwave',), batch_size=batch_size)
        self._domain = domain
        self._amplitude = initialize_field(amplitude, self.domain.shape(1, self._batch_size), dtype=np.complex64)
        self._is_normalized = is_normalized
        self._mass = mass

    @property
    def domain(self):
        return self._domain

    @property
    def amplitude(self):
        return self._amplitude

    @property
    def mass(self):
        return self._mass

    @property
    def is_normalized(self):
        return self._is_normalized

    def copied_with(self, **kwargs):
        if ('amplitude' in kwargs) and 'is_normalized' not in kwargs:
            kwargs['is_normalized'] = False
        if 'amplitude' in kwargs:
            kwargs['amplitude'] = initialize_field(kwargs['amplitude'], self.domain.shape(1, self._batch_size), dtype=np.complex64)
        return State.copied_with(self, **kwargs)

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
                potential = pot.apply_grid(potential, state.domain, False, dt)

        amplitude = state.amplitude

        # Rotate by potential
        rotation = math.exp(1j * math.to_complex(potential * dt))
        amplitude *= rotation

        # Move by rotating in Fourier space
        amplitude_fft = math.fft(amplitude)
        laplace = math.sum(math.fftfreq(staticshape(amplitude)[1:-1]) ** 2, axis=-1, keepdims=True)
        amplitude_fft *= math.exp(-1j * np.pi * math.to_complex(dt) * laplace / state.mass)  # TODO should be 2 pi^2
        amplitude = math.ifft(amplitude_fft)

        for obstacle in obstacles:
            amplitude *= 1 - obstacle.geometry.at(state.domain)

        normalized = False
        symmetric = False
        if not symmetric:
            boundary_mask = np.zeros(state.domain.shape(1, batch_size=1))
            boundary_mask[[slice(None)] + [slice(self.margin,-self.margin) for i in math.spatial_dimensions(boundary_mask)] + [slice(None)]] = 1
            amplitude *= boundary_mask

        if len(obstacles) > 0 or not symmetric:
            amplitude = normalize_probability(amplitude)
            normalized = True

        return state.copied_with(amplitude=amplitude, is_normalized=normalized or state.is_normalized)


SCHROEDINGER = Schroedinger()


StepPotential = lambda geometry, height: FieldEffect(ComplexConstantField(geometry, height), ['potential'], mode=ADD)


def wave_packet(grid, center, size, wave_vector, normalized=True, dtype=np.complex64):
    if len(np.shape(wave_vector)) == 0:
        wave_vector = math.expand_dims(wave_vector, 0)
    x = grid.center_points()
    envelope = math.exp(-0.5 * math.sum((x - center)**2, axis=-1, keepdims=True) / size**2)
    wave = math.exp(1j * math.expand_dims(np.dot(x, wave_vector), -1)) * envelope
    wave = math.cast(wave, dtype)
    if normalized: wave = normalize_probability(wave)
    return wave


def wave_packet_gen(center, size, wave_vector, normalized=True):
    def init(shape, dtype=np.complex64):
        grid = Grid(shape[1:-1])
        return wave_packet(grid, center, size, wave_vector, normalized, dtype=dtype)
    return init


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