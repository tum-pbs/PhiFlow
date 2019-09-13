from .domain import *
from .smoke import initialize_field



class QuantumWave(State):

    __struct__ = State.__struct__.extend(['_amplitude'], ['_domain', '_is_normalized', '_mass'])

    def __init__(self, domain, amplitude=1, is_normalized=False, mass=0.1, batch_size=None):
        State.__init__(self, tags=('qwave',), batch_size=batch_size)
        self._domain = domain
        self._amplitude = initialize_field(amplitude, self.grid.shape(1, self._batch_size), dtype=np.complex64)
        self._is_normalized = is_normalized
        self._mass = mass

    @property
    def domain(self):
        return self._domain

    @property
    def grid(self):
        return self._domain._grid

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
            kwargs['amplitude'] = initialize_field(kwargs['amplitude'], self.grid.shape(1, self._batch_size), dtype=np.complex64)
        return State.copied_with(self, **kwargs)

    def default_physics(self):
        return SCHROEDINGER


def normalize_probability(probability_amplitude):
    p = abs(probability_amplitude) ** 2
    P = sum(p, spatial_dimensions(p))
    return probability_amplitude / sqrt(P)


class Schroedinger(Physics):

    def __init__(self, margin=1):
        Physics.__init__(self, dependencies={'obstacles': 'obstacle'},
                         blocking_dependencies={'potentials': 'potential_effect'})
        self.margin = margin

    def step(self, state, dt=1.0, potentials=(), obstacles=()):
        if len(potentials) == 0:
            potential = 0
        else:
            potential = zeros_like(state.amplitude)
            for pot in potentials:
                potential = pot.apply_grid(potential, state.grid, False, dt)

        amplitude = state.amplitude

        # Rotate by potential
        amplitude *= exp(1j * potential)

        # Move by rotating in Fourier space
        amplitude_fft = fft(amplitude)
        k = np.meshgrid(*[np.fft.fftfreq(n) for n in amplitude.shape[1:-1]], indexing='ij')
        k = expand_dims(stack(k, -1), 0)
        laplace = sum(k ** 2, axis=-1, keepdims=True)
        amplitude_fft *= exp(-1j * np.pi * laplace / state.mass)
        amplitude = ifft(amplitude_fft)

        for obstacle in obstacles:
            amplitude *= 1 - obstacle.geometry.at(state.grid)

        normalized = False
        symmetric = False
        if not symmetric:
            boundary_mask = np.zeros(state.grid.shape(1, batch_size=1))
            boundary_mask[[slice(None)] + [slice(self.margin,-self.margin) for i in spatial_dimensions(boundary_mask)] + [slice(None)]] = 1
            amplitude *= boundary_mask

        if len(obstacles) > 0 or not symmetric:
            amplitude = normalize_probability(amplitude)
            normalized = True

        return state.copied_with(amplitude=amplitude, is_normalized=normalized or state.is_normalized)


SCHROEDINGER = Schroedinger()


StepPotential = lambda geometry, height: FieldEffect(ConstantField(geometry, height), ['potential'])


def wave_packet(domain, center, size, wave_vector):
    if len(np.shape(wave_vector)) == 0:
        wave_vector = expand_dims(wave_vector, 0)
    x = domain.grid.indices()
    envelope = exp(-0.5 * sum((x - center)**2, axis=-1, keepdims=True) / size**2)
    return exp(1j * expand_dims(np.dot(x, wave_vector), -1)) * envelope
