from .domain import *
from .smoke import initialize_field



class ProbabilityAmplitude(State):

    __struct__ = State.__struct__.extend(('_real', '_imag'), ('_domain', '_is_normalized'))

    def __init__(self, domain, real=1, imag=0, is_normalized=False, batch_size=None):
        State.__init__(self, tags=('pamp',), batch_size=batch_size)
        self._domain = domain
        self._real = initialize_field(real, self.grid.shape(1, self._batch_size))
        self._imag = initialize_field(imag, self.grid.shape(1, self._batch_size))
        self._is_normalized = is_normalized

    @property
    def domain(self):
        return self._domain

    @property
    def grid(self):
        return self._domain._grid

    @property
    def real(self):
        return self._real

    @property
    def imag(self):
        """
Imaginary component of the complex wave field.
This field is sampled half a time step later than the real component.
        :return: a tensor
        """
        return self._imag

    @property
    def is_normalized(self):
        return self._is_normalized

    def copied_with(self, **kwargs):
        if ('real' in kwargs or 'imag' in kwargs) and 'is_normalized' not in kwargs:
            kwargs['is_normalized'] = False
        if 'real' in kwargs:
            kwargs['real'] = initialize_field(kwargs['real'], self.grid.shape(1, self._batch_size))
        if 'imag' in kwargs:
            kwargs['imag'] = initialize_field(kwargs['imag'], self.grid.shape(1, self._batch_size))
        return State.copied_with(self, **kwargs)

    def default_physics(self):
        return SCHROEDINGER


def normalize_probability(probability_amplitude):
    q = probability_amplitude
    norm = sum(q.real ** 2 + q.imag ** 2, spatial_dimensions(q.real), keepdims=True)
    norm = sqrt(norm)
    return probability_amplitude.copied_with(real=q.real/norm, imag=q.imag/norm, is_normalized=True)


class Schroedinger(Physics):

    def __init__(self):
        Physics.__init__(self, {'potentials': 'potential_effect'})

    def step(self, state, dt=1.0, potentials=()):
        if len(potentials) == 0:
            potential = 0
        else:
            potential = zeros_like(state.real)
            for pot in potentials:
                potential = pot.apply_grid(potential, state.grid, False, dt)

        real = state.real + dt * (potential * state.imag - laplace(state.imag))
        imag = state.imag + dt * (laplace(real) - potential * real)
        return state.copied_with(real=real, imag=imag)


SCHROEDINGER = Schroedinger()


QuantumBarrier = lambda geometry, height: FieldEffect(ConstantField(geometry, height), ['potential'])


def wave_packet(domain, center, size, wave_vector):
    x = domain.grid.indices()
    envelope = exp(-0.5 * sum((x - center)**2, axis=-1, keepdims=True) / size**2)
    real = np.cos(expand_dims(np.dot(x, wave_vector), -1)) * envelope
    imag = np.sin(expand_dims(np.dot(x, wave_vector), -1)) * envelope
    return real, imag
