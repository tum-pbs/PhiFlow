from phi.field import ConstantField
from phi.field._field_math import *


def diffuse(field, diffusivity, dt, substeps=1):
    u"""
Simulate a finite-time diffusion process of the form dF/dt = α · ΔF on a given `Field` F with diffusion coefficient α.

If `field` is periodic (set via `extrapolation='periodic'`), diffusion may be simulated in Fourier space.
Otherwise, finite differencing is used to approximate the
    :param field: CenteredGrid, StaggeredGrid or ConstantField
    :param amount: number of Field, typically α · dt
    :param substeps: number of iterations to use
    :return: Field of same type as `field`
    :rtype: Field
    """
    if isinstance(field, ConstantField):
        return field
    assert isinstance(field, Grid), "Cannot diffuse field of type '%s'" % type(field)
    amount = diffusivity * dt
    if field.extrapolation == 'periodic' and not isinstance(amount, Field):
        fft_laplace = -(2 * pi) ** 2 * squared(fftfreq(field))
        diffuse_kernel = math.exp(fft_laplace * amount)
        return real(ifft(fft(field) * diffuse_kernel))
    else:
        if isinstance(amount, Field):
            amount = amount.at(field)
        for i in range(substeps):
            lap = laplace(field)
            field += amount / substeps * laplace(field)
        return field
