"""
Functions for simulating the wave equation on `phi.field.Field` objects.

Provides explicit leapfrog and Euler time integration for the scalar wave equation:

    ∂²u/∂t² = c² ∇²u

where u is the wave amplitude and c is the wave speed.
"""
from typing import Union

from phi import math
from phi.field import Field, laplace
from phiml.math import Tensor, wrap, spatial


def step(u: Field,
         u_prev: Field,
         c: Union[float, Tensor, Field] = 1.,
         dt: Union[float, Tensor] = 1.,
         source: Field = None) -> tuple:
    """
    Advance the wave equation by one time step using leapfrog (Verlet) integration.

    Solves ∂²u/∂t² = c² ∇²u + f where u is the wave amplitude, c is the wave speed,
    and f is an optional source term.

    This scheme is second-order accurate in both space and time and preserves energy.

    Args:
        u: Current wave amplitude as a `CenteredGrid` or other `Field`.
        u_prev: Wave amplitude at the previous time step.
        c: Wave speed. Can be a constant, `Tensor`, or spatially varying `Field`.
        dt: Time step size.
        source: Optional source term f(x, t) as a `Field`.

    Returns:
        Tuple of `(u_next, u)` where `u_next` is the amplitude at the new time step.
        Pass these as `(u, u_prev)` to advance again.

    Examples:
        >>> from phi.flow import *
        >>> u = CenteredGrid(Noise(), x=64, y=64, bounds=Box(x=1, y=1))
        >>> u_prev = u  # start from rest
        >>> u_next, u = wave.step(u, u_prev, c=1.0, dt=0.01)
    """
    c_sq = c * c if not isinstance(c, Field) else c * c
    acceleration = c_sq * laplace(u)
    if source is not None:
        acceleration = acceleration + source
    u_next = 2 * u - u_prev + dt * dt * acceleration
    return u_next, u


def euler_step(u: Field,
               v: Field,
               c: Union[float, Tensor, Field] = 1.,
               dt: Union[float, Tensor] = 1.,
               source: Field = None) -> tuple:
    """
    Advance the wave equation by one time step using symplectic Euler integration.

    Solves the first-order system:
        ∂u/∂t = v
        ∂v/∂t = c² ∇²u + f

    This is useful when the initial velocity v = ∂u/∂t is known directly.

    Args:
        u: Current wave amplitude as a `Field`.
        v: Current velocity (time derivative of u) as a `Field`.
        c: Wave speed. Can be a constant, `Tensor`, or spatially varying `Field`.
        dt: Time step size.
        source: Optional source term f(x, t) as a `Field`.

    Returns:
        Tuple of `(u_next, v_next)` for the next time step.

    Examples:
        >>> from phi.flow import *
        >>> u = CenteredGrid(0, x=64, y=64, bounds=Box(x=1, y=1))
        >>> v = CenteredGrid(Noise(), x=64, y=64, bounds=Box(x=1, y=1))
        >>> u_next, v_next = wave.euler_step(u, v, c=1.0, dt=0.01)
    """
    c_sq = c * c if not isinstance(c, Field) else c * c
    acceleration = c_sq * laplace(u)
    if source is not None:
        acceleration = acceleration + source
    v_next = v + dt * acceleration
    u_next = u + dt * v_next
    return u_next, v_next


def differential(u: Field,
                 c: Union[float, Tensor, Field] = 1.,
                 source: Field = None) -> Field:
    """
    Compute the acceleration term c² ∇²u + f without advancing time.

    This is the right-hand side of ∂²u/∂t² = c² ∇²u + f and can be used
    with custom time integrators such as `phi.physics.integrate.rk4`.

    Args:
        u: Wave amplitude as a `Field`.
        c: Wave speed.
        source: Optional source term.

    Returns:
        Acceleration field ∂²u/∂t² as a `Field`.
    """
    c_sq = c * c if not isinstance(c, Field) else c * c
    result = c_sq * laplace(u)
    if source is not None:
        result = result + source
    return result
