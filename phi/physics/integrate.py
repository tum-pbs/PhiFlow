from typing import Callable


def rk4(pde: Callable, *state, dt=1., **pde_aux_kwargs):
    tan0 = pde(*state, **pde_aux_kwargs)
    tan_half = pde(*[s + t * dt * .5 for (s, t) in zip(state, tan0)], **pde_aux_kwargs)
    tan_half2 = pde(*[s + t * dt * .5 for (s, t) in zip(state, tan_half)], **pde_aux_kwargs)
    tan_full = pde(*[s + t * dt for (s, t) in zip(state, tan_half2)], **pde_aux_kwargs)
    tan_rk4 = [(1 / 6.) * (t0 + 2 * (th + th2) + tf) for (s, t0, th, th2, tf) in zip(state, tan0, tan_half, tan_half2, tan_full)]
    return tuple([s + t * dt for (s, t) in zip(state, tan_rk4)])


def euler(pde: Callable, *state, dt=1., **pde_aux_kwargs):
    tan = pde(*state, **pde_aux_kwargs)
    return tuple([s + t * dt for s, t in zip(state, tan)])
