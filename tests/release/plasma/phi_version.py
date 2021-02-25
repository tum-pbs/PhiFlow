from .numpy_reference import HW, Namespace
from phi import math, field

math.set_global_precision(64)


def get_domain_phi(plasma, domain):
    return domain.grid(math.fourier_poisson(plasma.omega.values, plasma.dx))


def step_gradient_2d(params, plasma, phi, dt=0):
    """time gradient of model"""
    # Diffusion function
    def diffuse(arr, N, dx):
        for i in range(N):
            arr = field.laplace(arr)  # math.fourier_laplace(arr, dx)
        return arr
    # Calculate Gradients
    dx_p, dy_p = field.gradient(phi).unstack('gradient')
    # Get difference
    diff = (phi - plasma.density)
    # Step 2.1: New Omega.
    nu = (-1)**(params['N'] + 1) * params['nu']
    o = (params['c1'] * diff)
    if params['arakawa_coeff']:
        o += - params['arakawa_coeff'] * math._nd._periodic_2d_arakawa_poisson_bracket(phi.values, plasma.omega.values, plasma.dx)
    if nu and params['N']:
        o += nu * diffuse(plasma.omega, params['N'], plasma.dx)
    # Step 2.2: New Density.
    n = (params['c1'] * diff)
    if params['arakawa_coeff']:
        n += - params['arakawa_coeff'] * math._nd._periodic_2d_arakawa_poisson_bracket(phi.values, plasma.density.values, plasma.dx)
    if params['kappa_coeff']:
        n += - params['kappa_coeff'] * dy_p
    if nu:
        n += nu * diffuse(plasma.density, params['N'], plasma.dx)
    return Namespace(
        density=n,
        omega=o,
        phi=phi,  # NOTE: NOT A GRADIENT
        age=plasma.age + dt,
        dx=plasma.dx
    )


def rk4_step(params, get_phi, dt, gradient_func=step_gradient_2d, **kwargs):
    # RK4
    yn = Namespace(**kwargs)  # given dict to Namespace
    if yn.age == 0:
        pn = get_phi(yn)
    else:
        pn = yn.phi
    k1 = dt * gradient_func(params, yn, pn, dt=0)
    p1 = get_phi(yn + k1 * 0.5)  # , guess=pn)
    k2 = dt * gradient_func(params, yn + k1 * 0.5, p1, dt=dt / 2)
    p2 = get_phi(yn + k2 * 0.5)  # , guess=pn+p1*0.5)
    k3 = dt * gradient_func(params, yn + k2 * 0.5, p2, dt=dt / 2)
    p3 = get_phi(yn + k3)  # , guess=pn+p2*0.5)
    k4 = dt * gradient_func(params, yn + k3, p3, dt=dt)
    y1 = yn + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return Namespace(
        density=y1.density,
        omega=y1.omega,
        phi=get_phi(y1),  # TODO: Somehow this does not work properly
        age=yn.age + dt,  # y1 contains 2 time steps from compute
        dx=yn.dx
    )
