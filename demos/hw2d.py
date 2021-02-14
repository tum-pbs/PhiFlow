""" Hasegawa-Wakatani
Simple plasma flow model.
"""
# from phi.torch.flow import *
# from phi.tf.flow import *
from phi.flow import *
import time


math.set_global_precision(64)

# Simulation parameters
k0 = 0.15  # smallest wavenumber in the box
x = 64  # x size
y = 64  # y size
dt = 0.1  # timestep
DEBUG = False
# Physical Parameters
c1 = 0.1  # adiabatic coefficient
# Numerical Parameters
arakawa_coeff = 1  # Poisson bracket coefficient
kappa_coeff = 1  # background flow dy coefficient
nu = 0.001  # coefficient of hyperdiffusion
N = 3  # lap**(2*N) diffusion
# Derived
L = 2 * np.pi / k0  # Box Size
dx = L / x  # Grid Spaceing
nu = (-1)**(N + 1) * nu  # Smoothing coefficient & sign
# Packing
PARAMS = dict(c1=c1, nu=nu, N=N, arakawa_coeff=arakawa_coeff, kappa_coeff=kappa_coeff)


class Namespace(dict):

    def __mul__(self, other):
        if isinstance(other, Namespace):
            return Namespace({key:other[key] * val for key, val in self.items()})
        else:
            return Namespace({key:other * val for key, val in self.items()})

    __rmul__ = __mul__

    def __div__(self, other):
        if isinstance(other, Namespace):
            return Namespace({key:val / other[key] for key, val in self.items()})
        else:
            return Namespace({key:val / other for key, val in self.items()})

    def __truediv__(self, other):
        if isinstance(other, Namespace):
            return Namespace({key:val / other[key] for key, val in self.items()})
        else:
            return Namespace({key:val / other for key, val in self.items()})

    def __rdiv__(self, other):
        if isinstance(other, Namespace):
            return Namespace({key:other[key] / val for key, val in self.items()})
        else:
            return Namespace({key:other / val for key, val in self.items()})

    def __add__(self, other):
        if isinstance(other, Namespace):
            return Namespace({key:other[key] + val for key, val in self.items()})
        else:
            return Namespace({key:other + val for key, val in self.items()})

    __radd__ = __add__

    def __sub__(self, other):
        return Namespace({key:other - val for key, val in self.items()})

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)

    @property
    def dtype(self):
        return self['density'].dtype

    def copy(self):
        return Namespace({key:val for key, val in self.items()})


domain = Domain(x=x, y=y,
                boundaries=PERIODIC,
                bounds=Box[0:L, 0:L])
state = Namespace(density=domain.scalar_grid(math.random_normal(x=x, y=y)),
                  omega=domain.scalar_grid(math.random_normal(x=x, y=y)),
                  phi=domain.scalar_grid(math.random_normal(x=x, y=y)),
                  age=0,
                  dx=dx)


def get_phi(plasma, guess=None):
    """Fourier Poisson Solve for Phi"""
    centered_omega = plasma.omega  # (plasma.omega - np.mean(plasma.omega))
    phi = math.fourier_poisson(centered_omega.values, plasma.dx)
    return domain.scalar_grid(phi)


def step_gradient_2d(plasma, phi, dt=0):
    """time gradient of model"""
    # Diffusion function
    def diffuse(arr, N, dx):
        for i in range(N):
            arr = field.laplace(arr)  # math.fourier_laplace(arr, dx)
        return arr
    # Calculate Gradients
    dx_p, dy_p = field.gradient(phi).unstack('vector')
    # Get difference
    diff = (phi - plasma.density)
    # Step 2.1: New Omega.
    o = (PARAMS['c1'] * diff)
    if PARAMS['arakawa_coeff']:
        o += - PARAMS['arakawa_coeff'] * math._nd._periodic_2d_arakawa_poisson_bracket(phi.values, plasma.omega.values, plasma.dx)
    if PARAMS['nu'] and PARAMS['N']:
        o += PARAMS['nu'] * diffuse(plasma.omega, PARAMS['N'], plasma.dx)
    # Step 2.2: New Density.
    n = (PARAMS['c1'] * diff)
    if PARAMS['arakawa_coeff']:
        n += - PARAMS['arakawa_coeff'] * math._nd._periodic_2d_arakawa_poisson_bracket(phi.values, plasma.density.values, plasma.dx)
    if PARAMS['kappa_coeff']:
        n += - PARAMS['kappa_coeff'] * dy_p
    if PARAMS['nu']:
        n += +PARAMS['nu'] * diffuse(plasma.density, PARAMS['N'], plasma.dx)
    return Namespace(
        density=n,
        omega=o,
        phi=phi,  # NOTE: NOT A GRADIENT
        age=plasma.age + dt,
        dx=plasma.dx
    )


def euler_step(dt, gradient_func=step_gradient_2d, **kwargs):
    """Euler Step"""
    yn = Namespace(**kwargs)  # given dict to Namespace
    pn = get_phi(yn, guess=yn.phi)
    k1 = gradient_func(yn, pn, dt=dt)
    y1 = yn + dt * k1
    p1 = get_phi(y1, guess=yn.phi)
    return Namespace(
        density=y1.density,
        omega=y1.omega,
        phi=p1,
        age=yn.age + dt,  # y1 contains 2 time steps from compute
        dx=yn.dx
    )


def rk4_step(dt, gradient_func=step_gradient_2d, **kwargs):
    # RK4
    yn = Namespace(**kwargs)  # given dict to Namespace
    t0 = time.time()
    if yn.age == 0:
        pn = get_phi(yn, guess=yn.phi)
    else:
        pn = yn.phi
    k1 = dt * gradient_func(yn, pn, dt=0)
    p1 = get_phi(yn + k1 * 0.5)  # , guess=pn)
    k2 = dt * gradient_func(yn + k1 * 0.5, p1, dt=dt / 2)
    p2 = get_phi(yn + k2 * 0.5)  # , guess=pn+p1*0.5)
    k3 = dt * gradient_func(yn + k2 * 0.5, p2, dt=dt / 2)
    p3 = get_phi(yn + k3)  # , guess=pn+p2*0.5)
    k4 = dt * gradient_func(yn + k3, p3, dt=dt)
    y1 = yn + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    # phi = #, guess=pn+p3*0.5)
    t1 = time.time()
    if DEBUG:
        print(" | ".join([
            f"{yn.age + dt:<7.04g}",
            f"{np.max(np.abs(yn.density.data)):>7.02g}",
            f"{np.max(np.abs(k1.density.data)):>7.02g}",
            f"{np.max(np.abs(k2.density.data)):>7.02g}",
            f"{np.max(np.abs(k3.density.data)):>7.02g}",
            f"{np.max(np.abs(k4.density.data)):>7.02g}",
            f"{t1-t0:>6.02f}s"
        ]))
    return Namespace(
        density=y1.density,
        omega=y1.omega,
        phi=get_phi(y1),  # TODO: Somehow this does not work properly
        age=yn.age + dt,  # y1 contains 2 time steps from compute
        dx=yn.dx
    )


app = App("Hasegawa-Wakatani", "Simple plasma flow model.", framerate=10, dt=EditableFloat('dt', dt))
app.set_state(state, step_function=rk4_step, show=['density', 'omega', 'phi'])
app.prepare()
show(app, display=('density', 'omega', 'phi'))
